#!/usr/bin/env python3
"""Full evaluation pipeline for subroutine composition vs baselines.

Methods evaluated:
1. compose      — Our method: planner outputs composition plan using subroutine library
2. flat_inline   — Same DSL, no library calls (critical baseline)
3. cot_budget    — Compute-matched CoT with majority vote
4. retrieval_compose — Retrieval-conditioned planner

Metrics:
- accuracy, valid_plan_rate, execution_success, fallback_rate,
  fallback_free_accuracy, total_tokens, latency

Also computes oracle ceilings.
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.mcd_split import load_split
from src.template_dsl import (
    CompositionExecutor,
    CompositionPlan,
    Executor,
    Program,
    SubroutineLibrary,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def extract_answer(text: str) -> str | None:
    patterns = [
        r"####\s*([\-\d,]+\.?\d*)",
        r"\\boxed\{([^}]+)\}",
        r"(?:the answer is|answer:)\s*\$?\s*([\-\d,]+\.?\d*)",
        r"=\s*([\-\d,]+\.?\d*)\s*$",
        r"([\-\d,]+\.?\d*)\s*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text.strip(), re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).replace(",", "").strip()
    return None


def check_answer(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    gold_clean = extract_answer(gold) or gold.strip()
    try:
        return abs(float(pred) - float(gold_clean)) < 1e-3
    except (ValueError, TypeError):
        return pred.strip().lower() == gold_clean.strip().lower()


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 512) -> tuple[str, int]:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=3072).to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    n_tokens = output.shape[1] - inputs["input_ids"].shape[1]
    return response.strip(), n_tokens


def eval_compose(model, tokenizer, dataset, library, max_samples, max_tokens) -> dict:
    """Evaluate composition method: planner -> execute via library."""
    logger.info("  [compose] Evaluating...")
    comp_exec = CompositionExecutor(library)
    lib_sigs = "\n".join(library.signatures())

    correct, total, valid_plans, exec_success, fallback_used, total_tokens = 0, 0, 0, 0, 0, 0
    correct_no_fallback = 0
    valid_json_count = 0
    t0 = time.time()

    for i, ex in enumerate(dataset):
        if i >= max_samples:
            break
        question = ex.get("question", ex.get("problem", ""))
        gold = str(ex.get("answer", ex.get("solution", "")))

        prompt = (
            f"Available subroutines:\n{lib_sigs}\n\n"
            f"Problem: {question}\n\nGenerate a composition plan (JSON):"
        )
        response, tokens = generate(model, tokenizer, prompt, max_tokens)
        total_tokens += tokens
        total += 1

        # Waterfall: check JSON parse separately from plan parse
        json_match = re.search(r'\{[\s\S]*\}', response)
        is_valid_json = False
        if json_match:
            try:
                json.loads(json_match.group())
                is_valid_json = True
            except json.JSONDecodeError:
                pass
        if is_valid_json:
            valid_json_count += 1

        plan = _parse_plan(response)
        if plan is None:
            fallback_used += 1
            cot_prompt = f"Solve step by step:\n\nProblem: {question}\n\nSolution:"
            cot_resp, cot_tokens = generate(model, tokenizer, cot_prompt, max_tokens)
            total_tokens += cot_tokens
            pred = extract_answer(cot_resp)
            if check_answer(pred, gold):
                correct += 1
            continue

        valid_plans += 1
        numbers = re.findall(r'[\d,]+\.?\d*', question)
        bindings = {f"x{j}": (float(n.replace(",", "").strip()) if n.replace(",", "").strip() else 0) for j, n in enumerate(numbers)}

        success, result, stats = comp_exec.execute(plan, bindings)
        if success and result is not None:
            exec_success += 1
            pred = str(result)
            if check_answer(pred, gold):
                correct += 1
                correct_no_fallback += 1
        else:
            fallback_used += 1
            cot_prompt = f"Solve step by step:\n\nProblem: {question}\n\nSolution:"
            cot_resp, cot_tokens = generate(model, tokenizer, cot_prompt, max_tokens)
            total_tokens += cot_tokens
            pred = extract_answer(cot_resp)
            if check_answer(pred, gold):
                correct += 1

    elapsed = time.time() - t0
    non_fallback = total - fallback_used
    return {
        "method": "compose",
        "accuracy": round(correct / max(total, 1), 4),
        "valid_plan_rate": round(valid_plans / max(total, 1), 4),
        "execution_success": round(exec_success / max(total, 1), 4),
        "fallback_rate": round(fallback_used / max(total, 1), 4),
        "fallback_free_accuracy": round(correct_no_fallback / max(non_fallback, 1), 4) if non_fallback > 0 else 0.0,
        "avg_tokens": round(total_tokens / max(total, 1), 1),
        "latency_seconds": round(elapsed, 1),
        "correct": correct, "total": total,
        "waterfall": {
            "total": total,
            "valid_json": valid_json_count,
            "valid_plan": valid_plans,
            "executable": exec_success,
            "answer_correct": correct_no_fallback,
        },
    }


def eval_flat(model, tokenizer, dataset, max_samples, max_tokens) -> dict:
    """Evaluate flat-program baseline: same DSL, no library calls."""
    logger.info("  [flat_inline] Evaluating...")
    executor = Executor()
    correct, total, valid_programs, exec_success, fallback_used, total_tokens = 0, 0, 0, 0, 0, 0
    correct_no_fallback = 0
    valid_json_count = 0
    t0 = time.time()

    for i, ex in enumerate(dataset):
        if i >= max_samples:
            break
        question = ex.get("question", ex.get("problem", ""))
        gold = str(ex.get("answer", ex.get("solution", "")))

        prompt = f"Problem: {question}\n\nGenerate an executable program (JSON):"
        response, tokens = generate(model, tokenizer, prompt, max_tokens)
        total_tokens += tokens
        total += 1

        # Waterfall: check JSON parse separately from program parse
        json_match = re.search(r'\{[\s\S]*\}', response)
        is_valid_json = False
        if json_match:
            try:
                json.loads(json_match.group())
                is_valid_json = True
            except json.JSONDecodeError:
                pass
        if is_valid_json:
            valid_json_count += 1

        program = _parse_program(response)
        if program is None:
            fallback_used += 1
            cot_prompt = f"Solve step by step:\n\nProblem: {question}\n\nSolution:"
            cot_resp, cot_tokens = generate(model, tokenizer, cot_prompt, max_tokens)
            total_tokens += cot_tokens
            pred = extract_answer(cot_resp)
            if check_answer(pred, gold):
                correct += 1
            continue

        valid_programs += 1
        numbers = re.findall(r'[\d,]+\.?\d*', question)
        bindings = {}
        for j, slot in enumerate(program.slots):
            if j < len(numbers):
                cleaned = numbers[j].replace(",", "").strip()
                try:
                    bindings[slot.name] = float(cleaned) if cleaned else 0
                except ValueError:
                    bindings[slot.name] = 0
            else:
                bindings[slot.name] = 0

        success, result, env = executor.execute(program, bindings)
        if success and result is not None:
            exec_success += 1
            if check_answer(str(result), gold):
                correct += 1
                correct_no_fallback += 1
        else:
            fallback_used += 1
            cot_prompt = f"Solve step by step:\n\nProblem: {question}\n\nSolution:"
            cot_resp, cot_tokens = generate(model, tokenizer, cot_prompt, max_tokens)
            total_tokens += cot_tokens
            pred = extract_answer(cot_resp)
            if check_answer(pred, gold):
                correct += 1

    elapsed = time.time() - t0
    non_fallback = total - fallback_used
    return {
        "method": "flat_inline",
        "accuracy": round(correct / max(total, 1), 4),
        "valid_plan_rate": round(valid_programs / max(total, 1), 4),
        "execution_success": round(exec_success / max(total, 1), 4),
        "fallback_rate": round(fallback_used / max(total, 1), 4),
        "fallback_free_accuracy": round(correct_no_fallback / max(non_fallback, 1), 4) if non_fallback > 0 else 0.0,
        "avg_tokens": round(total_tokens / max(total, 1), 1),
        "latency_seconds": round(elapsed, 1),
        "correct": correct, "total": total,
        "waterfall": {
            "total": total,
            "valid_json": valid_json_count,
            "valid_program": valid_programs,
            "executable": exec_success,
            "answer_correct": correct_no_fallback,
        },
    }


def eval_cot_budget(model, tokenizer, dataset, max_samples, max_tokens, config_eval) -> dict:
    """Compute-matched CoT with majority vote. Uses evaluation.cot_budget config."""
    cot_cfg = config_eval.get("cot_budget", {})
    n_samples = config_eval.get("rerank_n", 3)
    temperature = cot_cfg.get("temperature", 0.6)
    top_p = cot_cfg.get("top_p", 0.95)
    vote_strategy = cot_cfg.get("vote", "majority_numeric")
    logger.info("  [cot_budget] Evaluating (n=%d, temp=%.2f, top_p=%.2f)...", n_samples, temperature, top_p)
    correct, total, total_tokens = 0, 0, 0
    t0 = time.time()

    for i, ex in enumerate(dataset):
        if i >= max_samples:
            break
        question = ex.get("question", ex.get("problem", ""))
        gold = str(ex.get("answer", ex.get("solution", "")))

        prompt = f"Solve the following problem step by step.\n\nProblem: {question}\n\nLet's think step by step:"

        answers = []
        for _ in range(n_samples):
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=True,
                                        temperature=temperature, top_p=top_p)
            response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            total_tokens += output.shape[1] - inputs["input_ids"].shape[1]
            pred = extract_answer(response)
            if pred:
                answers.append(pred)

        total += 1
        if answers:
            vote = Counter(answers).most_common(1)[0][0]
            if check_answer(vote, gold):
                correct += 1

    elapsed = time.time() - t0
    return {
        "method": "cot_budget",
        "accuracy": round(correct / max(total, 1), 4),
        "avg_tokens": round(total_tokens / max(total, 1), 1),
        "latency_seconds": round(elapsed, 1),
        "n_samples": n_samples,
        "temperature": temperature,
        "top_p": top_p,
        "vote_strategy": vote_strategy,
        "correct": correct, "total": total,
    }


def eval_direct_cot(model, tokenizer, dataset, max_samples, max_tokens) -> dict:
    """Direct CoT baseline (single pass)."""
    logger.info("  [direct_cot] Evaluating...")
    correct, total, total_tokens = 0, 0, 0
    t0 = time.time()

    for i, ex in enumerate(dataset):
        if i >= max_samples:
            break
        question = ex.get("question", ex.get("problem", ""))
        gold = str(ex.get("answer", ex.get("solution", "")))

        prompt = f"Solve step by step:\n\nProblem: {question}\n\nSolution:"
        response, tokens = generate(model, tokenizer, prompt, max_tokens)
        total_tokens += tokens
        total += 1

        pred = extract_answer(response)
        if check_answer(pred, gold):
            correct += 1

    elapsed = time.time() - t0
    return {
        "method": "direct_cot",
        "accuracy": round(correct / max(total, 1), 4),
        "avg_tokens": round(total_tokens / max(total, 1), 1),
        "latency_seconds": round(elapsed, 1),
        "correct": correct, "total": total,
    }


def _parse_plan(response: str) -> CompositionPlan | None:
    try:
        json_match = re.search(r'\{[\s\S]*\}', response)
        if not json_match:
            return None
        data = json.loads(json_match.group())
        if "plan" in data and isinstance(data["plan"], list):
            return CompositionPlan.from_dict(data)
        return None
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def _parse_program(response: str) -> Program | None:
    try:
        json_match = re.search(r'\{[\s\S]*\}', response)
        if not json_match:
            return None
        data = json.loads(json_match.group())
        return Program.from_dict(data)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def eval_retrieval_compose(model, tokenizer, dataset, library, max_samples, max_tokens, config_eval) -> dict:
    """Retrieval-conditioned planner: retrieve top-k subroutines, then compose.

    Uses evaluation.retrieval config for dense_weight, bm25_weight, top_k_funcs.
    Falls back to random-subset compose if sentence-transformers unavailable.
    Execution uses ONLY the retrieved sublibrary, not the full library.
    """
    retr_cfg = config_eval.get("retrieval", {})
    top_k_funcs = retr_cfg.get("top_k_funcs", 8)
    logger.info("  [retrieval_compose] Evaluating (top_k_funcs=%d)...", top_k_funcs)

    all_subs = list(library.subroutines.values())

    encoder = None
    try:
        from sentence_transformers import SentenceTransformer
        encoder_name = retr_cfg.get("encoder", "BAAI/bge-large-en-v1.5")
        encoder = SentenceTransformer(encoder_name)
        sub_texts = [f"{s.sub_id}: {' '.join(op.value for op in [st.op for st in s.program.steps])}" for s in all_subs]
        sub_embeddings = encoder.encode(sub_texts, normalize_embeddings=True)
        logger.info("  Loaded retrieval encoder: %s", encoder_name)
    except ImportError:
        logger.warning("  sentence-transformers not available; using random sub-selection fallback")
        sub_embeddings = None

    correct, total, valid_plans, exec_success, fallback_used, total_tokens = 0, 0, 0, 0, 0, 0
    correct_no_fallback = 0
    t0 = time.time()

    for i, ex in enumerate(dataset):
        if i >= max_samples:
            break
        question = ex.get("question", ex.get("problem", ""))
        gold = str(ex.get("answer", ex.get("solution", "")))

        if encoder is not None and sub_embeddings is not None:
            import numpy as _np
            q_emb = encoder.encode([question], normalize_embeddings=True)
            scores = (q_emb @ sub_embeddings.T).flatten()
            topk_idx = _np.argsort(-scores)[:top_k_funcs]
            selected = [all_subs[j] for j in topk_idx]
        else:
            import random as _rng
            _rng.seed(42 + i)
            k = min(top_k_funcs, len(all_subs))
            selected = _rng.sample(all_subs, k)

        sub_lib = SubroutineLibrary()
        for s in selected:
            sub_lib.add(s)
        lib_sigs = "\n".join(sub_lib.signatures())
        sub_exec = CompositionExecutor(sub_lib)

        prompt = (
            f"Available subroutines (retrieved):\n{lib_sigs}\n\n"
            f"Problem: {question}\n\nGenerate a composition plan (JSON):"
        )
        response, tokens = generate(model, tokenizer, prompt, max_tokens)
        total_tokens += tokens
        total += 1

        plan = _parse_plan(response)
        if plan is None:
            fallback_used += 1
            cot_prompt = f"Solve step by step:\n\nProblem: {question}\n\nSolution:"
            cot_resp, cot_tokens = generate(model, tokenizer, cot_prompt, max_tokens)
            total_tokens += cot_tokens
            pred = extract_answer(cot_resp)
            if check_answer(pred, gold):
                correct += 1
            continue

        valid_plans += 1
        numbers = re.findall(r'[\d,]+\.?\d*', question)
        bindings = {f"x{j}": (float(n.replace(",", "").strip()) if n.replace(",", "").strip() else 0) for j, n in enumerate(numbers)}

        success, result, stats = sub_exec.execute(plan, bindings)
        if success and result is not None:
            exec_success += 1
            if check_answer(str(result), gold):
                correct += 1
                correct_no_fallback += 1
        else:
            fallback_used += 1
            cot_prompt = f"Solve step by step:\n\nProblem: {question}\n\nSolution:"
            cot_resp, cot_tokens = generate(model, tokenizer, cot_prompt, max_tokens)
            total_tokens += cot_tokens
            pred = extract_answer(cot_resp)
            if check_answer(pred, gold):
                correct += 1

    elapsed = time.time() - t0
    non_fallback = total - fallback_used
    return {
        "method": "retrieval_compose",
        "accuracy": round(correct / max(total, 1), 4),
        "valid_plan_rate": round(valid_plans / max(total, 1), 4),
        "execution_success": round(exec_success / max(total, 1), 4),
        "fallback_rate": round(fallback_used / max(total, 1), 4),
        "fallback_free_accuracy": round(correct_no_fallback / max(non_fallback, 1), 4) if non_fallback > 0 else 0.0,
        "avg_tokens": round(total_tokens / max(total, 1), 1),
        "latency_seconds": round(elapsed, 1),
        "top_k_funcs": top_k_funcs,
        "correct": correct, "total": total,
    }


def run_binding_analysis(model, tokenizer, dataset, library, max_samples, max_tokens,
                         perturb_frac: float = 0.10, subset: int = 200) -> dict:
    """Binding sensitivity analysis: perturb each binding +/-10% and check if answer changes.

    Decomposes compose errors into: planner failures, binding failures (decorative
    bindings), and execution failures.  Reports % of active vs decorative bindings.
    """
    logger.info("  [binding_analysis] Running on %d examples (perturb=%.0f%%)...", subset, perturb_frac * 100)
    comp_exec = CompositionExecutor(library)
    lib_sigs = "\n".join(library.signatures())

    total_bindings, active_bindings = 0, 0
    planner_fail, exec_fail, answer_ok = 0, 0, 0
    analyzed = 0

    for i, ex in enumerate(dataset):
        if analyzed >= subset or i >= max_samples:
            break
        question = ex.get("question", ex.get("problem", ""))
        gold = str(ex.get("answer", ex.get("solution", "")))

        prompt = (
            f"Available subroutines:\n{lib_sigs}\n\n"
            f"Problem: {question}\n\nGenerate a composition plan (JSON):"
        )
        response, _ = generate(model, tokenizer, prompt, max_tokens)
        plan = _parse_plan(response)
        if plan is None:
            planner_fail += 1
            analyzed += 1
            continue

        numbers = re.findall(r'[\d,]+\.?\d*', question)
        bindings = {f"x{j}": (float(n.replace(",", "").strip()) if n.replace(",", "").strip() else 0) for j, n in enumerate(numbers)}
        success, base_result, _ = comp_exec.execute(plan, bindings)
        if not success or base_result is None:
            exec_fail += 1
            analyzed += 1
            continue

        if not check_answer(str(base_result), gold):
            analyzed += 1
            continue

        # This example succeeded — now perturb each binding
        answer_ok += 1
        for key, val in bindings.items():
            if val == 0:
                total_bindings += 1
                continue  # can't meaningfully perturb zero
            total_bindings += 1
            perturbed = dict(bindings)
            perturbed[key] = val * (1 + perturb_frac)
            ok_up, res_up, _ = comp_exec.execute(plan, perturbed)
            perturbed[key] = val * (1 - perturb_frac)
            ok_dn, res_dn, _ = comp_exec.execute(plan, perturbed)
            # Binding is "active" if perturbing it changes the output
            if (ok_up and res_up is not None and abs(float(res_up) - float(base_result)) > 1e-6) or \
               (ok_dn and res_dn is not None and abs(float(res_dn) - float(base_result)) > 1e-6):
                active_bindings += 1
        analyzed += 1

    active_pct = round(active_bindings / max(total_bindings, 1), 4)
    decorative_pct = round(1 - active_pct, 4)
    result = {
        "analyzed": analyzed,
        "planner_failures": planner_fail,
        "execution_failures": exec_fail,
        "correct_examples_probed": answer_ok,
        "total_bindings_probed": total_bindings,
        "active_bindings": active_bindings,
        "active_pct": active_pct,
        "decorative_pct": decorative_pct,
        "perturb_frac": perturb_frac,
    }
    logger.info("  [binding_analysis] active=%.1f%% decorative=%.1f%% (planner_fail=%d exec_fail=%d probed=%d)",
                active_pct * 100, decorative_pct * 100, planner_fail, exec_fail, answer_ok)
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate subroutine composition vs baselines")
    parser.add_argument("--config", type=str, default=str(Path(__file__).resolve().parent.parent / "configs" / "template_config.yaml"))
    parser.add_argument("--compose_dir", type=str, default="results/planner/compose")
    parser.add_argument("--flat_dir", type=str, default="results/planner/flat")
    parser.add_argument("--library_path", type=str, default="results/templates/subroutine_library.json")
    parser.add_argument("--programs_path", type=str, default="results/templates/all_programs.json")
    parser.add_argument("--split_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="results/eval")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--skip_cot", action="store_true")
    parser.add_argument("--skip_flat", action="store_true")
    parser.add_argument("--skip_retrieval", action="store_true")
    parser.add_argument("--binding_analysis", action="store_true",
                        help="Run binding sensitivity analysis on compose examples")
    parser.add_argument("--binding_subset", type=int, default=200,
                        help="Number of examples for binding analysis")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    os.makedirs(args.output_dir, exist_ok=True)

    base_model = config["planner"]["model"]
    max_tokens = 512
    config_eval = config.get("evaluation", {})

    library = None
    if os.path.exists(args.library_path):
        library = SubroutineLibrary.load(args.library_path)
        logger.info("Loaded library: %d subroutines", library.size)

    all_results = {"meta": {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "base_model": base_model,
        "seed": args.seed,
        "has_library": library is not None,
        "library_size": library.size if library else 0,
    }}

    mcd_test_data = None
    if args.split_path and os.path.exists(args.split_path):
        mcd_split_data = load_split(args.split_path)
        if os.path.exists(args.programs_path):
            with open(args.programs_path) as f:
                all_programs = json.load(f)
            test_indices = mcd_split_data.get("test", [])
            mcd_test_data = [all_programs[i] for i in test_indices if i < len(all_programs)]
            logger.info("Loaded MCD test split: %d examples from %s", len(mcd_test_data), args.programs_path)
        else:
            logger.warning("Programs file not found at %s, skipping MCD eval", args.programs_path)

    eval_keys = ["gsm8k", "math"]
    if mcd_test_data is not None:
        eval_keys.append("mcd_test")

    for ds_key in eval_keys:
        if ds_key == "mcd_test":
            ds = mcd_test_data
            max_s = args.max_samples or len(ds)
            max_tok = max_tokens
        else:
            ds_cfg = config["datasets"][ds_key]
            max_s = args.max_samples or ds_cfg.get("max_test", 500)
            max_tok = ds_cfg.get("max_new_tokens_plan", max_tokens)
            try:
                subset = ds_cfg.get("subset")
                if subset:
                    ds = load_dataset(ds_cfg["dataset_id"], subset, split=ds_cfg["test_split"])
                else:
                    ds = load_dataset(ds_cfg["dataset_id"], split=ds_cfg["test_split"])
                if len(ds) > max_s:
                    ds = ds.shuffle(seed=args.seed).select(range(max_s))
            except Exception as e:
                logger.warning("Failed to load %s: %s", ds_key, e)
                continue

        logger.info("=" * 60)
        logger.info("  Evaluating on %s (max %d)", ds_key, max_s)
        logger.info("=" * 60)

        ds_results = {}

        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Method 1: Compose (our method)
        if library:
            compose_model = AutoModelForCausalLM.from_pretrained(
                base_model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
            if os.path.exists(os.path.join(args.compose_dir, "adapter_config.json")):
                compose_model = PeftModel.from_pretrained(compose_model, args.compose_dir)
            compose_model.eval()
            ds_results["compose"] = eval_compose(compose_model, tokenizer, ds, library, max_s, max_tok)
            del compose_model
            torch.cuda.empty_cache()

        # Method 2: Flat inline (critical baseline)
        if not args.skip_flat:
            flat_model = AutoModelForCausalLM.from_pretrained(
                base_model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
            if os.path.exists(os.path.join(args.flat_dir, "adapter_config.json")):
                flat_model = PeftModel.from_pretrained(flat_model, args.flat_dir)
            flat_model.eval()
            ds_results["flat_inline"] = eval_flat(flat_model, tokenizer, ds, max_s, max_tok)
            del flat_model
            torch.cuda.empty_cache()

        # Method 3: Direct CoT
        if not args.skip_cot:
            base = AutoModelForCausalLM.from_pretrained(
                base_model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
            base.eval()
            ds_results["direct_cot"] = eval_direct_cot(base, tokenizer, ds, max_s, max_tok)

            # Method 4: CoT budget (majority vote, config-driven)
            cot_max_tok = max_tok
            if ds_key == "gsm8k":
                cot_max_tok = config_eval.get("cot_budget", {}).get("max_new_tokens_gsm8k", max_tok)
            elif ds_key == "math":
                cot_max_tok = config_eval.get("cot_budget", {}).get("max_new_tokens_math", max_tok)
            ds_results["cot_budget"] = eval_cot_budget(base, tokenizer, ds, max_s, cot_max_tok, config_eval)
            del base
            torch.cuda.empty_cache()

        # Method 5: Retrieval-conditioned compose
        if library and not args.skip_retrieval:
            retr_model = AutoModelForCausalLM.from_pretrained(
                base_model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
            if os.path.exists(os.path.join(args.compose_dir, "adapter_config.json")):
                retr_model = PeftModel.from_pretrained(retr_model, args.compose_dir)
            retr_model.eval()
            ds_results["retrieval_compose"] = eval_retrieval_compose(
                retr_model, tokenizer, ds, library, max_s, max_tok, config_eval)
            del retr_model
            torch.cuda.empty_cache()

        # Optional: binding sensitivity analysis
        if args.binding_analysis and library:
            ba_model = AutoModelForCausalLM.from_pretrained(
                base_model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
            if os.path.exists(os.path.join(args.compose_dir, "adapter_config.json")):
                ba_model = PeftModel.from_pretrained(ba_model, args.compose_dir)
            ba_model.eval()
            ds_results["binding_analysis"] = run_binding_analysis(
                ba_model, tokenizer, ds, library, max_s, max_tok,
                subset=args.binding_subset)
            del ba_model
            torch.cuda.empty_cache()

        all_results[ds_key] = ds_results

    if library:
        all_results["library_stats"] = library.stats()

    results_path = os.path.join(args.output_dir, f"eval_results_seed{args.seed}.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info("")
    logger.info("=" * 60)
    logger.info("  EVALUATION SUMMARY")
    logger.info("=" * 60)
    for section, data in all_results.items():
        if section in ("meta", "library_stats"):
            continue
        if isinstance(data, dict):
            for method, metrics in data.items():
                if isinstance(metrics, dict) and "accuracy" in metrics:
                    fb = metrics.get("fallback_free_accuracy", metrics["accuracy"])
                    fb_rate = metrics.get("fallback_rate", 0.0)
                    logger.info("  %s / %s: fb_free_acc=%.4f  acc=%.4f  fb_rate=%.4f  tokens=%.1f",
                                section, method, fb, metrics["accuracy"], fb_rate,
                                metrics.get("avg_tokens", 0))
    logger.info("  Results: %s", results_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
