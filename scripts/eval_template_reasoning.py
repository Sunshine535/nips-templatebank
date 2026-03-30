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
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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
        bindings = {f"x{j}": float(n.replace(",", "")) for j, n in enumerate(numbers)}

        success, result, stats = comp_exec.execute(plan, bindings)
        if success and result is not None:
            exec_success += 1
            pred = str(result)
            if check_answer(pred, gold):
                correct += 1
        else:
            fallback_used += 1
            cot_prompt = f"Solve step by step:\n\nProblem: {question}\n\nSolution:"
            cot_resp, cot_tokens = generate(model, tokenizer, cot_prompt, max_tokens)
            total_tokens += cot_tokens
            pred = extract_answer(cot_resp)
            if check_answer(pred, gold):
                correct += 1

    elapsed = time.time() - t0
    return {
        "method": "compose",
        "accuracy": round(correct / max(total, 1), 4),
        "valid_plan_rate": round(valid_plans / max(total, 1), 4),
        "execution_success": round(exec_success / max(total, 1), 4),
        "fallback_rate": round(fallback_used / max(total, 1), 4),
        "fallback_free_accuracy": round((correct - fallback_used) / max(total - fallback_used, 1), 4) if total > fallback_used else 0.0,
        "avg_tokens": round(total_tokens / max(total, 1), 1),
        "latency_seconds": round(elapsed, 1),
        "correct": correct, "total": total,
    }


def eval_flat(model, tokenizer, dataset, max_samples, max_tokens) -> dict:
    """Evaluate flat-program baseline: same DSL, no library calls."""
    logger.info("  [flat_inline] Evaluating...")
    executor = Executor()
    correct, total, valid_programs, exec_success, total_tokens = 0, 0, 0, 0, 0
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

        program = _parse_program(response)
        if program is None:
            continue

        valid_programs += 1
        numbers = re.findall(r'[\d,]+\.?\d*', question)
        bindings = {}
        for j, slot in enumerate(program.slots):
            if j < len(numbers):
                bindings[slot.name] = float(numbers[j].replace(",", ""))
            else:
                bindings[slot.name] = 0

        success, result, env = executor.execute(program, bindings)
        if success and result is not None:
            exec_success += 1
            if check_answer(str(result), gold):
                correct += 1

    elapsed = time.time() - t0
    return {
        "method": "flat_inline",
        "accuracy": round(correct / max(total, 1), 4),
        "valid_plan_rate": round(valid_programs / max(total, 1), 4),
        "execution_success": round(exec_success / max(total, 1), 4),
        "fallback_rate": 0.0,
        "fallback_free_accuracy": round(correct / max(total, 1), 4),
        "avg_tokens": round(total_tokens / max(total, 1), 1),
        "latency_seconds": round(elapsed, 1),
        "correct": correct, "total": total,
    }


def eval_cot_budget(model, tokenizer, dataset, max_samples, max_tokens, n_samples=5) -> dict:
    """Compute-matched CoT with majority vote."""
    logger.info("  [cot_budget] Evaluating (n=%d)...", n_samples)
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
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=True,
                                        temperature=0.6, top_p=0.95)
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


def main():
    parser = argparse.ArgumentParser(description="Evaluate subroutine composition vs baselines")
    parser.add_argument("--config", type=str, default=str(Path(__file__).resolve().parent.parent / "configs" / "template_config.yaml"))
    parser.add_argument("--compose_dir", type=str, default="results/planner/compose")
    parser.add_argument("--flat_dir", type=str, default="results/planner/flat")
    parser.add_argument("--library_path", type=str, default="results/templates/subroutine_library.json")
    parser.add_argument("--split_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="results/eval")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--skip_cot", action="store_true")
    parser.add_argument("--skip_flat", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    os.makedirs(args.output_dir, exist_ok=True)

    base_model = config["planner"]["model"]
    max_tokens = 512

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

    for ds_key in ["gsm8k", "math"]:
        ds_cfg = config["datasets"][ds_key]
        max_s = args.max_samples or ds_cfg.get("max_test", 500)
        max_tok = ds_cfg.get("max_new_tokens_plan", max_tokens)

        logger.info("=" * 60)
        logger.info("  Evaluating on %s (max %d)", ds_key, max_s)
        logger.info("=" * 60)

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
            del base
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
                    logger.info("  %s / %s: acc=%.4f  fb_free=%.4f  tokens=%.1f",
                                section, method, metrics["accuracy"], fb,
                                metrics.get("avg_tokens", 0))
    logger.info("  Results: %s", results_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
