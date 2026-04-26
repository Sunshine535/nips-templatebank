#!/usr/bin/env python3
"""Evaluate one ablation variant on GSM8K test set.

For dataflow_plan variants: parses DataflowPlan JSON, executes via
DataflowExecutor, checks answer. For flat variants: parses Program
JSON, executes via Executor.

Usage:
  python3 scripts/eval_ablation.py \
      --variant full_gift_step \
      --seed 42 \
      --checkpoint results/gift_ablation/full_gift_step/seed42/model_final \
      --library results/gift_step/library_gift.json \
      --max_samples 200 \
      --output results/gift_ablation/full_gift_step/seed42/eval_results.json
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.dataflow_plan import (
    BindingRef,
    ConsistencyExecutor,
    DataflowExecutor,
    DataflowPlan,
    ValueAnnotatedDataflowPlan,
)
from src.template_dsl import Executor, Program, SubroutineLibrary

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def extract_numbers(text):
    numbers = re.findall(r"[\d,]+\.?\d*", text)
    result = []
    for n in numbers:
        cleaned = n.replace(",", "").strip()
        if cleaned and cleaned != ".":
            try:
                val = float(cleaned)
                result.append(val if "." in cleaned else int(float(cleaned)))
            except ValueError:
                continue
    return result


def extract_quantities(text):
    numbers = re.findall(r"[\d,]+\.?\d*", text)
    q = {}
    for i, n in enumerate(numbers):
        cleaned = n.replace(",", "").strip()
        if cleaned and cleaned != ".":
            try:
                val = float(cleaned)
                q[f"q{i}"] = val if "." in cleaned else int(float(cleaned))
            except ValueError:
                continue
    return q


def answer_matches(result, answer):
    try:
        r = float(str(result).replace(",", "").strip())
        a = float(str(answer).replace(",", "").strip())
        if a == 0:
            return abs(r) < 1e-3
        return abs(r - a) / max(abs(a), 1e-8) < 0.01
    except (ValueError, TypeError):
        return str(result).strip() == str(answer).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True,
                        choices=["old_fragment_only", "flat_matched_565",
                                 "gift_no_call_output", "gift_no_active_gate",
                                 "gift_no_explicit_refs_oracle_values",
                                 "full_gift_step",
                                 "vgift_full", "vgift_no_value_hints",
                                 "vgift_no_consistency", "vgift_value_only"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--library",
                        default="results/gift_step/library_gift.json",
                        help="Library for dataflow variants")
    parser.add_argument("--base_model", default="/root/assets/models/Qwen3.5-9B")
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--output", required=True)
    parser.add_argument("--predictions_out", default=None,
                        help="Per-example JSONL log path")
    parser.add_argument("--mechanism_log_out", default=None,
                        help="V-GIFT mechanism diagnostics JSON path")
    args = parser.parse_args()

    is_dataflow = args.variant not in ("old_fragment_only", "flat_matched_565")
    is_vgift = args.variant.startswith("vgift_")
    use_consistency = args.variant in ("vgift_full",)
    use_value_hints = args.variant in ("vgift_full", "vgift_no_consistency", "vgift_value_only")

    ds = load_dataset("openai/gsm8k", "main", split="test")
    test_data = list(ds)[:args.max_samples]
    logger.info("GSM8K test: %d samples", len(test_data))

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16,
        device_map={"": 0}, trust_remote_code=True,
    )
    if os.path.exists(os.path.join(args.checkpoint, "adapter_config.json")):
        model = PeftModel.from_pretrained(model, args.checkpoint)
        logger.info("Loaded LoRA from %s", args.checkpoint)
    else:
        logger.error("No adapter_config.json at %s", args.checkpoint)
        sys.exit(2)
    model.eval()

    flat_executor = Executor()
    library = None
    df_executor = None
    consistency_executor = None
    if is_dataflow and os.path.exists(args.library):
        library = SubroutineLibrary.load(args.library)
        df_executor = DataflowExecutor(library)
        if use_value_hints or use_consistency:
            consistency_executor = ConsistencyExecutor(library)
        logger.info("Loaded library: %d subs", library.size)

    correct = 0
    parsed = 0
    executable = 0
    total = 0
    predictions_log = []
    mech = {
        "value_hints_present": 0,
        "consistency_checked": 0,
        "consistency_passed": 0,
        "symbolic_correct": 0,
        "valuehint_correct": 0,
        "consistency_gate_correct": 0,
    }

    for i, item in enumerate(test_data):
        total += 1
        problem = item["question"]
        answer = item["answer"].split("####")[-1].strip()

        log_entry = {
            "idx": i, "question": problem, "gold": answer,
            "raw_response": None, "parsed": False, "parsed_obj": None,
            "exec_ok": False, "exec_result": None, "correct": False,
            "error": None,
        }

        if is_dataflow:
            prompt = f"Problem: {problem}\n\nGenerate a dataflow composition plan (JSON):"
        else:
            prompt = f"Problem: {problem}\n\nGenerate an executable JSON program:"

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=1024).to(model.device)

        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        response = tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
        )
        log_entry["raw_response"] = response[:2000]

        try:
            json_match = re.search(r"\{[\s\S]*\}", response)
            if json_match is None:
                log_entry["error"] = "no_json_match"
                predictions_log.append(log_entry)
                continue
            obj = json.loads(json_match.group())
            log_entry["parsed_obj"] = obj
        except Exception as e:
            log_entry["error"] = f"json_parse_error: {type(e).__name__}: {str(e)[:100]}"
            predictions_log.append(log_entry)
            continue

        ok = False
        result = None
        if is_dataflow:
            try:
                if "calls" not in obj:
                    log_entry["error"] = "missing_calls"
                    predictions_log.append(log_entry)
                    continue

                quantities = extract_quantities(problem)

                if use_value_hints and consistency_executor is not None:
                    plan = ValueAnnotatedDataflowPlan.from_dict(obj)
                    parsed += 1
                    log_entry["parsed"] = True
                    cr = consistency_executor.execute(plan, quantities)
                    log_entry["consistency_result"] = {
                        "symbolic_ok": cr.get("symbolic_exec_ok"),
                        "symbolic_result": str(cr.get("symbolic_result")),
                        "value_hint_result": str(cr.get("value_hint_result")),
                        "agreement": cr.get("final_agreement"),
                        "consistency_errors": cr.get("consistency_errors"),
                        "hints_present": cr.get("value_hints_present"),
                        "hints_consistent": cr.get("value_hints_consistent"),
                    }
                    if cr.get("value_hints_present", 0) > 0:
                        mech["value_hints_present"] += 1
                    sym_result = cr.get("symbolic_result")
                    vh_result = cr.get("value_hint_result")
                    if use_consistency:
                        mech["consistency_checked"] += 1
                        if cr.get("final_agreement"):
                            mech["consistency_passed"] += 1
                            ok = cr.get("symbolic_exec_ok", False)
                            result = sym_result
                        else:
                            ok = False
                            result = None
                    else:
                        ok = cr.get("symbolic_exec_ok", False)
                        result = sym_result

                    if sym_result is not None and answer_matches(sym_result, answer):
                        mech["symbolic_correct"] += 1
                    if vh_result is not None and answer_matches(vh_result, answer):
                        mech["valuehint_correct"] += 1
                    if use_consistency and cr.get("final_agreement") and sym_result is not None and answer_matches(sym_result, answer):
                        mech["consistency_gate_correct"] += 1
                else:
                    plan = DataflowPlan.from_dict(obj)
                    parsed += 1
                    log_entry["parsed"] = True
                    ok, result, _ = df_executor.execute_with_quantities(plan, quantities)
            except Exception as e:
                log_entry["error"] = f"dataflow_exec_error: {type(e).__name__}: {str(e)[:100]}"
                predictions_log.append(log_entry)
                continue
        else:
            try:
                prog = Program.from_dict(obj)
                parsed += 1
                log_entry["parsed"] = True
                numbers = extract_numbers(problem)
                bindings = {
                    slot.name: numbers[k] if k < len(numbers) else 0
                    for k, slot in enumerate(prog.slots)
                }
                ok, result, _ = flat_executor.execute(prog, bindings)
            except Exception as e:
                log_entry["error"] = f"flat_exec_error: {type(e).__name__}: {str(e)[:100]}"
                predictions_log.append(log_entry)
                continue

        log_entry["exec_ok"] = bool(ok)
        log_entry["exec_result"] = str(result) if result is not None else None
        if ok and result is not None:
            executable += 1
            if answer_matches(result, answer):
                correct += 1
                log_entry["correct"] = True
        predictions_log.append(log_entry)

        if (i + 1) % 50 == 0:
            logger.info(
                "[%s] %d/%d: correct=%d, parsed=%d, exec=%d",
                args.variant, i + 1, len(test_data),
                correct, parsed, executable,
            )
            sys.stdout.flush()

    results = {
        "variant": args.variant,
        "seed": args.seed,
        "checkpoint": args.checkpoint,
        "dataset": "gsm8k_test",
        "samples": total,
        "correct": correct,
        "accuracy": correct / max(total, 1),
        "parsed": parsed,
        "parse_rate": parsed / max(total, 1),
        "executable": executable,
        "exec_rate": executable / max(total, 1),
    }

    logger.info("=" * 60)
    logger.info(
        "  %s seed=%d : %d/%d (%.1f%%), parse=%.0f%%, exec=%.0f%%",
        args.variant, args.seed, correct, total,
        100 * results["accuracy"],
        100 * results["parse_rate"],
        100 * results["exec_rate"],
    )
    logger.info("=" * 60)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved: %s", args.output)

    pred_path = args.predictions_out or args.output.replace(
        "eval_results.json", "predictions.jsonl",
    )
    if pred_path != args.output:
        os.makedirs(os.path.dirname(pred_path) or ".", exist_ok=True)
        with open(pred_path, "w") as f:
            for entry in predictions_log:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.info("Saved predictions: %s (%d entries)", pred_path, len(predictions_log))

    if args.mechanism_log_out or is_vgift:
        import hashlib, subprocess
        mech_path = args.mechanism_log_out or args.output.replace(
            "eval_results.json", "mechanism_log.json",
        )
        try:
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
            ).decode().strip()[:10]
        except Exception:
            git_commit = "unknown"
        mechanism_log = {
            "mechanism_activation_indicator": (
                mech["value_hints_present"] > 0 or not use_value_hints
            ),
            "variant": args.variant,
            "seed": args.seed,
            "checkpoint_path": args.checkpoint,
            "git_commit": git_commit,
            "value_hint_present_rate": mech["value_hints_present"] / max(total, 1),
            "call_output_ref_present_rate": parsed / max(total, 1),
            "symbolic_exec_rate": executable / max(total, 1),
            "symbolic_value_agreement_rate": (
                mech["consistency_passed"] / max(mech["consistency_checked"], 1)
                if mech["consistency_checked"] > 0 else None
            ),
            "consistency_pass_rate": (
                mech["consistency_passed"] / max(mech["consistency_checked"], 1)
                if mech["consistency_checked"] > 0 else None
            ),
            "final_accuracy_symbolic_only": mech["symbolic_correct"] / max(total, 1),
            "final_accuracy_value_hint_only": mech["valuehint_correct"] / max(total, 1),
            "final_accuracy_consistency_gate": mech["consistency_gate_correct"] / max(total, 1),
            "parse_rate": parsed / max(total, 1),
            "exec_rate": executable / max(total, 1),
            "fallback_used": False,
            "number_of_predictions": total,
            "use_value_hints": use_value_hints,
            "use_consistency": use_consistency,
        }
        os.makedirs(os.path.dirname(mech_path) or ".", exist_ok=True)
        with open(mech_path, "w") as f:
            json.dump(mechanism_log, f, indent=2)
        logger.info("Mechanism log: %s", mech_path)
        logger.info("  value_hint_present_rate: %.1f%%", 100 * mechanism_log["value_hint_present_rate"])
        logger.info("  symbolic_correct: %d, valuehint_correct: %d, gate_correct: %d",
                     mech["symbolic_correct"], mech["valuehint_correct"], mech["consistency_gate_correct"])


if __name__ == "__main__":
    main()
