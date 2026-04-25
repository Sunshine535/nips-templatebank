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
from src.dataflow_plan import BindingRef, DataflowExecutor, DataflowPlan
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
                                 "full_gift_step"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--library",
                        default="results/gift_step/library_gift.json",
                        help="Library for dataflow variants")
    parser.add_argument("--base_model", default="/root/assets/models/Qwen3.5-9B")
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--output", required=True)
    parser.add_argument("--predictions_out", default=None,
                        help="Per-example JSONL log path (predictions, gold, exec result, error)")
    args = parser.parse_args()

    is_dataflow = args.variant not in ("old_fragment_only", "flat_matched_565")

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
    if is_dataflow and os.path.exists(args.library):
        library = SubroutineLibrary.load(args.library)
        df_executor = DataflowExecutor(library)
        logger.info("Loaded library: %d subs", library.size)

    correct = 0
    parsed = 0
    executable = 0
    total = 0
    predictions_log = []

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
                plan = DataflowPlan.from_dict(obj)
                parsed += 1
                log_entry["parsed"] = True
                quantities = extract_quantities(problem)
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


if __name__ == "__main__":
    main()
