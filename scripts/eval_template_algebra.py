#!/usr/bin/env python3
"""Evaluate template compiler: token efficiency, compositional generalization."""

import argparse
import json
import logging
import math
import os
import re
import sys

import torch
import yaml
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.template_algebra import TemplateAlgebra, TemplateBank


def extract_answer(text: str) -> str:
    """Extract numerical answer from model output."""
    patterns = [
        r"(?:the answer is|answer:|=)\s*\$?\\?boxed\{([^}]+)\}",
        r"(?:the answer is|answer:|=)\s*\$?\s*([\d,]+\.?\d*)",
        r"####\s*([\d,]+\.?\d*)",
        r"(\d[\d,]*\.?\d*)\s*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            return match.group(1).replace(",", "").strip()
    return text.strip().split("\n")[-1].strip()


def evaluate_accuracy(model, tokenizer, dataset, max_samples: int = 1000, max_new_tokens: int = 1024) -> dict:
    """Evaluate accuracy on math benchmarks."""
    correct = 0
    total = 0
    total_tokens = 0

    for i, ex in enumerate(dataset):
        if i >= max_samples:
            break

        question = ex.get("question", ex.get("problem", ""))
        gold_answer = ex.get("answer", ex.get("solution", ""))
        gold_num = extract_answer(str(gold_answer))

        prompt = (
            f"<|im_start|>system\nYou are a reasoning template compiler. "
            f"Select the best template and solve step by step.<|im_end|>\n"
            f"<|im_start|>user\nSolve: {question}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)

        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

        response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred_num = extract_answer(response)
        tokens_used = output.shape[1] - inputs["input_ids"].shape[1]
        total_tokens += tokens_used

        try:
            is_correct = abs(float(pred_num) - float(gold_num)) < 1e-3
        except (ValueError, TypeError):
            is_correct = pred_num.strip() == gold_num.strip()

        if is_correct:
            correct += 1
        total += 1

        if i < 5:
            logger.info("  Q: %s", question[:80])
            logger.info("  Gold: %s | Pred: %s | Correct: %s | Tokens: %d", gold_num, pred_num, is_correct, tokens_used)

    return {
        "accuracy": correct / max(total, 1),
        "correct": correct,
        "total": total,
        "avg_tokens": total_tokens / max(total, 1),
    }


def evaluate_compositional_generalization(model, tokenizer, bank: TemplateBank, n_tests: int = 200) -> dict:
    """Test compositional generalization: compose templates to solve novel problems."""
    algebra = TemplateAlgebra()
    templates = bank.search(min_reuse=0)

    if len(templates) < 2:
        logger.warning("Need at least 2 templates for composition tests")
        return {"accuracy": 0.0, "num_tests": 0}

    correct = 0
    total = 0
    for i in range(min(n_tests, len(templates) * (len(templates) - 1) // 2)):
        t1 = templates[i % len(templates)]
        t2 = templates[(i + 1) % len(templates)]
        composed = algebra.compose(t1, t2, name=f"test_compose_{i}")
        template_prompt = composed.to_prompt()

        # Generate a test problem that needs the composed template
        a, b, c = (i * 7 + 3) % 50 + 1, (i * 11 + 5) % 50 + 1, (i * 13 + 7) % 10 + 1
        test_problem = f"A store has {a} items at ${b} each. After a {c}0% discount, what is the total cost?"
        expected = a * b * (100 - c * 10) / 100

        prompt = (
            f"<|im_start|>system\nUse the following template to solve the problem.\n"
            f"{template_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{test_problem}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = extract_answer(response)

        try:
            if abs(float(pred) - expected) < 1e-1:
                correct += 1
        except (ValueError, TypeError):
            pass
        total += 1

    return {"accuracy": correct / max(total, 1), "correct": correct, "total": total}


def main():
    parser = argparse.ArgumentParser(description="Evaluate template algebra")
    parser.add_argument("--config", type=str, default="configs/template_config.yaml")
    parser.add_argument("--compiler_dir", type=str, default="outputs/compiler")
    parser.add_argument("--template_bank", type=str, default="outputs/templates/template_bank.json")
    parser.add_argument("--output_dir", type=str, default="outputs/eval")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    base_model = config["extraction"]["student_model"]
    logger.info("Loading model: %s", base_model)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto",
    )

    # Load compiler adapter
    if os.path.exists(os.path.join(args.compiler_dir, "adapter_config.json")):
        logger.info("Loading compiler adapter from %s", args.compiler_dir)
        model = PeftModel.from_pretrained(model, args.compiler_dir)
    model.eval()

    # Load template bank
    bank = None
    if os.path.exists(args.template_bank):
        bank = TemplateBank.load(args.template_bank)
        logger.info("Loaded template bank: %d templates", len(bank.templates))

    all_results = {}

    # 1. Evaluate on test sets
    for ds_cfg in config["evaluation"]["test_datasets"]:
        name = ds_cfg["name"]
        logger.info("=== Evaluating on %s ===", name)
        try:
            subset = ds_cfg.get("subset")
            if subset:
                ds = load_dataset(ds_cfg["dataset_id"], subset, split=ds_cfg["split"])
            else:
                ds = load_dataset(ds_cfg["dataset_id"], split=ds_cfg["split"])
            max_s = ds_cfg.get("max_samples", 1000)
            if len(ds) > max_s:
                ds = ds.shuffle(seed=42).select(range(max_s))
            result = evaluate_accuracy(model, tokenizer, ds, max_samples=max_s)
            all_results[name] = result
            logger.info("  %s: Accuracy=%.4f, Avg tokens=%.1f", name, result["accuracy"], result["avg_tokens"])
        except Exception as e:
            logger.warning("Failed to evaluate %s: %s", name, e)

    # 2. Template algebra evaluation
    if bank and len(bank.templates) >= 2:
        logger.info("=== Evaluating compositional generalization ===")
        n_comp = config["evaluation"].get("num_compositional_tests", 200)
        comp_result = evaluate_compositional_generalization(model, tokenizer, bank, n_tests=n_comp)
        all_results["compositional_generalization"] = comp_result
        logger.info("  Compositional: Accuracy=%.4f (%d/%d)", comp_result["accuracy"],
                    comp_result["correct"], comp_result["total"])

    # 3. Template bank statistics
    if bank:
        all_results["template_bank_stats"] = bank.stats()

    output_path = os.path.join(args.output_dir, "template_eval_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info("\n=== EVALUATION SUMMARY ===")
    for name, res in all_results.items():
        if isinstance(res, dict) and "accuracy" in res:
            logger.info("  %s: Accuracy=%.4f", name, res["accuracy"])
    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
