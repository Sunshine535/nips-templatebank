#!/usr/bin/env python3
"""Full evaluation pipeline for Template Algebra reasoning.

Evaluates on:
- GSM8K test (1319 problems)
- MATH test (5000 problems)

Comparison methods:
1. Direct CoT (base model, no template guidance)
2. Template-guided CoT (our method: select template → fill → solve)
3. Few-shot templates (provide 3 template examples in context)

Metrics:
- Accuracy (exact match on final answer)
- Answer extraction rate
- Template match rate
- Token efficiency
- Compositional evaluation (problems requiring template composition)
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.template_algebra import TemplateAlgebra, TemplateBank

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)



def extract_answer(text: str) -> str | None:
    """Extract numerical answer from model output."""
    patterns = [
        r"####\s*([\-\d,]+\.?\d*)",
        r"(?:the answer is|answer:|answer is)\s*\$?\\?boxed\{([^}]+)\}",
        r"(?:the answer is|answer:|answer is)\s*\$?\s*([\-\d,]+\.?\d*)",
        r"\$\\boxed\{([^}]+)\}\$",
        r"\\boxed\{([^}]+)\}",
        r"=\s*([\-\d,]+\.?\d*)\s*$",
        r"([\-\d,]+\.?\d*)\s*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text.strip(), re.IGNORECASE | re.MULTILINE)
        if match:
            val = match.group(1).replace(",", "").strip()
            if val:
                return val
    return None


def check_answer(pred: str | None, gold: str) -> bool:
    """Check if predicted answer matches gold."""
    if pred is None:
        return False
    gold_clean = extract_answer(gold) or gold.strip()
    try:
        return abs(float(pred) - float(gold_clean)) < 1e-3
    except (ValueError, TypeError):
        return pred.strip().lower() == gold_clean.strip().lower()


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 1024) -> tuple[str, int]:
    """Generate response and return (text, num_tokens)."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    num_tokens = output.shape[1] - inputs["input_ids"].shape[1]
    return response.strip(), num_tokens


# ===== Method 1: Direct CoT =====

def evaluate_direct_cot(model, tokenizer, dataset, max_samples: int, max_tokens: int) -> dict:
    """Evaluate with direct chain-of-thought prompting."""
    logger.info("  Evaluating: Direct CoT")
    correct, total, extracted, total_tokens = 0, 0, 0, 0
    t0 = time.time()

    for i, ex in enumerate(dataset):
        if i >= max_samples:
            break
        question = ex.get("question", ex.get("problem", ""))
        gold = str(ex.get("answer", ex.get("solution", "")))

        prompt = f"Solve the following problem step by step.\n\nProblem: {question}\n\nLet's think step by step:"
        response, tokens = generate(model, tokenizer, prompt, max_tokens)
        total_tokens += tokens

        pred = extract_answer(response)
        if pred is not None:
            extracted += 1
        if check_answer(pred, gold):
            correct += 1
        total += 1

        if i < 3:
            logger.info("    Q: %s", question[:80])
            logger.info("    Gold: %s | Pred: %s | Correct: %s", extract_answer(gold), pred, check_answer(pred, gold))

    elapsed = time.time() - t0
    return {
        "method": "direct_cot",
        "accuracy": round(correct / max(total, 1), 4),
        "correct": correct,
        "total": total,
        "answer_extraction_rate": round(extracted / max(total, 1), 4),
        "avg_tokens": round(total_tokens / max(total, 1), 1),
        "time_seconds": round(elapsed, 1),
    }


# ===== Method 2: Template-Guided CoT =====

def evaluate_template_guided(model, tokenizer, dataset, bank: TemplateBank,
                              max_samples: int, max_tokens: int) -> dict:
    """Evaluate with template-guided reasoning."""
    logger.info("  Evaluating: Template-Guided CoT")
    templates = bank.search(min_reuse=0)
    if not templates:
        logger.warning("No templates available, skipping")
        return {"method": "template_guided", "accuracy": 0.0, "total": 0}

    correct, total, extracted, matched, total_tokens = 0, 0, 0, 0, 0
    t0 = time.time()

    for i, ex in enumerate(dataset):
        if i >= max_samples:
            break
        question = ex.get("question", ex.get("problem", ""))
        gold = str(ex.get("answer", ex.get("solution", "")))

        best_template = _match_template(question, templates)
        if best_template:
            matched += 1
            template_prompt = best_template.to_prompt()
            prompt = (
                f"Use the following reasoning template to solve the problem.\n\n"
                f"{template_prompt}\n\n"
                f"Problem: {question}\n\n"
                f"Apply the template step by step:"
            )
        else:
            prompt = f"Solve step by step:\n\nProblem: {question}\n\nSolution:"

        response, tokens = generate(model, tokenizer, prompt, max_tokens)
        total_tokens += tokens

        pred = extract_answer(response)
        if pred is not None:
            extracted += 1
        if check_answer(pred, gold):
            correct += 1
        total += 1

    elapsed = time.time() - t0
    return {
        "method": "template_guided",
        "accuracy": round(correct / max(total, 1), 4),
        "correct": correct,
        "total": total,
        "template_match_rate": round(matched / max(total, 1), 4),
        "answer_extraction_rate": round(extracted / max(total, 1), 4),
        "avg_tokens": round(total_tokens / max(total, 1), 1),
        "time_seconds": round(elapsed, 1),
    }


# ===== Method 3: Few-Shot Templates =====

def evaluate_fewshot_templates(model, tokenizer, dataset, bank: TemplateBank,
                                 max_samples: int, max_tokens: int) -> dict:
    """Evaluate with few-shot template examples in context."""
    logger.info("  Evaluating: Few-Shot Templates")
    templates = bank.search(min_reuse=0)

    fewshot_context = "Here are example reasoning templates:\n\n"
    for t in templates[:3]:
        fewshot_context += f"---\n{t.to_prompt()}\n---\n\n"

    correct, total, extracted, total_tokens = 0, 0, 0, 0
    t0 = time.time()

    for i, ex in enumerate(dataset):
        if i >= max_samples:
            break
        question = ex.get("question", ex.get("problem", ""))
        gold = str(ex.get("answer", ex.get("solution", "")))

        prompt = (
            f"{fewshot_context}"
            f"Now solve this problem by selecting and applying an appropriate template:\n\n"
            f"Problem: {question}\n\nSolution:"
        )
        response, tokens = generate(model, tokenizer, prompt, max_tokens)
        total_tokens += tokens

        pred = extract_answer(response)
        if pred is not None:
            extracted += 1
        if check_answer(pred, gold):
            correct += 1
        total += 1

    elapsed = time.time() - t0
    return {
        "method": "fewshot_templates",
        "accuracy": round(correct / max(total, 1), 4),
        "correct": correct,
        "total": total,
        "answer_extraction_rate": round(extracted / max(total, 1), 4),
        "avg_tokens": round(total_tokens / max(total, 1), 1),
        "time_seconds": round(elapsed, 1),
    }


# ===== Compositional Evaluation =====

def evaluate_compositional(model, tokenizer, bank: TemplateBank, n_tests: int = 200) -> dict:
    """Evaluate problems requiring template composition."""
    logger.info("  Evaluating: Compositional Generalization")
    algebra = TemplateAlgebra()
    templates = bank.search(min_reuse=0)
    if len(templates) < 2:
        return {"method": "compositional", "accuracy": 0.0, "total": 0}

    correct, total = 0, 0
    for i in range(min(n_tests, len(templates) * (len(templates) - 1) // 2)):
        t1 = templates[i % len(templates)]
        t2 = templates[(i + 1) % len(templates)]
        composed = algebra.compose(t1, t2, name=f"comp_{i}")

        a = (i * 7 + 3) % 50 + 1
        b = (i * 11 + 5) % 50 + 1
        c = (i * 13 + 7) % 10 + 1
        problem = f"A store has {a} items at ${b} each. After a {c}0% discount, what is the total cost?"
        expected = a * b * (100 - c * 10) / 100

        prompt = (
            f"Use this composed template to solve:\n\n{composed.to_prompt()}\n\n"
            f"Problem: {problem}\n\nSolution:"
        )
        response, _ = generate(model, tokenizer, prompt, max_new_tokens=512)
        pred = extract_answer(response)

        try:
            if pred and abs(float(pred) - expected) < 1.0:
                correct += 1
        except (ValueError, TypeError):
            pass
        total += 1

    return {
        "method": "compositional",
        "accuracy": round(correct / max(total, 1), 4),
        "correct": correct,
        "total": total,
    }


def _match_template(question: str, templates: list) -> object | None:
    """Simple keyword matching to find best template for a problem."""
    q_lower = question.lower()
    best, best_score = None, 0
    for t in templates:
        score = 0
        for step in t.steps:
            for word in re.findall(r'[a-z]+', step.expression.lower()):
                if len(word) > 3 and word in q_lower:
                    score += 1
        if t.domain in q_lower:
            score += 2
        if score > best_score:
            best_score = score
            best = t
    return best if best_score > 0 else None


def main():
    parser = argparse.ArgumentParser(description="Evaluate template-guided reasoning")
    parser.add_argument("--config", type=str, default=str(Path(__file__).resolve().parent.parent / "configs" / "template_config.yaml"))
    parser.add_argument("--compiler_dir", type=str, default="results/compiler/stage2_filling",
                        help="Directory with trained compiler adapter")
    parser.add_argument("--template_bank", type=str, default="results/templates/template_bank.json")
    parser.add_argument("--output_dir", type=str, default="results/eval")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--skip_direct_cot", action="store_true")
    parser.add_argument("--skip_fewshot", action="store_true")
    parser.add_argument("--skip_compositional", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    os.makedirs(args.output_dir, exist_ok=True)

    base_model = config["extraction"]["student_model"]
    max_tokens = config["evaluation"]["max_new_tokens"]

    # Load model
    logger.info("Loading model: %s", base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto",
    )

    compiler_adapter = os.path.join(args.compiler_dir, "adapter_config.json")
    if os.path.exists(compiler_adapter):
        logger.info("Loading compiler adapter from %s", args.compiler_dir)
        model = PeftModel.from_pretrained(model, args.compiler_dir)
    model.eval()

    # Load template bank
    bank = None
    if os.path.exists(args.template_bank):
        bank = TemplateBank.load(args.template_bank)
        logger.info("Loaded template bank: %d templates", len(bank.templates))
    else:
        logger.warning("Template bank not found at %s", args.template_bank)

    # Load test datasets
    all_results = {"meta": {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "base_model": base_model,
        "compiler_dir": args.compiler_dir,
        "has_compiler": os.path.exists(compiler_adapter),
        "num_templates": len(bank.templates) if bank else 0,
    }}

    for ds_cfg in config["evaluation"]["test_datasets"]:
        ds_name = ds_cfg["name"]
        max_s = args.max_samples or ds_cfg.get("max_samples", 1000)

        logger.info("=" * 50)
        logger.info("  Evaluating on %s (max %d samples)", ds_name, max_s)
        logger.info("=" * 50)

        try:
            subset = ds_cfg.get("subset")
            if subset:
                ds = load_dataset(ds_cfg["dataset_id"], subset, split=ds_cfg["split"])
            else:
                ds = load_dataset(ds_cfg["dataset_id"], split=ds_cfg["split"])
            if len(ds) > max_s:
                ds = ds.shuffle(seed=42).select(range(max_s))
        except Exception as e:
            logger.warning("Failed to load %s: %s", ds_name, e)
            continue

        ds_results = {}

        # Method 1: Direct CoT (reload base model without adapter)
        if not args.skip_direct_cot:
            base_model_obj = AutoModelForCausalLM.from_pretrained(
                base_model, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto",
            )
            base_model_obj.eval()
            ds_results["direct_cot"] = evaluate_direct_cot(base_model_obj, tokenizer, ds, max_s, max_tokens)
            del base_model_obj
            torch.cuda.empty_cache()

        # Method 2: Template-guided CoT (with compiler)
        if bank:
            ds_results["template_guided"] = evaluate_template_guided(model, tokenizer, ds, bank, max_s, max_tokens)

        # Method 3: Few-shot templates
        if not args.skip_fewshot and bank:
            ds_results["fewshot_templates"] = evaluate_fewshot_templates(model, tokenizer, ds, bank, max_s, max_tokens)

        all_results[ds_name] = ds_results

    # Compositional evaluation
    if not args.skip_compositional and bank and len(bank.templates) >= 2:
        n_comp = config["evaluation"].get("num_compositional_tests", 200)
        all_results["compositional"] = evaluate_compositional(model, tokenizer, bank, n_tests=n_comp)

    # Template bank stats
    if bank:
        all_results["template_bank_stats"] = bank.stats()

    # Save results
    results_path = os.path.join(args.output_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("  EVALUATION SUMMARY")
    logger.info("=" * 60)
    for section, data in all_results.items():
        if section == "meta" or section == "template_bank_stats":
            continue
        if isinstance(data, dict):
            for method, metrics in data.items():
                if isinstance(metrics, dict) and "accuracy" in metrics:
                    logger.info("  %s / %s: Acc=%.4f  Tokens=%.1f",
                                section, method, metrics["accuracy"], metrics.get("avg_tokens", 0))
    logger.info("  Results: %s", results_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
