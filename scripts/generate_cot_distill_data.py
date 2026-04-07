#!/usr/bin/env python3
"""Generate CoT distillation data from teacher model.

For each TRAIN problem in the MCD split, generates teacher CoT solutions,
filters to correct-answer-only traces, and outputs training data for
the CoT distillation baseline.

Output format: [{"problem": str, "cot": str, "answer": str}]
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

import torch
from datasets import concatenate_datasets, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

COT_PROMPT = "Solve this math problem step by step.\n\nProblem: {problem}\n\nSolution:"


def extract_gold_answer(solution: str, dataset: str) -> str:
    """Extract gold answer from dataset solution field."""
    if "####" in solution:
        return solution.split("####")[-1].strip().replace(",", "")
    if dataset == "math":
        boxed = re.findall(r"\\boxed\{([^}]*)\}", solution)
        if boxed:
            return boxed[-1].strip()
    nums = re.findall(r"[\-\d,]+\.?\d*", solution)
    if nums:
        return nums[-1].replace(",", "")
    return ""


def extract_predicted_answer(text: str) -> str | None:
    """Extract predicted answer from model CoT output."""
    # GSM8K-style
    m = re.search(r"####\s*([\-\d,]+\.?\d*)", text)
    if m:
        return m.group(1).replace(",", "").strip()
    # MATH-style
    m = re.search(r"\\boxed\{([^}]+)\}", text)
    if m:
        return m.group(1).strip()
    # "the answer is" pattern
    m = re.search(r"(?:the answer is|answer:)\s*\$?\s*([\-\d,]+\.?\d*)", text, re.IGNORECASE)
    if m:
        return m.group(1).replace(",", "").strip()
    # Trailing number
    m = re.search(r"([\-\d,]+\.?\d*)\s*$", text.strip())
    if m:
        return m.group(1).replace(",", "").strip()
    return None


def check_answer(pred: str | None, gold: str) -> bool:
    if pred is None or gold == "":
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-3
    except (ValueError, TypeError):
        return pred.strip().lower() == gold.strip().lower()


def load_problems(dataset: str, split_path: str, config_path: str) -> list[dict]:
    """Load train-split problems from the MCD split."""
    import yaml

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Load MCD split to get train indices
    with open(split_path) as f:
        mcd_split = json.load(f)

    train_indices = set()
    train_partition = mcd_split.get("train", [])
    if isinstance(train_partition, list):
        if train_partition and isinstance(train_partition[0], int):
            train_indices = set(train_partition)
        elif train_partition and isinstance(train_partition[0], dict):
            for entry in train_partition:
                if "index" in entry:
                    train_indices.add(entry["index"])
                elif "id" in entry:
                    train_indices.add(entry["id"])

    # Load raw dataset
    ds_cfg = config["datasets"][dataset]
    if dataset == "math":
        subsets = ds_cfg.get("subsets")
        if subsets:
            parts = [load_dataset(ds_cfg["dataset_id"], s, split=ds_cfg["train_split"]) for s in subsets]
            ds = concatenate_datasets(parts)
        else:
            ds = load_dataset(ds_cfg["dataset_id"], split=ds_cfg["train_split"])
    else:
        subset = ds_cfg.get("subset")
        if subset:
            ds = load_dataset(ds_cfg["dataset_id"], subset, split=ds_cfg["train_split"])
        else:
            ds = load_dataset(ds_cfg["dataset_id"], split=ds_cfg["train_split"])

    max_train = ds_cfg.get("max_train", len(ds))
    if len(ds) > max_train:
        ds = ds.shuffle(seed=42).select(range(max_train))

    # Build problem list, filtering to train indices if available
    problems = []
    for idx, ex in enumerate(ds):
        if train_indices and idx not in train_indices:
            continue
        problem = ex.get("question", ex.get("problem", ""))
        solution = str(ex.get("answer", ex.get("solution", "")))
        gold = extract_gold_answer(solution, dataset)
        if problem and gold:
            problems.append({"problem": problem, "gold_answer": gold, "index": idx})

    # If no MCD split filtering was applied (empty train_indices), use all
    if not train_indices:
        logger.warning("MCD split train partition empty or not parseable; using all %d problems", len(problems))

    logger.info("Loaded %d train problems for dataset=%s", len(problems), dataset)
    return problems


def generate_cot_traces(
    problems: list[dict],
    model,
    tokenizer,
    samples_per_example: int,
    max_new_tokens: int,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> list[dict]:
    """Generate CoT traces and filter to correct-answer-only."""
    results = []
    total_generated = 0
    total_correct = 0

    for i, item in enumerate(problems):
        prompt = COT_PROMPT.format(problem=item["problem"][:1024])
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)

        correct_traces = []
        for k in range(samples_per_example):
            with torch.no_grad():
                if k == 0:
                    output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
                else:
                    output = model.generate(
                        **inputs, max_new_tokens=max_new_tokens,
                        do_sample=True, temperature=temperature, top_p=top_p,
                    )
            response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            total_generated += 1

            pred = extract_predicted_answer(response)
            if check_answer(pred, item["gold_answer"]):
                correct_traces.append(response)
                total_correct += 1

        # Keep shortest correct trace
        if correct_traces:
            best = min(correct_traces, key=len)
            results.append({
                "problem": item["problem"],
                "cot": best,
                "answer": item["gold_answer"],
            })

        if (i + 1) % 100 == 0:
            logger.info(
                "  %d/%d problems | generated=%d correct=%d kept=%d",
                i + 1, len(problems), total_generated, total_correct, len(results),
            )

    logger.info(
        "Done: %d/%d problems with correct traces (generated=%d, correct=%d)",
        len(results), len(problems), total_generated, total_correct,
    )
    return results


def main():
    parser = argparse.ArgumentParser(description="Generate CoT distillation data from teacher")
    parser.add_argument("--teacher_model", type=str, required=True, help="Teacher model name/path")
    parser.add_argument("--dataset", type=str, required=True, choices=["gsm8k", "math"])
    parser.add_argument("--split", type=str, required=True, help="Path to MCD split JSON")
    parser.add_argument("--output", type=str, required=True, help="Output JSON path")
    parser.add_argument("--samples_per_example", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--config", type=str, default=str(Path(__file__).resolve().parent.parent / "configs" / "template_config.yaml"))
    args = parser.parse_args()

    problems = load_problems(args.dataset, args.split, args.config)

    logger.info("Loading teacher model: %s", args.teacher_model)
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    model.eval()

    results = generate_cot_traces(
        problems, model, tokenizer,
        samples_per_example=args.samples_per_example,
        max_new_tokens=args.max_new_tokens,
    )

    del model
    torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Saved %d distillation examples to %s", len(results), args.output)


if __name__ == "__main__":
    main()
