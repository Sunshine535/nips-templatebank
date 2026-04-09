#!/usr/bin/env python3
"""Evaluate Test-Time Tool Building (SEVAL Phase 2).

Measures Claim C3: per-problem tool building recovers ≥20% of failures.
Compares against matched-budget baselines (search without building).

Usage:
    python scripts/eval_test_time_tools.py \\
        --library results/seval/library_final.json \\
        --model_dir results/seval/model_final \\
        --eval_data results/mcd_split_v2.json \\
        --output_dir results/seval/test_time_eval
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.template_dsl import (
    CompositionExecutor,
    CompositionPlan,
    SubroutineLibrary,
)
from src.test_time_tools import TestTimeToolBuilder, BuildStats
from src.mcts_search import mcts_solve, mcts_solve_with_repair

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate test-time tool building")
    p.add_argument("--library", type=str, required=True)
    p.add_argument("--model_dir", type=str, required=True)
    p.add_argument("--eval_data", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="results/seval/test_time_eval")
    p.add_argument("--max_samples", type=int, default=200)
    p.add_argument("--max_new_tools", type=int, default=3)
    p.add_argument("--max_verify_attempts", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_model(model_dir: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(model_dir, "../tokenizer_final"),
        trust_remote_code=True,
    )
    base_model_name = tokenizer.name_or_path
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, model_dir)
    model.eval()
    return model, tokenizer


def generate_plans(model, tokenizer, problem: str, library: SubroutineLibrary, n: int = 5):
    """Generate n composition plans for a problem."""
    lib_sigs = "\n".join(library.signatures())
    prompt = (
        f"Available subroutines:\n{lib_sigs}\n\n"
        f"Problem: {problem}\n\n"
        f"Output a composition plan as JSON:\nPlan:"
    )
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    plans = []
    with torch.no_grad():
        for _ in range(n):
            output = model.generate(
                **inputs, max_new_tokens=384,
                do_sample=True, temperature=0.6,
            )
            completion = tokenizer.decode(
                output[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            plan = _parse_plan(completion)
            if plan is not None:
                plans.append(plan)
    return plans


def _parse_plan(text: str):
    text = text.strip()
    for start_tok in ['{"plan"', '[{"sub_id"']:
        idx = text.find(start_tok)
        if idx >= 0:
            candidate = text[idx:]
            try:
                if candidate.startswith("["):
                    obj = json.loads(candidate[:candidate.rindex("]") + 1])
                    return CompositionPlan(calls=obj)
                else:
                    obj = json.loads(candidate[:candidate.rindex("}") + 1])
                    return CompositionPlan.from_dict(obj)
            except (json.JSONDecodeError, ValueError):
                pass
    return None


def evaluate(args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # Load resources
    library = SubroutineLibrary.load(args.library)
    logger.info(f"Library: {library.size} subroutines")

    model, tokenizer = load_model(args.model_dir)

    with open(args.eval_data) as f:
        eval_data = json.load(f)

    if isinstance(eval_data, dict) and "test" in eval_data:
        eval_data = eval_data["test"]
    eval_data = eval_data[:args.max_samples]
    logger.info(f"Evaluating on {len(eval_data)} problems")

    os.makedirs(args.output_dir, exist_ok=True)

    # Build test-time tool builder
    builder = TestTimeToolBuilder(
        library=library,
        model=model,
        tokenizer=tokenizer,
        max_new_tools=args.max_new_tools,
        max_verify_attempts=args.max_verify_attempts,
    )

    comp_exec = CompositionExecutor(library)
    results = {
        "baseline_correct": 0,
        "baseline_failed": 0,
        "building_recovered": 0,
        "building_failed": 0,
        "total": 0,
        "per_problem": [],
    }

    for i, problem in enumerate(eval_data):
        question = problem.get("question", "")
        bindings = problem.get("bindings", {})
        gold = problem.get("gold_answer")
        results["total"] += 1

        # Step 1: Generate plans with base library
        plans = generate_plans(model, tokenizer, question, library, n=5)

        # Step 2: Try each plan
        baseline_solved = False
        failed_plans = []
        for plan in plans:
            success, result, stats = comp_exec.execute(plan, bindings)
            if success and result is not None and gold is not None:
                try:
                    if abs(float(result) - float(gold)) < 1e-3 * max(abs(float(gold)), 1.0):
                        baseline_solved = True
                        break
                except (ValueError, TypeError):
                    pass
            failed_plans.append(plan)

        if baseline_solved:
            results["baseline_correct"] += 1
            results["per_problem"].append({
                "idx": i, "baseline": True, "building": None,
            })
            continue

        results["baseline_failed"] += 1

        # Step 3: Try test-time tool building
        recovered_plan, build_stats = builder.solve_with_building(
            problem=question,
            bindings=bindings,
            failed_plans=failed_plans,
            gold_answer=gold,
        )

        if build_stats.recovered:
            results["building_recovered"] += 1
        else:
            results["building_failed"] += 1

        results["per_problem"].append({
            "idx": i,
            "baseline": False,
            "building": build_stats.recovered,
            "tools_generated": build_stats.candidates_generated,
            "tools_accepted": build_stats.candidates_accepted,
            "retries": build_stats.compositions_retried,
        })

        if (i + 1) % 20 == 0:
            recovery_rate = (
                results["building_recovered"] / max(results["baseline_failed"], 1) * 100
            )
            logger.info(
                f"[{i+1}/{len(eval_data)}] "
                f"Baseline: {results['baseline_correct']}/{results['total']} "
                f"({results['baseline_correct']/results['total']*100:.1f}%), "
                f"Building recovered: {results['building_recovered']}/{results['baseline_failed']} "
                f"({recovery_rate:.1f}%)"
            )

    # Final stats
    recovery_rate = (
        results["building_recovered"] / max(results["baseline_failed"], 1) * 100
    )
    results["recovery_rate"] = recovery_rate
    results["baseline_accuracy"] = results["baseline_correct"] / max(results["total"], 1)
    results["total_accuracy"] = (
        (results["baseline_correct"] + results["building_recovered"]) / max(results["total"], 1)
    )

    logger.info("=" * 60)
    logger.info("Test-Time Tool Building Results")
    logger.info(f"  Total problems: {results['total']}")
    logger.info(f"  Baseline correct: {results['baseline_correct']} ({results['baseline_accuracy']*100:.1f}%)")
    logger.info(f"  Baseline failed: {results['baseline_failed']}")
    logger.info(f"  Building recovered: {results['building_recovered']} ({recovery_rate:.1f}%)")
    logger.info(f"  Total accuracy: {results['total_accuracy']*100:.1f}%")
    logger.info(f"  Claim C3 target: ≥20% recovery → {'PASS' if recovery_rate >= 20 else 'FAIL'}")
    logger.info("=" * 60)

    # Save results
    with open(os.path.join(args.output_dir, "test_time_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {args.output_dir}/test_time_results.json")


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
