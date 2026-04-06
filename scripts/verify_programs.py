#!/usr/bin/env python3
"""Post-process extracted programs: verify answer correctness and filter.

This script takes all_programs.json from Stage 1 (which may contain programs
that execute but produce wrong answers) and filters to keep only programs
whose execution result matches the gold answer.

Usage:
    python scripts/verify_programs.py \
        --input results/templates_full/all_programs.json \
        --output results/templates_full/verified_programs.json
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.template_dsl import DType, Executor, Program

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def extract_gold_answer(item: dict) -> str:
    """Extract gold answer from problem data."""
    answer = item.get("answer", "")
    if answer:
        return str(answer).strip()
    solution = item.get("solution", "")
    # Try boxed
    m = re.search(r'\\boxed\{([^}]+)\}', str(solution))
    if m:
        return m.group(1).strip()
    # Try ####
    if "####" in str(solution):
        return str(solution).split("####")[-1].strip()
    # Last number
    nums = re.findall(r'[\-\d,]+\.?\d*', str(solution))
    return nums[-1].replace(",", "") if nums else ""


def answer_matches(exec_result, gold_answer: str, tol: float = 1e-3) -> bool:
    """Check if execution result matches gold answer."""
    if not gold_answer or gold_answer.strip() == "":
        return False
    try:
        exec_val = float(str(exec_result).replace(",", ""))
        gold_val = float(str(gold_answer).replace(",", ""))
        if gold_val == 0:
            return abs(exec_val) < tol
        return abs(exec_val - gold_val) / max(abs(gold_val), 1e-12) < tol
    except (ValueError, TypeError):
        return str(exec_result).strip().lower() == str(gold_answer).strip().lower()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to all_programs.json")
    parser.add_argument("--output", required=True, help="Path to write verified_programs.json")
    parser.add_argument("--re_execute", action="store_true", help="Re-execute programs (don't trust stored exec_result)")
    args = parser.parse_args()

    with open(args.input) as f:
        programs = json.load(f)
    logger.info("Loaded %d programs from %s", len(programs), args.input)

    executor = Executor()
    verified = []
    stats = {"total": len(programs), "has_answer": 0, "exec_ok": 0, "answer_correct": 0}
    by_source = {}

    for i, item in enumerate(programs):
        gold = extract_gold_answer(item)
        if not gold:
            continue
        stats["has_answer"] += 1

        if args.re_execute:
            prog = Program.from_dict(item["program"])
            bindings = item.get("bindings", {})
            # Ensure bindings are properly typed
            typed_bindings = {}
            for slot in prog.slots:
                if slot.name in bindings:
                    try:
                        typed_bindings[slot.name] = DType.coerce(bindings[slot.name], slot.dtype)
                    except TypeError:
                        typed_bindings[slot.name] = bindings[slot.name]
                else:
                    typed_bindings[slot.name] = 0
            success, result, env = executor.execute(prog, typed_bindings)
            if not success or result is None:
                continue
            stats["exec_ok"] += 1
            exec_result = result
        else:
            exec_result = item.get("exec_result")
            if exec_result is None:
                continue
            stats["exec_ok"] += 1

        if answer_matches(exec_result, gold):
            stats["answer_correct"] += 1
            item["verified"] = True
            item["gold_answer"] = gold
            verified.append(item)

            src = item.get("source", "unknown")
            by_source[src] = by_source.get(src, 0) + 1

        if (i + 1) % 500 == 0:
            logger.info("  Progress: %d/%d, verified=%d", i + 1, len(programs), len(verified))

    with open(args.output, "w") as f:
        json.dump(verified, f, indent=2, ensure_ascii=False)

    logger.info("=" * 60)
    logger.info("  Verification Results")
    logger.info("=" * 60)
    logger.info("  Total programs: %d", stats["total"])
    logger.info("  With gold answer: %d (%.1f%%)", stats["has_answer"], 100 * stats["has_answer"] / max(stats["total"], 1))
    logger.info("  Executable: %d (%.1f%%)", stats["exec_ok"], 100 * stats["exec_ok"] / max(stats["has_answer"], 1))
    logger.info("  Answer-correct: %d (%.1f%%)", stats["answer_correct"], 100 * stats["answer_correct"] / max(stats["exec_ok"], 1))
    logger.info("  By source: %s", json.dumps(by_source))
    logger.info("  Verified programs saved to %s", args.output)

    # Save stats
    stats_path = args.output.replace(".json", "_stats.json")
    with open(stats_path, "w") as f:
        json.dump({**stats, "by_source": by_source}, f, indent=2)


if __name__ == "__main__":
    main()
