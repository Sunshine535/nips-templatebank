#!/usr/bin/env python3
"""Build CFQ-style Maximum Compound Divergence split from composition plans.

Reads plans_with_programs.json, constructs MCD split, saves to split file.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.mcd_split import build_mcd_split, save_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Build MCD split from composition plans")
    parser.add_argument("--plans", type=str, default="results/templates/plans_with_programs.json")
    parser.add_argument("--output", type=str, default="results/mcd_split.json")
    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--dev_ratio", type=float, default=0.2)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--max_atom_tvd", type=float, default=0.02)
    parser.add_argument("--min_unseen_compounds", type=float, default=0.40)
    parser.add_argument("--num_trials", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logger.info("Loading plans from %s", args.plans)
    with open(args.plans) as f:
        plans = json.load(f)
    logger.info("Loaded %d plans", len(plans))

    examples = []
    for item in plans:
        examples.append({"plan_data": item.get("plan_data", {})})

    split = build_mcd_split(
        examples,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        test_ratio=args.test_ratio,
        max_atom_tvd=args.max_atom_tvd,
        min_unseen_compounds=args.min_unseen_compounds,
        num_trials=args.num_trials,
        seed=args.seed,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    save_split(split, args.output)
    logger.info("MCD split saved to %s", args.output)
    logger.info("Stats: %s", json.dumps(split.get("stats", {}), indent=2))


if __name__ == "__main__":
    main()
