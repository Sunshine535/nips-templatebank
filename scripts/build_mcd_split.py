#!/usr/bin/env python3
"""Build CFQ-style Maximum Compound Divergence split from composition plans.

Reads plans_with_programs.json, constructs MCD split, saves to split file.
Defaults come from the config's mcd_split block; CLI args override them.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.mcd_split import build_mcd_split, save_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Build MCD split from composition plans")
    parser.add_argument("--config", type=str, default=str(Path(__file__).resolve().parent.parent / "configs" / "template_config.yaml"))
    parser.add_argument("--plans", type=str, default="results/templates/plans_with_programs.json")
    parser.add_argument("--output", type=str, default="results/mcd_split.json")
    parser.add_argument("--train_ratio", type=float, default=None)
    parser.add_argument("--dev_ratio", type=float, default=None)
    parser.add_argument("--test_ratio", type=float, default=None)
    parser.add_argument("--max_atom_tvd", type=float, default=None)
    parser.add_argument("--min_unseen_compounds", type=float, default=None)
    parser.add_argument("--num_trials", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    mcd_cfg = {}
    if os.path.exists(args.config):
        with open(args.config) as f:
            config = yaml.safe_load(f)
        mcd_cfg = config.get("mcd_split", {})
        logger.info("Loaded mcd_split config from %s", args.config)
    else:
        logger.warning("Config not found at %s, using built-in defaults", args.config)

    train_ratio = args.train_ratio if args.train_ratio is not None else mcd_cfg.get("train", 0.6)
    dev_ratio = args.dev_ratio if args.dev_ratio is not None else mcd_cfg.get("dev", 0.2)
    test_ratio = args.test_ratio if args.test_ratio is not None else mcd_cfg.get("test", 0.2)
    max_atom_tvd = args.max_atom_tvd if args.max_atom_tvd is not None else mcd_cfg.get("max_atom_tvd", 0.02)
    min_unseen = args.min_unseen_compounds if args.min_unseen_compounds is not None else mcd_cfg.get("min_unseen_test_compounds", 0.40)
    num_trials = args.num_trials if args.num_trials is not None else mcd_cfg.get("num_trials", 500)

    logger.info("MCD params: train=%.2f dev=%.2f test=%.2f atom_tvd<=%.3f unseen>=%.3f trials=%d",
                train_ratio, dev_ratio, test_ratio, max_atom_tvd, min_unseen, num_trials)

    logger.info("Loading plans from %s", args.plans)
    with open(args.plans) as f:
        plans = json.load(f)
    logger.info("Loaded %d plans", len(plans))

    examples = []
    for item in plans:
        examples.append({"plan_data": item.get("plan_data", {})})

    split = build_mcd_split(
        examples,
        train_ratio=train_ratio,
        dev_ratio=dev_ratio,
        test_ratio=test_ratio,
        max_atom_tvd=max_atom_tvd,
        min_unseen_compounds=min_unseen,
        num_trials=num_trials,
        seed=args.seed,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    save_split(split, args.output)
    logger.info("MCD split saved to %s", args.output)
    logger.info("Stats: %s", json.dumps(split.get("stats", {}), indent=2))


if __name__ == "__main__":
    main()
