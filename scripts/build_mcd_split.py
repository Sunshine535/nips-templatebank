#!/usr/bin/env python3
"""Build CFQ-style Maximum Compound Divergence split from programs.

Can accept either:
  - plans_with_programs.json (pre-built plans)
  - all_programs.json (raw programs — builds temporary library+plans internally)

Defaults come from the config's mcd_split block; CLI args override them.
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.mcd_split import build_mcd_split, save_split
from src.template_dsl import (
    CompositionPlan,
    Program,
    Subroutine,
    SubroutineLibrary,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _build_temp_library_and_plans(programs: list, config: dict) -> list:
    """Build temporary library and plans from raw programs (for MCD split only).

    This is a lightweight version used ONLY for split construction.
    The real library is rebuilt post-split from the train partition.
    """
    lib_cfg = config.get("library", {})
    target_size = lib_cfg.get("main_size", 16)
    min_support = max(lib_cfg.get("min_support_gsm8k", 5), 2)  # at least 2 for temp lib

    # Cluster programs by operation signature
    fp_groups = defaultdict(list)
    for item in programs:
        prog = Program.from_dict(item["program"])
        steps_sig = tuple(s.op.value for s in prog.steps)
        fp_groups[steps_sig].append((prog, item))

    sorted_groups = sorted(fp_groups.items(), key=lambda x: len(x[1]), reverse=True)

    lib = SubroutineLibrary()
    sub_counter = 0
    for sig, progs in sorted_groups:
        if lib.size >= target_size:
            break
        if len(progs) < min_support:
            continue
        representative = progs[0][0]
        sub_id = f"L{sub_counter:02d}"
        sub = Subroutine(sub_id=sub_id, program=representative, support=len(progs))
        if lib.add(sub):
            sub_counter += 1

    logger.info("Temporary library: %d subroutines (for split construction only)", lib.size)

    # Build temporary plans using greedy covering
    plans = []
    for item in programs:
        prog = Program.from_dict(item["program"])
        steps = prog.steps
        calls = []
        i = 0
        while i < len(steps):
            best_sub, best_len = None, 0
            for sub in lib.subroutines.values():
                sub_steps = sub.program.steps
                sub_len = len(sub_steps)
                if i + sub_len <= len(steps):
                    prog_sig = tuple(s.op.value for s in steps[i:i + sub_len])
                    sub_sig = tuple(s.op.value for s in sub_steps)
                    if prog_sig == sub_sig and sub_len > best_len:
                        best_sub = sub
                        best_len = sub_len
            if best_sub is not None:
                call_bindings = {}
                for slot in best_sub.program.slots:
                    if slot.name in item.get("bindings", {}):
                        call_bindings[slot.name] = item["bindings"][slot.name]
                calls.append({"sub_id": best_sub.sub_id, "bindings": call_bindings})
                i += best_len
            else:
                i += 1

        if not calls and lib.size > 0:
            first_sub = list(lib.subroutines.values())[0]
            calls = [{"sub_id": first_sub.sub_id, "bindings": item.get("bindings", {})}]

        plan = CompositionPlan(calls=calls)
        plans.append({**item, "plan_data": plan.to_dict()})

    return plans


def main():
    parser = argparse.ArgumentParser(description="Build MCD split from programs or plans")
    parser.add_argument("--config", type=str, default=str(Path(__file__).resolve().parent.parent / "configs" / "template_config.yaml"))
    parser.add_argument("--plans", type=str, default=None, help="Path to plans_with_programs.json (pre-built plans)")
    parser.add_argument("--programs", type=str, default=None, help="Path to all_programs.json (raw programs)")
    parser.add_argument("--output", type=str, default="results/mcd_split.json")
    parser.add_argument("--train_ratio", type=float, default=None)
    parser.add_argument("--dev_ratio", type=float, default=None)
    parser.add_argument("--test_ratio", type=float, default=None)
    parser.add_argument("--max_atom_tvd", type=float, default=None)
    parser.add_argument("--min_unseen_compounds", type=float, default=None)
    parser.add_argument("--num_trials", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--compound_mode",
        choices=["legacy", "true_dataflow"],
        default="legacy",
        help="legacy: adjacency-based compounds. true_dataflow: only explicit call_output edges.",
    )
    args = parser.parse_args()

    mcd_cfg = {}
    config = {}
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

    # Determine input: prefer --programs (new pipeline), fall back to --plans (legacy)
    if args.programs:
        logger.info("Loading raw programs from %s", args.programs)
        with open(args.programs) as f:
            raw_programs = json.load(f)
        logger.info("Loaded %d programs, building temporary library+plans for split...", len(raw_programs))
        plans = _build_temp_library_and_plans(raw_programs, config)
    elif args.plans:
        logger.info("Loading pre-built plans from %s", args.plans)
        with open(args.plans) as f:
            plans = json.load(f)
    else:
        # Default: try programs first, then plans
        default_programs = "results/templates/all_programs.json"
        default_plans = "results/templates/plans_with_programs.json"
        if os.path.exists(default_programs):
            logger.info("Loading raw programs from %s (default)", default_programs)
            with open(default_programs) as f:
                raw_programs = json.load(f)
            logger.info("Loaded %d programs, building temporary library+plans...", len(raw_programs))
            plans = _build_temp_library_and_plans(raw_programs, config)
        elif os.path.exists(default_plans):
            logger.info("Loading pre-built plans from %s (legacy default)", default_plans)
            with open(default_plans) as f:
                plans = json.load(f)
        else:
            logger.error("No input found. Provide --programs or --plans.")
            sys.exit(1)

    logger.info("Total examples for split: %d", len(plans))

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
        compound_mode=args.compound_mode,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    save_split(split, args.output)
    logger.info("MCD split saved to %s", args.output)
    logger.info("Stats: %s", json.dumps(split.get("stats", {}), indent=2))


if __name__ == "__main__":
    main()
