#!/usr/bin/env python3
"""Semantic audit of the subroutine library for manual paper-time inspection.

Outputs a structured report (JSON + human-readable) per subroutine:
  - Signature, steps, support count
  - Random sample of composition plans that reference the subroutine
  - Coherence checklist prompts for manual review
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.template_dsl import SubroutineLibrary


def load_plans(plans_path: str) -> list[dict]:
    with open(plans_path) as f:
        return json.load(f)


def sample_plans_for_sub(plans: list[dict], sub_id: str, k: int = 100) -> list[dict]:
    """Return up to k plans that reference sub_id."""
    matches = []
    for p in plans:
        plan_data = p.get("plan", p.get("composition_plan", {}))
        calls = plan_data.get("plan", []) if isinstance(plan_data, dict) else []
        if any(c.get("sub_id") == sub_id for c in calls):
            matches.append(p)
    random.shuffle(matches)
    return matches[:k]


def audit(library: SubroutineLibrary, plans: list[dict] | None, sample_k: int) -> list[dict]:
    report = []
    for sid, sub in sorted(library.subroutines.items()):
        entry = {
            "sub_id": sid,
            "signature": sub.signature,
            "support": sub.support,
            "mdl_gain": sub.mdl_gain,
            "num_steps": len(sub.program.steps),
            "steps": [
                {"op": s.op.value, "target": s.target, "expr": s.expr}
                for s in sub.program.steps
            ],
            "slots": [
                {"name": sl.name, "dtype": sl.dtype.value, "description": sl.description}
                for sl in sub.program.slots
            ],
            "coherence_checklist": {
                "steps_semantically_related": None,  # fill manually: true/false
                "recognizable_math_pattern": None,    # fill manually: true/false
                "pattern_name": "",                    # e.g. "percentage_change", "unit_conversion"
                "notes": "",
            },
        }
        if plans is not None:
            samples = sample_plans_for_sub(plans, sid, sample_k)
            entry["sample_plan_count"] = len(samples)
            entry["sample_plans"] = [
                {k: v for k, v in p.items() if k in ("question", "problem", "answer", "solution", "composition_plan", "plan")}
                for p in samples[:5]  # include full detail for first 5 only
            ]
        report.append(entry)
    return report


def print_human_readable(report: list[dict]):
    print(f"{'='*70}")
    print(f"  SUBROUTINE LIBRARY AUDIT — {len(report)} subroutines")
    print(f"{'='*70}\n")
    for entry in report:
        print(f"--- {entry['sub_id']} (support={entry['support']}, steps={entry['num_steps']}, mdl_gain={entry['mdl_gain']:.3f}) ---")
        print(f"  Signature: {entry['signature']}")
        for s in entry["steps"]:
            print(f"    {s['op']:>10}  {s['target']} = {s['expr']}")
        if "sample_plan_count" in entry:
            print(f"  Plans referencing this subroutine: {entry['sample_plan_count']}")
        print(f"  [ ] Steps semantically related?")
        print(f"  [ ] Recognizable math pattern?  Name: ___________")
        print()


def main():
    parser = argparse.ArgumentParser(description="Audit subroutine library for semantic coherence")
    parser.add_argument("--library_path", type=str, default="results/templates/subroutine_library.json")
    parser.add_argument("--plans_path", type=str, default="results/templates/compose_train.json",
                        help="Path to composition plans (for usage sampling)")
    parser.add_argument("--output", type=str, default="results/eval/subroutine_audit.json")
    parser.add_argument("--sample_k", type=int, default=100, help="Max plans to sample per subroutine")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    library = SubroutineLibrary.load(args.library_path)
    plans = load_plans(args.plans_path) if Path(args.plans_path).exists() else None

    report = audit(library, plans, args.sample_k)
    print_human_readable(report)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nJSON report saved to {args.output}")


if __name__ == "__main__":
    main()
