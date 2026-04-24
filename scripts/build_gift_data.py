#!/usr/bin/env python3
"""Build faithful GIFT training data from verified programs.

For each verified program, try to construct an explicit DataflowPlan
where every slot is bound to a problem quantity or a previous call output.
Only keep plans that execute to the original verified answer.

Usage:
    python3 scripts/build_gift_data.py \
        --programs results/templates_verified/all_programs.json \
        --library results/templates_verified/subroutine_library.json \
        --output_dir results/gift \
        --max_examples 0
"""

import argparse
import json
import logging
import os
import re
import sys
from itertools import permutations
from math import factorial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataflow_plan import BindingRef, DataflowExecutor, DataflowPlan, PlanCall
from src.template_dsl import (
    DType,
    Executor,
    Program,
    Slot,
    SubroutineLibrary,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def extract_quantities(problem_text):
    """Extract grounded quantities from problem text."""
    numbers = re.findall(r'[\d,]+\.?\d*', problem_text)
    quantities = []
    for i, n in enumerate(numbers):
        cleaned = n.replace(",", "").strip()
        if not cleaned or cleaned == ".":
            continue
        try:
            val = float(cleaned)
            if "." not in cleaned:
                val = int(val)
            quantities.append({
                "id": f"q{i}",
                "value": val,
                "span": n,
            })
        except ValueError:
            continue
    return quantities


def answer_matches(result, answer):
    try:
        r = float(str(result).replace(",", "").strip())
        a = float(str(answer).replace(",", "").strip())
        if a == 0:
            return abs(r) < 1e-3
        return abs(r - a) / max(abs(a), 1e-8) < 0.01
    except (ValueError, TypeError):
        return str(result).strip().lower() == str(answer).strip().lower()


def try_build_single_call_plan(program, subroutine, quantities, answer, executor):
    """Try to build a faithful single-call DataflowPlan."""
    sub_slots = subroutine.program.slots
    n_slots = len(sub_slots)
    values = [q["value"] for q in quantities]

    if len(values) < n_slots:
        return None

    n_perms = factorial(len(values)) // factorial(len(values) - n_slots)
    if n_perms > 2000:
        return None

    for perm in permutations(values, n_slots):
        bindings_dict = {slot.name: val for slot, val in zip(sub_slots, perm)}
        success, result, _ = executor.execute(subroutine.program, bindings_dict)
        if success and result is not None and answer_matches(result, answer):
            binding_refs = {}
            for slot, val in zip(sub_slots, perm):
                q_match = next((q for q in quantities if q["value"] == val), None)
                if q_match:
                    binding_refs[slot.name] = BindingRef(
                        source="quantity", value=val
                    )
                else:
                    binding_refs[slot.name] = BindingRef(
                        source="constant", value=val
                    )

            plan = DataflowPlan(
                calls=[PlanCall(
                    call_id="c0",
                    sub_id=subroutine.sub_id,
                    bindings=binding_refs,
                )],
                final=BindingRef(source="call_output", call_id="c0"),
            )
            return plan, result

    return None


def try_build_two_call_plan(program, library, quantities, answer, executor):
    """Try to build a faithful two-call DataflowPlan with real dataflow."""
    values = [q["value"] for q in quantities]
    subs = list(library.subroutines.values())

    for s1 in subs:
        for s2 in subs:
            s1_slots = s1.program.slots
            s2_slots = s2.program.slots
            n1 = len(s1_slots)
            n2 = len(s2_slots)

            if n1 + n2 - 1 > len(values) + 1:
                continue

            if n1 > len(values):
                continue

            n_perms_1 = factorial(len(values)) // factorial(len(values) - n1) if len(values) >= n1 else 0
            if n_perms_1 > 200:
                continue

            for perm1 in permutations(values, n1):
                b1 = {slot.name: val for slot, val in zip(s1_slots, perm1)}
                ok1, r1, _ = executor.execute(s1.program, b1)
                if not (ok1 and r1 is not None):
                    continue

                remaining = [v for v in values]
                for v in perm1:
                    if v in remaining:
                        remaining.remove(v)

                for s2_slot_idx in range(n2):
                    other_slots = [s for i, s in enumerate(s2_slots) if i != s2_slot_idx]
                    n_other = len(other_slots)

                    if n_other > len(remaining):
                        continue
                    n_perms_2 = factorial(len(remaining)) // factorial(len(remaining) - n_other) if len(remaining) >= n_other else 0
                    if n_perms_2 > 100:
                        continue

                    for perm2 in permutations(remaining, n_other):
                        b2 = {}
                        p2_idx = 0
                        for i, slot in enumerate(s2_slots):
                            if i == s2_slot_idx:
                                b2[slot.name] = r1
                            else:
                                b2[slot.name] = perm2[p2_idx]
                                p2_idx += 1

                        ok2, r2, _ = executor.execute(s2.program, b2)
                        if ok2 and r2 is not None and answer_matches(r2, answer):
                            refs1 = {}
                            for slot, val in zip(s1_slots, perm1):
                                refs1[slot.name] = BindingRef(source="quantity", value=val)

                            refs2 = {}
                            p2_idx = 0
                            for i, slot in enumerate(s2_slots):
                                if i == s2_slot_idx:
                                    refs2[slot.name] = BindingRef(source="call_output", call_id="c0")
                                else:
                                    refs2[slot.name] = BindingRef(source="quantity", value=perm2[p2_idx])
                                    p2_idx += 1

                            plan = DataflowPlan(
                                calls=[
                                    PlanCall(call_id="c0", sub_id=s1.sub_id, bindings=refs1),
                                    PlanCall(call_id="c1", sub_id=s2.sub_id, bindings=refs2),
                                ],
                                final=BindingRef(source="call_output", call_id="c1"),
                            )
                            return plan, r2

    return None


def build_flat_faithful(item):
    """Build a faithful flat training example from the original verified program."""
    return {
        "instruction": f"Problem: {item['problem']}\n\nGenerate an executable JSON program:",
        "output": json.dumps(item["program"]),
        "problem": item["problem"],
        "answer": item["answer"],
        "source": item.get("source", ""),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--programs", default="results/templates_verified/all_programs.json")
    parser.add_argument("--library", default="results/templates_verified/subroutine_library.json")
    parser.add_argument("--output_dir", default="results/gift")
    parser.add_argument("--max_examples", type=int, default=0, help="0 = all")
    parser.add_argument("--try_two_call", action="store_true", help="Try 2-call plans (slow)")
    args = parser.parse_args()

    programs = json.load(open(args.programs))
    library = SubroutineLibrary.load(args.library)
    executor = Executor()
    df_executor = DataflowExecutor(library)

    if args.max_examples > 0:
        programs = programs[:args.max_examples]

    logger.info("Programs: %d, Library: %d subs", len(programs), library.size)

    os.makedirs(args.output_dir, exist_ok=True)

    gift_plans = []
    flat_faithful = []
    audit = {
        "total": len(programs),
        "single_call_faithful": 0,
        "two_call_faithful": 0,
        "no_match": 0,
        "true_dataflow": 0,
        "empty_binding": 0,
        "active_binding_tested": 0,
        "active_binding_passed": 0,
    }

    for i, item in enumerate(programs):
        problem = item["problem"]
        answer = item["answer"]
        quantities = extract_quantities(problem)

        best_plan = None
        best_result = None

        for sub in library.subroutines.values():
            prog = Program.from_dict(item["program"])
            sub_steps = sub.program.steps
            if len(sub_steps) != len(prog.steps):
                continue
            ops_match = all(
                s1.op.value == s2.op.value
                for s1, s2 in zip(sub_steps, prog.steps)
            )
            if not ops_match:
                continue

            result = try_build_single_call_plan(prog, sub, quantities, answer, executor)
            if result is not None:
                best_plan, best_result = result
                audit["single_call_faithful"] += 1
                break

        if best_plan is None and args.try_two_call and len(quantities) >= 3:
            result = try_build_two_call_plan(
                Program.from_dict(item["program"]), library, quantities, answer, executor
            )
            if result is not None:
                best_plan, best_result = result
                audit["two_call_faithful"] += 1
                audit["true_dataflow"] += 1

        if best_plan is None:
            audit["no_match"] += 1
        else:
            ok, verified_result, stats = df_executor.execute(best_plan)
            if not (ok and answer_matches(verified_result, answer)):
                audit["no_match"] += 1
                continue

            if best_plan.has_true_dataflow():
                audit["true_dataflow"] += 1

            plan_json = json.dumps(best_plan.to_dict())
            gift_plans.append({
                "problem": problem,
                "answer": answer,
                "source": item.get("source", ""),
                "plan": best_plan.to_dict(),
                "plan_json": plan_json,
                "verified_result": best_result,
                "has_dataflow": best_plan.has_true_dataflow(),
            })

        flat_faithful.append(build_flat_faithful(item))

        if (i + 1) % 100 == 0:
            logger.info(
                "  %d/%d: gift=%d, flat=%d, no_match=%d",
                i + 1, len(programs), len(gift_plans), len(flat_faithful), audit["no_match"],
            )

    coverage = len(gift_plans) / max(len(programs), 1)
    audit["gift_plans"] = len(gift_plans)
    audit["flat_faithful"] = len(flat_faithful)
    audit["coverage"] = coverage

    logger.info("=" * 60)
    logger.info("  GIFT Data Build Results")
    logger.info("=" * 60)
    logger.info("  Total programs: %d", len(programs))
    logger.info("  GIFT plans (faithful): %d (%.1f%%)", len(gift_plans), 100 * coverage)
    logger.info("    Single-call: %d", audit["single_call_faithful"])
    logger.info("    Two-call (true dataflow): %d", audit["two_call_faithful"])
    logger.info("  No match: %d", audit["no_match"])
    logger.info("  Flat faithful: %d", len(flat_faithful))
    logger.info("  True dataflow plans: %d", audit["true_dataflow"])
    logger.info("=" * 60)

    if coverage < 0.30:
        logger.warning("COVERAGE BELOW 30%% (%.1f%%) — library mining may be too coarse", 100 * coverage)

    with open(os.path.join(args.output_dir, "compose_train_gift.json"), "w") as f:
        json.dump(gift_plans, f, indent=2, ensure_ascii=False)

    with open(os.path.join(args.output_dir, "flat_train_faithful.json"), "w") as f:
        json.dump(flat_faithful, f, indent=2, ensure_ascii=False)

    library.save(os.path.join(args.output_dir, "library_gift.json"))

    with open(os.path.join(args.output_dir, "plan_audit.json"), "w") as f:
        json.dump(audit, f, indent=2)

    logger.info("Saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
