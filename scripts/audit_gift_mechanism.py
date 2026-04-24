#!/usr/bin/env python3
"""Audit GIFT plans for active-binding mechanism (Task 4).

Active binding = perturbing a used input produces a different output.
If perturbation does NOT change output, the binding is dead (not active).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataflow_plan import BindingRef, DataflowExecutor, DataflowPlan, PlanCall
from src.template_dsl import SubroutineLibrary


def perturb_value(v):
    try:
        fv = float(v)
        if fv == 0:
            return 1.0
        return fv * 2 + 7.0
    except (ValueError, TypeError):
        return v


def test_active_binding(plan_dict, library):
    """Test if perturbing each used quantity binding changes the output."""
    executor = DataflowExecutor(library)
    plan = DataflowPlan.from_dict(plan_dict)

    ok, baseline_result, _ = executor.execute(plan)
    if not ok or baseline_result is None:
        return {"baseline_ok": False, "active": 0, "tested": 0, "dead": 0}

    tested = 0
    active = 0
    dead = 0

    for call_idx, call in enumerate(plan.calls):
        for slot_name, ref in list(call.bindings.items()):
            if ref.source != "quantity":
                continue
            tested += 1
            original = ref.value
            perturbed = perturb_value(original)
            new_calls = []
            for ci, c in enumerate(plan.calls):
                new_bindings = {}
                for sn, r in c.bindings.items():
                    if ci == call_idx and sn == slot_name:
                        new_bindings[sn] = BindingRef(
                            source=r.source, value=perturbed,
                            qid=r.qid, span=r.span, dtype=r.dtype,
                        )
                    else:
                        new_bindings[sn] = r
                new_calls.append(PlanCall(c.call_id, c.sub_id, new_bindings))
            new_plan = DataflowPlan(calls=new_calls, final=plan.final)
            ok2, new_result, _ = executor.execute(new_plan)
            if ok2 and new_result is not None:
                try:
                    if abs(float(new_result) - float(baseline_result)) > 1e-6:
                        active += 1
                    else:
                        dead += 1
                except (ValueError, TypeError):
                    if str(new_result) != str(baseline_result):
                        active += 1
                    else:
                        dead += 1
            else:
                active += 1
    return {
        "baseline_ok": True,
        "baseline": baseline_result,
        "tested": tested,
        "active": active,
        "dead": dead,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gift_data", default="results/gift/compose_train_gift.json")
    parser.add_argument("--library", default="results/gift/library_gift.json")
    parser.add_argument("--output", default="results/gift/active_binding_audit.json")
    parser.add_argument("--max_examples", type=int, default=0)
    args = parser.parse_args()

    data = json.load(open(args.gift_data))
    library = SubroutineLibrary.load(args.library)
    if args.max_examples > 0:
        data = data[:args.max_examples]

    total_tested = 0
    total_active = 0
    total_dead = 0
    plans_with_any_dead = 0
    plans_all_active = 0
    n_plans = 0

    for item in data:
        plan_dict = item.get("plan") or item.get("plan_data") or item
        r = test_active_binding(plan_dict, library)
        if not r["baseline_ok"]:
            continue
        n_plans += 1
        total_tested += r["tested"]
        total_active += r["active"]
        total_dead += r["dead"]
        if r["dead"] > 0:
            plans_with_any_dead += 1
        if r["dead"] == 0 and r["tested"] > 0:
            plans_all_active += 1

    audit = {
        "data_file": args.gift_data,
        "n_plans_executed": n_plans,
        "total_bindings_tested": total_tested,
        "total_active": total_active,
        "total_dead": total_dead,
        "active_binding_rate": total_active / max(total_tested, 1),
        "plans_all_active": plans_all_active,
        "plans_with_any_dead": plans_with_any_dead,
        "plans_all_active_rate": plans_all_active / max(n_plans, 1),
    }

    print("=" * 60)
    print(f"  Active-Binding Audit: {args.gift_data}")
    print("=" * 60)
    print(f"  Plans executed OK: {n_plans}")
    print(f"  Bindings tested: {total_tested}")
    print(f"  Active: {total_active} ({100*audit['active_binding_rate']:.1f}%)")
    print(f"  Dead: {total_dead}")
    print(f"  Plans with all-active bindings: {plans_all_active} ({100*audit['plans_all_active_rate']:.1f}%)")
    print("=" * 60)

    with open(args.output, "w") as f:
        json.dump(audit, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
