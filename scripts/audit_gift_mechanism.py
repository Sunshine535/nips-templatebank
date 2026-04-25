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


def _diff(a, b):
    try:
        return abs(float(a) - float(b)) > 1e-6
    except (ValueError, TypeError):
        return str(a) != str(b)


def test_active_binding(plan_dict, library):
    """Audit per-binding mechanism activity for one plan.

    Two audits performed:
      1. Quantity perturbation: replace ref.value/qid value, re-execute.
      2. Call-output edge perturbation: redirect each call_output ref to
         a different prior call's output (or constant 0 if no other),
         re-execute. Confirms the edge is causal.
    """
    executor = DataflowExecutor(library)
    plan = DataflowPlan.from_dict(plan_dict)

    ok, baseline_result, _ = executor.execute(plan)
    if not ok or baseline_result is None:
        return {
            "baseline_ok": False,
            "quantity_tested": 0, "quantity_active": 0, "quantity_dead": 0,
            "edge_tested": 0, "edge_active": 0, "edge_dead": 0,
        }

    q_tested = q_active = q_dead = 0
    e_tested = e_active = e_dead = 0

    for call_idx, call in enumerate(plan.calls):
        for slot_name, ref in list(call.bindings.items()):
            if ref.source != "quantity":
                continue
            q_tested += 1
            perturbed = perturb_value(ref.value)
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
                if _diff(new_result, baseline_result):
                    q_active += 1
                else:
                    q_dead += 1
            else:
                q_active += 1

    for call_idx, call in enumerate(plan.calls):
        prior_call_ids = [c.call_id for c in plan.calls[:call_idx]]
        for slot_name, ref in list(call.bindings.items()):
            if ref.source != "call_output":
                continue
            e_tested += 1
            alternates = [c for c in prior_call_ids if c != ref.call_id]
            if alternates:
                redirect = BindingRef(
                    source="call_output", call_id=alternates[0],
                    dtype=ref.dtype,
                )
            else:
                redirect = BindingRef(
                    source="constant", value=0, dtype=ref.dtype,
                )
            new_calls = []
            for ci, c in enumerate(plan.calls):
                new_bindings = {}
                for sn, r in c.bindings.items():
                    if ci == call_idx and sn == slot_name:
                        new_bindings[sn] = redirect
                    else:
                        new_bindings[sn] = r
                new_calls.append(PlanCall(c.call_id, c.sub_id, new_bindings))
            new_plan = DataflowPlan(calls=new_calls, final=plan.final)
            ok2, new_result, _ = executor.execute(new_plan)
            if ok2 and new_result is not None:
                if _diff(new_result, baseline_result):
                    e_active += 1
                else:
                    e_dead += 1
            else:
                e_active += 1

    return {
        "baseline_ok": True,
        "baseline": baseline_result,
        "quantity_tested": q_tested,
        "quantity_active": q_active,
        "quantity_dead": q_dead,
        "edge_tested": e_tested,
        "edge_active": e_active,
        "edge_dead": e_dead,
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

    q_tested_sum = q_active_sum = q_dead_sum = 0
    e_tested_sum = e_active_sum = e_dead_sum = 0
    n_plans = 0

    for item in data:
        plan_dict = item.get("plan") or item.get("plan_data") or item
        r = test_active_binding(plan_dict, library)
        if not r["baseline_ok"]:
            continue
        n_plans += 1
        q_tested_sum += r["quantity_tested"]
        q_active_sum += r["quantity_active"]
        q_dead_sum += r["quantity_dead"]
        e_tested_sum += r["edge_tested"]
        e_active_sum += r["edge_active"]
        e_dead_sum += r["edge_dead"]

    audit = {
        "data_file": args.gift_data,
        "n_plans_executed": n_plans,
        "quantity_tested": q_tested_sum,
        "quantity_active": q_active_sum,
        "quantity_dead": q_dead_sum,
        "quantity_active_rate": q_active_sum / max(q_tested_sum, 1),
        "call_output_edges_tested": e_tested_sum,
        "call_output_edges_active": e_active_sum,
        "call_output_edges_dead": e_dead_sum,
        "call_output_edge_active_rate": e_active_sum / max(e_tested_sum, 1),
    }

    print("=" * 60)
    print(f"  Active-Binding Audit (v2): {args.gift_data}")
    print("=" * 60)
    print(f"  Plans executed OK: {n_plans}")
    print(f"  Quantity bindings: tested={q_tested_sum}, active={q_active_sum} "
          f"({100*audit['quantity_active_rate']:.1f}%), dead={q_dead_sum}")
    print(f"  Call-output edges:  tested={e_tested_sum}, active={e_active_sum} "
          f"({100*audit['call_output_edge_active_rate']:.1f}%), dead={e_dead_sum}")
    print("=" * 60)

    with open(args.output, "w") as f:
        json.dump(audit, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
