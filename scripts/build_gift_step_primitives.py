#!/usr/bin/env python3
"""Build GIFT data from step-level primitives (GPT-5.5 Task 5).

Instead of matching verified programs to whole-program templates,
this script mines 1-step and 2-step primitives directly from each
verified program. Every program then becomes a true multi-call
DataflowPlan with explicit call_output edges.

Output:
  results/gift_step/compose_train_gift.json
  results/gift_step/library_gift.json
  results/gift_step/plan_audit.json
  results/gift_step/MANIFEST.json
"""

import argparse
import hashlib
import json
import logging
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataflow_plan import (
    BindingRef,
    DataflowExecutor,
    DataflowPlan,
    PlanCall,
)
from src.template_dsl import (
    DType,
    Executor,
    Op,
    Program,
    Slot,
    Step,
    Subroutine,
    SubroutineLibrary,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def extract_quantities(problem_text):
    """Extract grounded quantities with stable qids from problem text."""
    numbers = re.findall(r'[\d,]+\.?\d*', problem_text)
    quantities = {}
    for i, n in enumerate(numbers):
        cleaned = n.replace(",", "").strip()
        if not cleaned or cleaned == ".":
            continue
        try:
            val = float(cleaned)
            if "." not in cleaned:
                val = int(val)
            quantities[f"q{i}"] = {"value": val, "span": n}
        except ValueError:
            continue
    return quantities


def step_signature(step):
    """Canonical signature for a step: op + operators used in expr."""
    ops_in_expr = set()
    for op_str in ['+', '-', '*', '/', '**', '//', '%', 'min(', 'max(',
                   'abs(', 'round(', 'sum(', 'len(', 'sqrt(', 'ceil(', 'floor(']:
        if op_str in step.expr:
            ops_in_expr.add(op_str.rstrip('('))
    return f"{step.op.value}:{'|'.join(sorted(ops_in_expr))}"


def canonicalize_expr(expr, var_map):
    """Replace variables in expr with canonical names (a, b, c, ...)."""
    str_keys = {str(k): v for k, v in var_map.items()}
    sorted_orig = sorted(str_keys.keys(), key=lambda x: -len(x))
    out = str(expr)
    for orig in sorted_orig:
        out = re.sub(r'\b' + re.escape(orig) + r'\b', str_keys[orig], out)
    return out


def mine_step_primitives(programs, min_support=3):
    """Mine 1-step primitives from all programs.

    For each non-output step, extract a canonicalized Subroutine
    with N input slots (where N is the number of distinct input vars
    actually used by the expression).
    """
    patterns = defaultdict(list)
    for prog_item in programs:
        prog = Program.from_dict(prog_item["program"])
        for step_idx, step in enumerate(prog.steps):
            if step.op == Op.OUTPUT:
                continue
            input_vars = []
            for inp in step.inputs:
                s_inp = str(inp)
                if s_inp and s_inp not in input_vars:
                    input_vars.append(s_inp)
            if not input_vars:
                continue
            canonical_names = [chr(ord('a') + i) for i in range(len(input_vars))]
            var_map = {orig: canon for orig, canon in zip(input_vars, canonical_names)}
            canon_expr = canonicalize_expr(step.expr, var_map)
            sig = (step.op.value, canon_expr, len(input_vars))
            patterns[sig].append({
                "prog_id": prog.program_id,
                "step_idx": step_idx,
                "input_vars": input_vars,
                "target": step.target,
                "target_dtype": step.target_dtype.value,
            })

    library = SubroutineLibrary()
    sub_counter = 0
    sig_to_subid = {}
    for sig, instances in sorted(patterns.items(), key=lambda x: -len(x[1])):
        if len(instances) < min_support:
            continue
        op_val, canon_expr, n_slots = sig
        canonical_names = [chr(ord('a') + i) for i in range(n_slots)]
        slots = [Slot(n, DType.FLOAT, f"input {n}") for n in canonical_names]
        steps = [
            Step(
                Op(op_val), "result", canon_expr, canonical_names, DType.FLOAT,
            ),
            Step(Op.OUTPUT, "__output__", "result", ["result"], DType.FLOAT),
        ]
        sub_id = f"P{sub_counter:03d}"
        sub_program = Program(
            program_id=f"primitive_{sub_id}",
            slots=slots,
            steps=steps,
        )
        sub = Subroutine(
            sub_id=sub_id,
            program=sub_program,
            support=len(instances),
            mdl_gain=len(instances),
        )
        library.add(sub)
        sig_to_subid[sig] = sub_id
        sub_counter += 1

    return library, sig_to_subid


def build_program_plan(prog, sig_to_subid, quantities, library):
    """Build a DataflowPlan that executes each non-output step as a primitive call.

    Returns (plan, success) where success indicates whether a faithful
    plan could be built.
    """
    prog = Program.from_dict(prog) if isinstance(prog, dict) else prog

    var_to_source = {}
    prog_slots = {slot.name for slot in prog.slots}
    prog_slot_values = {}
    if hasattr(prog, "slots"):
        for slot in prog.slots:
            prog_slot_values[slot.name] = None

    calls = []
    call_counter = 0
    last_call_id = None

    q_values_list = [(qid, q["value"]) for qid, q in quantities.items()]
    q_value_to_qids = defaultdict(list)
    for qid, val in q_values_list:
        try:
            q_value_to_qids[float(val)].append(qid)
        except (ValueError, TypeError):
            q_value_to_qids[str(val)].append(qid)

    slot_to_qid = {}
    q_assignment_idx = 0
    for slot in prog.slots:
        if q_assignment_idx < len(q_values_list):
            slot_to_qid[slot.name] = q_values_list[q_assignment_idx][0]
            q_assignment_idx += 1

    for step_idx, step in enumerate(prog.steps):
        if step.op == Op.OUTPUT:
            continue

        input_vars = []
        for inp in step.inputs:
            s_inp = str(inp)
            if s_inp and s_inp not in input_vars:
                input_vars.append(s_inp)
        if not input_vars:
            return None

        canonical_names = [chr(ord('a') + i) for i in range(len(input_vars))]
        var_map = {orig: canon for orig, canon in zip(input_vars, canonical_names)}
        canon_expr = canonicalize_expr(step.expr, var_map)
        sig = (step.op.value, canon_expr, len(input_vars))

        sub_id = sig_to_subid.get(sig)
        if sub_id is None:
            return None

        call_id = f"c{call_counter}"
        bindings = {}
        for orig_var, canon_name in zip(input_vars, canonical_names):
            if orig_var in var_to_source:
                bindings[canon_name] = var_to_source[orig_var]
            elif orig_var in prog_slot_values or orig_var in slot_to_qid:
                qid = slot_to_qid.get(orig_var)
                if qid is not None:
                    bindings[canon_name] = BindingRef(
                        source="quantity",
                        qid=qid,
                        value=quantities[qid]["value"],
                        span=quantities[qid]["span"],
                    )
                else:
                    return None
            else:
                return None

        calls.append(PlanCall(call_id=call_id, sub_id=sub_id, bindings=bindings))
        var_to_source[step.target] = BindingRef(source="call_output", call_id=call_id)
        last_call_id = call_id
        call_counter += 1

    output_step = next((s for s in prog.steps if s.op == Op.OUTPUT), None)
    if output_step is None or not output_step.inputs:
        return None
    output_var = output_step.inputs[0]
    final_ref = var_to_source.get(output_var)
    if final_ref is None:
        return None

    plan = DataflowPlan(calls=calls, final=final_ref)
    return plan


def answer_matches(result, answer):
    try:
        r = float(str(result).replace(",", "").strip())
        a = float(str(answer).replace(",", "").strip())
        if a == 0:
            return abs(r) < 1e-3
        return abs(r - a) / max(abs(a), 1e-8) < 0.01
    except (ValueError, TypeError):
        return str(result).strip().lower() == str(answer).strip().lower()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--programs", default="results/templates_verified/all_programs.json")
    parser.add_argument("--output_dir", default="results/gift_step")
    parser.add_argument("--min_support", type=int, default=3)
    parser.add_argument("--max_examples", type=int, default=0)
    args = parser.parse_args()

    programs = json.load(open(args.programs))
    if args.max_examples > 0:
        programs = programs[:args.max_examples]
    logger.info("Programs loaded: %d", len(programs))

    logger.info("Mining step-level primitives (min_support=%d)...", args.min_support)
    library, sig_to_subid = mine_step_primitives(programs, min_support=args.min_support)
    logger.info("Library: %d step primitives", library.size)

    executor = DataflowExecutor(library)
    faithful_plans = []
    audit = {
        "total": len(programs),
        "plan_built": 0,
        "plan_executed": 0,
        "plan_correct": 0,
        "two_call_true_dataflow": 0,
        "multi_call_true_dataflow": 0,
        "call_counts": Counter(),
        "library_size": library.size,
    }

    for i, item in enumerate(programs):
        quantities = extract_quantities(item["problem"])
        plan = build_program_plan(item["program"], sig_to_subid, quantities, library)
        if plan is None:
            continue
        audit["plan_built"] += 1
        audit["call_counts"][len(plan.calls)] += 1
        if len(plan.calls) >= 2 and plan.has_true_dataflow():
            audit["multi_call_true_dataflow"] += 1
            if len(plan.calls) == 2:
                audit["two_call_true_dataflow"] += 1

        ok, result, stats = executor.execute_with_quantities(
            plan, {qid: q["value"] for qid, q in quantities.items()}
        )
        if ok and result is not None:
            audit["plan_executed"] += 1
            if answer_matches(result, item["answer"]):
                audit["plan_correct"] += 1
                faithful_plans.append({
                    "problem": item["problem"],
                    "answer": item["answer"],
                    "source": item.get("source", ""),
                    "quantities": quantities,
                    "plan": plan.to_dict(),
                    "verified_result": result,
                    "has_dataflow": plan.has_true_dataflow(),
                    "num_calls": len(plan.calls),
                })

        if (i + 1) % 100 == 0:
            logger.info(
                "  %d/%d: built=%d, executed=%d, correct=%d, multicall_flow=%d",
                i + 1, len(programs),
                audit["plan_built"], audit["plan_executed"],
                audit["plan_correct"], audit["multi_call_true_dataflow"],
            )

    audit["call_counts"] = dict(audit["call_counts"])
    coverage = audit["plan_correct"] / max(audit["total"], 1)
    audit["coverage"] = coverage
    audit["true_dataflow_rate"] = (
        audit["multi_call_true_dataflow"] / max(audit["plan_correct"], 1)
    )

    logger.info("=" * 60)
    logger.info("  GIFT Step-Primitive Build Results")
    logger.info("=" * 60)
    logger.info("  Library size: %d step primitives", library.size)
    logger.info("  Total programs: %d", audit["total"])
    logger.info("  Faithful GIFT plans: %d (%.1f%%)",
                audit["plan_correct"], 100 * coverage)
    logger.info("  Multi-call true dataflow: %d (%.1f%%)",
                audit["multi_call_true_dataflow"],
                100 * audit["true_dataflow_rate"])
    logger.info("  Two-call plans: %d", audit["two_call_true_dataflow"])
    logger.info("  Call count distribution: %s", audit["call_counts"])
    logger.info("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, "compose_train_gift.json"), "w") as f:
        json.dump(faithful_plans, f, indent=2, ensure_ascii=False)
    library.save(os.path.join(args.output_dir, "library_gift.json"))
    with open(os.path.join(args.output_dir, "plan_audit.json"), "w") as f:
        json.dump(audit, f, indent=2)

    manifest = {
        "gift_version": "step_primitive_v1",
        "builder": "scripts/build_gift_step_primitives.py",
        "source_programs": args.programs,
        "min_support": args.min_support,
        "coverage": coverage,
        "true_dataflow_rate": audit["true_dataflow_rate"],
    }
    for fname in ["compose_train_gift.json", "library_gift.json", "plan_audit.json"]:
        path = os.path.join(args.output_dir, fname)
        if os.path.exists(path):
            with open(path, "rb") as fp:
                manifest[fname + "_sha256"] = hashlib.sha256(fp.read()).hexdigest()
                manifest[fname + "_size"] = os.path.getsize(path)
    with open(os.path.join(args.output_dir, "MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    if coverage < 0.30:
        logger.warning("COVERAGE BELOW 30%% (%.1f%%)", 100 * coverage)
    if audit["multi_call_true_dataflow"] == 0:
        logger.warning("ZERO MULTI-CALL TRUE DATAFLOW — mechanism not active")
    logger.info("Saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
