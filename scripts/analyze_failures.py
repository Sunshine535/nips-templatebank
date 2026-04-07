#!/usr/bin/env python3
"""Post-hoc failure analysis for subroutine composition predictions.

Categorizes each incorrect prediction into one of 8 failure bins,
computes failure-rate breakdowns by plan depth and subroutine count,
and writes a JSON report + human-readable summary.

Usage:
    python scripts/analyze_failures.py \
        --predictions results/eval/predictions.json \
        --library results/templates/subroutine_library.json \
        --n_sample 150 \
        --output results/failure_analysis.json
"""

import argparse
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.template_dsl import (
    CompositionExecutor,
    CompositionPlan,
    DType,
    Executor,
    Program,
    SubroutineLibrary,
)

# ---------------------------------------------------------------------------
# Failure bins
# ---------------------------------------------------------------------------

FAILURE_BINS = [
    "missing_abstraction",
    "wrong_abstraction",
    "wrong_bindings",
    "wrong_order",
    "spurious_subroutine",
    "dsl_limitation",
    "answer_extraction",
    "search_failed",
]

BIN_DESCRIPTIONS = {
    "missing_abstraction": "No subroutine in the library matches the required operation",
    "wrong_abstraction": "Planner chose the wrong subroutine (correct one exists but was not selected)",
    "wrong_bindings": "Right subroutine selected but wrong variable bindings supplied",
    "wrong_order": "Right subroutines selected but wrong composition order",
    "spurious_subroutine": "Plan includes a subroutine that does not contribute to the answer",
    "dsl_limitation": "Executor cannot handle the math operation needed (e.g. symbolic, trig)",
    "answer_extraction": "Plan executed correctly but answer format does not match gold",
    "search_failed": "MCTS explored the plan space but could not find a working plan",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalise_answer(raw: str) -> Optional[str]:
    """Strip formatting and return a normalised numeric string, or None."""
    if raw is None:
        return None
    s = str(raw).strip()
    # \boxed{...}
    m = re.search(r"\\boxed\{([^}]+)\}", s)
    if m:
        s = m.group(1)
    # #### prefix
    m = re.search(r"####\s*(.*)", s)
    if m:
        s = m.group(1)
    s = s.replace(",", "").replace("$", "").replace("%", "").strip()
    try:
        return str(float(s))
    except (ValueError, TypeError):
        return s.lower().strip() if s else None


def _answers_match(pred: Any, gold: Any) -> bool:
    pn = _normalise_answer(str(pred)) if pred is not None else None
    gn = _normalise_answer(str(gold))
    if pn is None or gn is None:
        return False
    try:
        return abs(float(pn) - float(gn)) < 1e-3
    except (ValueError, TypeError):
        return pn == gn


def _parse_plan_from_record(record: dict) -> Optional[CompositionPlan]:
    """Extract a CompositionPlan from a prediction record."""
    plan_field = record.get("plan")
    if plan_field is None:
        return None
    if isinstance(plan_field, str):
        try:
            plan_field = json.loads(plan_field)
        except (json.JSONDecodeError, TypeError):
            return None
    if isinstance(plan_field, dict):
        calls = plan_field.get("plan", plan_field.get("calls", []))
        if isinstance(calls, list):
            return CompositionPlan(calls=calls)
    if isinstance(plan_field, list):
        return CompositionPlan(calls=plan_field)
    return None


def _extract_sub_ids(plan: Optional[CompositionPlan]) -> List[str]:
    if plan is None:
        return []
    return [c.get("sub_id", "") for c in plan.calls if c.get("sub_id")]


def _plan_depth(plan: Optional[CompositionPlan]) -> int:
    if plan is None:
        return 0
    return plan.num_calls


def _depth_bucket(depth: int) -> str:
    if depth <= 1:
        return "1-call"
    elif depth == 2:
        return "2-call"
    else:
        return "3+-call"


# ---------------------------------------------------------------------------
# Error-message pattern matching for heuristic classification
# ---------------------------------------------------------------------------

_PATTERN_UNKNOWN_SUB = re.compile(r"Unknown subroutine '([^']*)'", re.IGNORECASE)
_PATTERN_BIND_FAIL = re.compile(r"Cannot bind slot '([^']*)'", re.IGNORECASE)
_PATTERN_MISSING_BIND = re.compile(r"Missing binding for slot '([^']*)'", re.IGNORECASE)
_PATTERN_EXEC_STEP = re.compile(r"Step \d+ \(([^)]*)\):\s*(.*)", re.IGNORECASE)
_PATTERN_COERCE = re.compile(r"Cannot coerce", re.IGNORECASE)
_PATTERN_MAX_STEPS = re.compile(r"Max (steps|calls) exceeded", re.IGNORECASE)

# DSL limitation indicators in execution errors
_DSL_LIMIT_KEYWORDS = [
    "name 'sin' is not defined",
    "name 'cos' is not defined",
    "name 'tan' is not defined",
    "name 'pi' is not defined",
    "name 'factorial' is not defined",
    "name 'comb' is not defined",
    "name 'perm' is not defined",
    "name 'gcd' is not defined",
    "name 'lcm' is not defined",
    "name 'mod' is not defined",
    "name 'frac' is not defined",
    "name 'Rational' is not defined",
    "complex",
    "imaginary",
    "is not defined",
    "unsupported operand",
    "division by zero",
    "math domain error",
    "overflow",
]

# MCTS search failure indicators
_SEARCH_FAIL_KEYWORDS = [
    "search_failed",
    "no solution found",
    "mcts",
    "simulations",
    "max_simulations",
    "solutions_found.*0",
    "timeout",
]


def _classify_failure(
    record: dict,
    plan: Optional[CompositionPlan],
    library: Optional[SubroutineLibrary],
    executor: Executor,
) -> str:
    """Heuristic classification of a single failure into one of 8 bins."""

    exec_log = record.get("execution_log", "")
    if isinstance(exec_log, dict):
        exec_log = json.dumps(exec_log)
    exec_log_lower = exec_log.lower() if exec_log else ""
    error_field = record.get("error", "")
    if isinstance(error_field, dict):
        error_field = json.dumps(error_field)
    error_lower = error_field.lower() if error_field else ""
    method = record.get("method", "")
    pred = record.get("predicted_answer")
    gold = record.get("gold_answer")

    combined = f"{exec_log_lower} {error_lower}"

    # 1. search_failed: MCTS explored but found nothing
    if method in ("mcts", "search", "mcts_compose"):
        for kw in _SEARCH_FAIL_KEYWORDS:
            if re.search(kw, combined):
                return "search_failed"
        stats = record.get("search_stats", {})
        if isinstance(stats, dict) and stats.get("solutions_found", 1) == 0:
            return "search_failed"

    # 2. answer_extraction: execution succeeded, numeric result exists, but
    #    the predicted answer doesn't match gold due to formatting
    if pred is not None and gold is not None:
        pn = _normalise_answer(str(pred))
        gn = _normalise_answer(str(gold))
        if pn is not None and gn is not None:
            try:
                pf, gf = float(pn), float(gn)
                # Close but not matching (rounding / formatting)
                if abs(pf - gf) < max(abs(gf) * 0.02, 1.0) and abs(pf - gf) >= 1e-3:
                    return "answer_extraction"
                # Integer vs float mismatch
                if pf == int(pf) and gf == int(gf) and int(pf) != int(gf):
                    pass  # not extraction, actual wrong answer
                elif pn != gn and abs(pf - gf) < 1e-3:
                    return "answer_extraction"
            except (ValueError, TypeError):
                # String comparison: if very similar, it's extraction
                if pn is not None and gn is not None and pn != gn:
                    # Check if one is a substring of the other
                    if pn in gn or gn in pn:
                        return "answer_extraction"

    # 3. No plan at all (planner couldn't produce valid JSON)
    if plan is None:
        return "missing_abstraction"

    sub_ids = _extract_sub_ids(plan)

    # 4. Unknown subroutine referenced
    if _PATTERN_UNKNOWN_SUB.search(combined):
        return "missing_abstraction"

    # 5. DSL limitation
    for kw in _DSL_LIMIT_KEYWORDS:
        if kw.lower() in combined:
            return "dsl_limitation"

    # 6. Binding failures
    if _PATTERN_BIND_FAIL.search(combined) or _PATTERN_MISSING_BIND.search(combined):
        return "wrong_bindings"
    if _PATTERN_COERCE.search(combined):
        return "wrong_bindings"

    # 7. wrong_abstraction vs wrong_order vs spurious_subroutine
    #    These require the library to inspect.
    if library is not None and sub_ids:
        all_lib_ids = set(library.subroutines.keys())
        used_ids = set(sub_ids)
        referenced_but_missing = used_ids - all_lib_ids
        if referenced_but_missing:
            return "missing_abstraction"

        # Check for spurious subroutines: high-frequency, low-MDL-gain subs
        # that appear in the plan but whose removal would not change (or would
        # improve) the output
        supports = sorted(s.support for s in library.subroutines.values())
        gains = sorted(s.mdl_gain for s in library.subroutines.values())
        med_support = supports[len(supports) // 2] if supports else 0
        med_gain = gains[len(gains) // 2] if gains else 0.0

        spurious_candidates = []
        for sid in sub_ids:
            sub = library.get(sid)
            if sub is None:
                continue
            if sub.support >= med_support and sub.mdl_gain <= med_gain:
                spurious_candidates.append(sid)

        # If more than half the plan is spurious candidates, classify as spurious
        if len(spurious_candidates) > 0 and len(spurious_candidates) >= len(sub_ids) / 2:
            return "spurious_subroutine"

        # wrong_order: same set of subroutines might work in a different order
        if len(sub_ids) >= 2:
            # Check if the sub_ids used are plausible but the error is from a
            # step execution failure (which often indicates ordering issues)
            if _PATTERN_EXEC_STEP.search(combined):
                return "wrong_order"

        # wrong_abstraction: execution failed and the subs used don't match the
        # problem's needed operation
        if "Execution failed" in (exec_log or "") or "Execution failed" in (error_field or ""):
            return "wrong_abstraction"

    # 8. Catch remaining execution failures
    if _PATTERN_EXEC_STEP.search(combined):
        return "dsl_limitation"

    if _PATTERN_MAX_STEPS.search(combined):
        return "wrong_order"

    # Default: if we have a plan but it produced the wrong answer, most likely
    # the planner selected wrong subroutines
    if plan is not None and len(sub_ids) > 0:
        return "wrong_abstraction"

    return "missing_abstraction"


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------


def analyze_failures(
    predictions: List[dict],
    library: Optional[SubroutineLibrary],
    n_sample: int,
) -> dict:
    """Run failure analysis on prediction records.

    Returns a structured report dict.
    """
    executor = Executor()

    # Filter to failures only
    failures = []
    successes = 0
    for rec in predictions:
        pred = rec.get("predicted_answer")
        gold = rec.get("gold_answer")
        if _answers_match(pred, gold):
            successes += 1
        else:
            failures.append(rec)

    total = len(predictions)
    n_failures = len(failures)

    # Sample if needed
    if n_sample > 0 and n_sample < n_failures:
        random.seed(42)
        sampled = random.sample(failures, n_sample)
    else:
        sampled = failures
        n_sample = n_failures

    # Classify each failure
    bin_counts: Dict[str, int] = Counter()
    bin_examples: Dict[str, List[dict]] = defaultdict(list)
    per_record: List[dict] = []

    # Depth / subroutine-count breakdowns
    depth_total: Dict[str, int] = Counter()
    depth_fail: Dict[str, int] = Counter()
    nsub_total: Dict[int, int] = Counter()
    nsub_fail: Dict[int, int] = Counter()

    # Compute depth/nsub stats over ALL predictions (not just sampled failures)
    for rec in predictions:
        plan = _parse_plan_from_record(rec)
        d = _plan_depth(plan)
        bucket = _depth_bucket(d)
        depth_total[bucket] += 1
        n_distinct = len(set(_extract_sub_ids(plan)))
        nsub_total[n_distinct] += 1
        if not _answers_match(rec.get("predicted_answer"), rec.get("gold_answer")):
            depth_fail[bucket] += 1
            nsub_fail[n_distinct] += 1

    # Classify sampled failures
    for rec in sampled:
        plan = _parse_plan_from_record(rec)
        category = _classify_failure(rec, plan, library, executor)
        bin_counts[category] += 1

        example_entry = {
            "category": category,
            "problem": (rec.get("problem", "") or "")[:200],
            "gold_answer": str(rec.get("gold_answer", "")),
            "predicted_answer": str(rec.get("predicted_answer", "")),
            "method": rec.get("method", ""),
            "plan_depth": _plan_depth(plan),
            "sub_ids": _extract_sub_ids(plan),
        }
        per_record.append(example_entry)

        # Keep first example per bin as representative
        if len(bin_examples[category]) < 1:
            bin_examples[category].append({
                "problem": (rec.get("problem", "") or "")[:300],
                "gold_answer": str(rec.get("gold_answer", "")),
                "predicted_answer": str(rec.get("predicted_answer", "")),
                "method": rec.get("method", ""),
                "execution_log": (str(rec.get("execution_log", "")) or "")[:300],
                "plan_depth": _plan_depth(plan),
                "sub_ids": _extract_sub_ids(plan),
            })

    # Build bin summary
    bins_summary = []
    for b in FAILURE_BINS:
        count = bin_counts.get(b, 0)
        pct = round(count / max(n_sample, 1) * 100, 1)
        entry = {
            "bin": b,
            "description": BIN_DESCRIPTIONS[b],
            "count": count,
            "pct_of_failures": pct,
            "representative_example": bin_examples.get(b, [None])[0],
        }
        bins_summary.append(entry)

    # Depth breakdown
    depth_breakdown = {}
    for bucket in ["1-call", "2-call", "3+-call"]:
        t = depth_total.get(bucket, 0)
        f = depth_fail.get(bucket, 0)
        depth_breakdown[bucket] = {
            "total": t,
            "failures": f,
            "failure_rate": round(f / max(t, 1), 4),
        }

    # Distinct-subroutine breakdown
    nsub_breakdown = {}
    for k in sorted(set(list(nsub_total.keys()) + list(nsub_fail.keys()))):
        t = nsub_total.get(k, 0)
        f = nsub_fail.get(k, 0)
        nsub_breakdown[str(k)] = {
            "total": t,
            "failures": f,
            "failure_rate": round(f / max(t, 1), 4),
        }

    report = {
        "summary": {
            "total_predictions": total,
            "successes": successes,
            "failures": n_failures,
            "overall_failure_rate": round(n_failures / max(total, 1), 4),
            "sampled_for_analysis": n_sample,
        },
        "bins": bins_summary,
        "failure_rate_by_depth": depth_breakdown,
        "failure_rate_by_n_distinct_subs": nsub_breakdown,
        "per_record_classifications": per_record,
    }
    return report


def print_summary(report: dict):
    """Print a human-readable summary to stdout."""
    s = report["summary"]
    print(f"{'='*70}")
    print(f"  FAILURE ANALYSIS")
    print(f"{'='*70}")
    print(f"  Total predictions:   {s['total_predictions']}")
    print(f"  Successes:           {s['successes']}")
    print(f"  Failures:            {s['failures']} ({s['overall_failure_rate']*100:.1f}%)")
    print(f"  Sampled for analysis:{s['sampled_for_analysis']}")
    print()

    print(f"  {'Bin':<25s} {'Count':>6s} {'%':>7s}")
    print(f"  {'-'*25} {'-'*6} {'-'*7}")
    for b in report["bins"]:
        print(f"  {b['bin']:<25s} {b['count']:>6d} {b['pct_of_failures']:>6.1f}%")
    print()

    print(f"  Failure rate by plan depth:")
    for bucket, data in report["failure_rate_by_depth"].items():
        print(f"    {bucket:<10s}  {data['failures']}/{data['total']}  ({data['failure_rate']*100:.1f}%)")
    print()

    print(f"  Failure rate by # distinct subroutines:")
    for k, data in report["failure_rate_by_n_distinct_subs"].items():
        print(f"    n={k:<3s}  {data['failures']}/{data['total']}  ({data['failure_rate']*100:.1f}%)")
    print()

    print(f"  Representative examples:")
    for b in report["bins"]:
        ex = b.get("representative_example")
        if ex is None:
            continue
        print(f"    [{b['bin']}]")
        print(f"      problem:   {ex['problem'][:100]}")
        print(f"      gold:      {ex['gold_answer']}")
        print(f"      predicted: {ex['predicted_answer']}")
        print(f"      depth:     {ex['plan_depth']}  subs: {ex['sub_ids']}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Categorize prediction failures into analysis bins"
    )
    parser.add_argument(
        "--predictions", type=str, required=True,
        help="Path to predictions JSON (list of records with problem, gold_answer, "
             "predicted_answer, plan, execution_log, method)",
    )
    parser.add_argument(
        "--library", type=str, default=None,
        help="Path to subroutine library JSON (optional, enables deeper classification)",
    )
    parser.add_argument(
        "--n_sample", type=int, default=150,
        help="Number of failures to sample for classification (0 = all)",
    )
    parser.add_argument(
        "--output", type=str, default="results/failure_analysis.json",
        help="Output path for the JSON report",
    )
    args = parser.parse_args()

    # Load predictions
    with open(args.predictions) as f:
        predictions = json.load(f)
    if isinstance(predictions, dict):
        # Handle case where predictions are nested under a key
        for key in ("predictions", "results", "data", "records"):
            if key in predictions and isinstance(predictions[key], list):
                predictions = predictions[key]
                break
        else:
            # Flatten from per-dataset structure
            flat = []
            for k, v in predictions.items():
                if isinstance(v, list):
                    flat.extend(v)
            if flat:
                predictions = flat

    if not isinstance(predictions, list):
        print(f"ERROR: predictions file does not contain a list (got {type(predictions).__name__})")
        sys.exit(1)

    print(f"Loaded {len(predictions)} prediction records from {args.predictions}")

    # Load library (optional)
    library = None
    if args.library and os.path.exists(args.library):
        library = SubroutineLibrary.load(args.library)
        print(f"Loaded library: {library.size} subroutines")

    report = analyze_failures(predictions, library, args.n_sample)

    # Write JSON
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nJSON report written to {out_path}")

    # Print human-readable summary
    print()
    print_summary(report)


if __name__ == "__main__":
    main()
