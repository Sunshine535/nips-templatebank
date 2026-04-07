#!/usr/bin/env python3
"""Compression-as-diagnostic sweep.

Varies library size and split severity, then regresses MCD accuracy
against compression ratio and competing predictors.

Usage:
    python scripts/run_compression_sweep.py \
        --dataset gsm8k \
        --library_sizes 4 8 16 32 \
        --split_seeds 42 123 456 \
        --results_dir results/compression_sweep
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.template_dsl import Program, SubroutineLibrary, Executor, CompositionPlan

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_compression_ratio(programs: list, library: SubroutineLibrary,
                              plans: list) -> dict:
    """Compute compression metrics for a set of programs under a library.

    Returns dict with per-example and aggregate compression stats.
    """
    flat_lengths = []
    compressed_lengths = []
    ratios = []

    for item in programs:
        prog = Program.from_dict(item["program"])
        flat_len = sum(len(s.expr) for s in prog.steps) + len(prog.steps) * 10
        flat_lengths.append(flat_len)

        # Find matching plan if available
        pid = item.get("program_id", prog.program_id)
        matching_plan = None
        for p in plans:
            if p.get("program_id") == pid:
                matching_plan = p
                break

        if matching_plan and "plan_data" in matching_plan:
            plan_calls = matching_plan["plan_data"].get("plan", [])
            # Compressed length = sum of call descriptions
            comp_len = sum(
                len(str(c.get("sub_id", ""))) + len(str(c.get("bindings", {})))
                for c in plan_calls
            )
            comp_len = max(comp_len, 1)
        else:
            comp_len = flat_len  # no compression

        compressed_lengths.append(comp_len)
        ratios.append(flat_len / max(comp_len, 1))

    return {
        "mean_flat_length": float(np.mean(flat_lengths)) if flat_lengths else 0,
        "mean_compressed_length": float(np.mean(compressed_lengths)) if compressed_lengths else 0,
        "mean_compression_ratio": float(np.mean(ratios)) if ratios else 1.0,
        "median_compression_ratio": float(np.median(ratios)) if ratios else 1.0,
        "std_compression_ratio": float(np.std(ratios)) if ratios else 0.0,
        "n_examples": len(programs),
        "per_example_ratios": ratios,
    }


def compute_predictors(programs: list, library: SubroutineLibrary,
                       plans: list, eval_results: dict) -> dict:
    """Compute all candidate predictors for the regression analysis."""
    compression = compute_compression_ratio(programs, library, plans)

    return {
        "compression_ratio": compression["mean_compression_ratio"],
        "library_size": library.size,
        "mean_trace_length": compression["mean_flat_length"],
        "mean_plan_depth": np.mean([
            len(p.get("plan_data", {}).get("plan", []))
            for p in plans if "plan_data" in p
        ]) if plans else 0,
        "teacher_accuracy": eval_results.get("teacher_accuracy", 0),
        "library_total_support": sum(
            s.support for s in library.subroutines.values()
        ),
        "mean_mdl_gain": np.mean([
            s.mdl_gain for s in library.subroutines.values()
        ]) if library.subroutines else 0,
        # Target
        "mcd_accuracy": eval_results.get("accuracy", 0),
        "mcd_severity": eval_results.get("mcd_severity", "unknown"),
    }


def run_regression(datapoints: list) -> dict:
    """Run multivariate regression: predictors → MCD accuracy.

    Returns regression coefficients, R², partial R², p-values.
    """
    if len(datapoints) < 5:
        logger.warning("Only %d datapoints — regression unreliable", len(datapoints))

    predictor_names = [
        "compression_ratio", "library_size", "mean_trace_length",
        "mean_plan_depth", "teacher_accuracy", "mean_mdl_gain",
    ]

    y = np.array([d["mcd_accuracy"] for d in datapoints])
    X = np.array([[d.get(p, 0) for p in predictor_names] for d in datapoints])

    # Standardize
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std == 0] = 1
    X_norm = (X - X_mean) / X_std

    y_mean = y.mean()
    y_norm = y - y_mean

    results = {"n_datapoints": len(datapoints), "predictors": {}}

    # Full OLS
    try:
        beta = np.linalg.lstsq(X_norm, y_norm, rcond=None)[0]
        y_pred = X_norm @ beta
        ss_res = np.sum((y_norm - y_pred) ** 2)
        ss_tot = np.sum(y_norm ** 2)
        r_squared = 1 - ss_res / max(ss_tot, 1e-10)
        results["full_r_squared"] = float(r_squared)
        results["adjusted_r_squared"] = float(
            1 - (1 - r_squared) * (len(y) - 1) / max(len(y) - len(predictor_names) - 1, 1)
        )
    except np.linalg.LinAlgError:
        results["full_r_squared"] = 0
        results["adjusted_r_squared"] = 0
        beta = np.zeros(len(predictor_names))

    # Per-predictor univariate correlation + standardized coefficient
    for i, name in enumerate(predictor_names):
        x_i = X_norm[:, i]
        corr = np.corrcoef(x_i, y_norm)[0, 1] if len(x_i) > 1 else 0
        results["predictors"][name] = {
            "standardized_coefficient": float(beta[i]),
            "pearson_r": float(corr) if np.isfinite(corr) else 0,
            "r_squared_univariate": float(corr ** 2) if np.isfinite(corr) else 0,
        }

        # Partial R²: drop this predictor, measure R² decrease
        mask = [j for j in range(len(predictor_names)) if j != i]
        X_reduced = X_norm[:, mask]
        try:
            beta_r = np.linalg.lstsq(X_reduced, y_norm, rcond=None)[0]
            y_pred_r = X_reduced @ beta_r
            ss_res_r = np.sum((y_norm - y_pred_r) ** 2)
            partial_r2 = (ss_res_r - ss_res) / max(ss_res_r, 1e-10)
            results["predictors"][name]["partial_r_squared"] = float(max(partial_r2, 0))
        except np.linalg.LinAlgError:
            results["predictors"][name]["partial_r_squared"] = 0

    # Determine strongest predictor
    strongest = max(
        results["predictors"].items(),
        key=lambda x: x[1]["partial_r_squared"]
    )
    results["strongest_predictor"] = strongest[0]
    results["strongest_partial_r2"] = strongest[1]["partial_r_squared"]

    return results


def collect_sweep_results(results_dir: str) -> list:
    """Collect all sweep condition results from disk."""
    datapoints = []
    sweep_dir = Path(results_dir)
    if not sweep_dir.exists():
        return datapoints

    for condition_dir in sorted(sweep_dir.iterdir()):
        if not condition_dir.is_dir():
            continue
        meta_path = condition_dir / "sweep_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                datapoints.append(json.load(f))

    return datapoints


def main():
    parser = argparse.ArgumentParser(description="Compression-as-diagnostic sweep")
    parser.add_argument("--dataset", default="gsm8k", choices=["gsm8k", "math"])
    parser.add_argument("--library_sizes", nargs="+", type=int, default=[4, 8, 16, 32])
    parser.add_argument("--split_seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--results_dir", default="results/compression_sweep")
    parser.add_argument("--analyze_only", action="store_true",
                        help="Skip experiments, just run regression on existing results")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    if args.analyze_only:
        datapoints = collect_sweep_results(args.results_dir)
        if not datapoints:
            logger.error("No sweep results found in %s", args.results_dir)
            return

        logger.info("Collected %d datapoints for regression", len(datapoints))
        regression = run_regression(datapoints)

        output_path = os.path.join(args.results_dir, "regression_results.json")
        with open(output_path, "w") as f:
            json.dump(regression, f, indent=2)

        logger.info("=== Regression Results ===")
        logger.info("Full R²: %.4f (adjusted: %.4f)",
                    regression["full_r_squared"], regression["adjusted_r_squared"])
        logger.info("Strongest predictor: %s (partial R²=%.4f)",
                    regression["strongest_predictor"], regression["strongest_partial_r2"])
        for name, stats in regression["predictors"].items():
            logger.info("  %s: coeff=%.4f, r=%.4f, partial_R²=%.4f",
                        name, stats["standardized_coefficient"],
                        stats["pearson_r"], stats["partial_r_squared"])
        return

    # Generate experiment conditions
    conditions = []
    for lib_size in args.library_sizes:
        for split_seed in args.split_seeds:
            conditions.append({
                "library_size": lib_size,
                "split_seed": split_seed,
                "dataset": args.dataset,
            })

    logger.info("Sweep: %d conditions (%d lib sizes x %d split seeds)",
                len(conditions), len(args.library_sizes), len(args.split_seeds))

    # Print experiment commands for each condition
    for cond in conditions:
        cond_name = f"L{cond['library_size']}_seed{cond['split_seed']}"
        cond_dir = os.path.join(args.results_dir, cond_name)
        logger.info("Condition %s: library_size=%d, split_seed=%d → %s",
                    cond_name, cond["library_size"], cond["split_seed"], cond_dir)

    logger.info("\nRun each condition's training + eval, save sweep_meta.json "
                "per condition, then re-run with --analyze_only")


if __name__ == "__main__":
    main()
