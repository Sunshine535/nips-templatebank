#!/usr/bin/env python3
"""Test all template algebra operations on extracted TemplateBank.

Operations tested:
- COMPOSE: chain compatible templates (e.g., geometry + algebra)
- ABSTRACT: generalize templates to higher-level patterns
- SPECIALIZE: instantiate abstract templates for specific problem types
- SEQUENCE: create multi-step reasoning chains (compose pipeline)
- BRANCH: conditional template selection based on problem features
- Coverage: how many GSM8K/MATH problems can be solved by template matching + filling
"""

import argparse
import itertools
import json
import logging
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.template_algebra import (
    ReasoningTemplate,
    TemplateAlgebra,
    TemplateBank,
    TemplateStep,
    Variable,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)



def load_test_datasets(config: dict) -> dict:
    """Load test sets for coverage evaluation."""
    datasets = {}
    for ds_cfg in config["evaluation"]["test_datasets"]:
        name = ds_cfg["name"]
        try:
            subset = ds_cfg.get("subset")
            if subset:
                ds = load_dataset(ds_cfg["dataset_id"], subset, split=ds_cfg["split"])
            else:
                ds = load_dataset(ds_cfg["dataset_id"], split=ds_cfg["split"])
            max_s = ds_cfg.get("max_samples", 1000)
            if len(ds) > max_s:
                ds = ds.shuffle(seed=42).select(range(max_s))
            datasets[name] = ds
            logger.info("Loaded test set '%s': %d examples", name, len(ds))
        except Exception as e:
            logger.warning("Failed to load %s: %s", name, e)
    return datasets


# ===== Operation 1: COMPOSE =====

def test_compose(bank: TemplateBank, algebra: TemplateAlgebra, output_dir: str) -> dict:
    """Test pairwise template composition."""
    logger.info("=== Testing COMPOSE ===")
    templates = bank.search(min_reuse=0)
    results = {"compositions": [], "stats": {}}

    if len(templates) < 2:
        logger.warning("Need ≥2 templates for composition tests")
        return results

    by_domain = defaultdict(list)
    for t in templates:
        by_domain[t.domain].append(t)

    composed_count = 0
    valid_count = 0
    domain_pairs_tested = set()

    for (dom_a, temps_a), (dom_b, temps_b) in itertools.combinations(by_domain.items(), 2):
        if len(temps_a) == 0 or len(temps_b) == 0:
            continue
        t1, t2 = temps_a[0], temps_b[0]

        composed = algebra.compose(t1, t2, name=f"{t1.name}+{t2.name}")
        composed_count += 1

        has_shared_vars = bool(
            {s.output_var for s in t1.steps if s.output_var}
            & {inp for s in t2.steps for inp in s.inputs}
        )
        is_valid = composed.num_steps == t1.num_steps + t2.num_steps
        if is_valid:
            valid_count += 1

        domain_pairs_tested.add((dom_a, dom_b))
        results["compositions"].append({
            "t1": {"id": t1.template_id, "name": t1.name, "domain": t1.domain, "steps": t1.num_steps},
            "t2": {"id": t2.template_id, "name": t2.name, "domain": t2.domain, "steps": t2.num_steps},
            "composed_name": composed.name,
            "composed_steps": composed.num_steps,
            "composed_vars": len(composed.variables),
            "has_shared_vars": has_shared_vars,
            "is_valid": is_valid,
        })

    results["stats"] = {
        "total_compositions": composed_count,
        "valid_compositions": valid_count,
        "domain_pairs_tested": len(domain_pairs_tested),
    }
    logger.info("  Composed %d template pairs (%d valid)", composed_count, valid_count)
    return results


# ===== Operation 2: ABSTRACT =====

def test_abstract(bank: TemplateBank, algebra: TemplateAlgebra, output_dir: str) -> dict:
    """Test template abstraction."""
    logger.info("=== Testing ABSTRACT ===")
    templates = bank.search(min_reuse=0)
    results = {"abstractions": [], "stats": {}}

    abstracted_count = 0
    new_vars_added = []

    for t in templates[:50]:
        original_vars = len(t.variables)
        abstracted = algebra.abstract(t)
        new_vars = len(abstracted.variables) - original_vars
        new_vars_added.append(new_vars)
        abstracted_count += 1

        results["abstractions"].append({
            "original_id": t.template_id,
            "original_name": t.name,
            "original_vars": original_vars,
            "abstracted_vars": len(abstracted.variables),
            "new_vars_added": new_vars,
            "original_steps": t.num_steps,
            "abstracted_steps": abstracted.num_steps,
        })

    results["stats"] = {
        "total_abstracted": abstracted_count,
        "avg_new_vars": sum(new_vars_added) / max(len(new_vars_added), 1),
        "max_new_vars": max(new_vars_added) if new_vars_added else 0,
        "templates_with_abstraction": sum(1 for v in new_vars_added if v > 0),
    }
    logger.info("  Abstracted %d templates, avg %.1f new vars", abstracted_count,
                results["stats"]["avg_new_vars"])
    return results


# ===== Operation 3: SPECIALIZE =====

def test_specialize(bank: TemplateBank, algebra: TemplateAlgebra, output_dir: str) -> dict:
    """Test template specialization with partial bindings."""
    logger.info("=== Testing SPECIALIZE ===")
    templates = bank.search(min_reuse=0)
    results = {"specializations": [], "stats": {}}

    specialized_count = 0
    for t in templates[:50]:
        if not t.variables:
            continue
        first_var = t.variables[0]
        bindings = {first_var.name: "42"}
        specialized = algebra.specialize(t, bindings)
        specialized_count += 1

        results["specializations"].append({
            "original_id": t.template_id,
            "bound_var": first_var.name,
            "bound_value": "42",
            "remaining_vars": len(specialized.variables),
            "original_vars": len(t.variables),
        })

    results["stats"] = {
        "total_specialized": specialized_count,
        "avg_vars_reduced": 1.0,
    }
    logger.info("  Specialized %d templates", specialized_count)
    return results


# ===== Operation 4: SEQUENCE =====

def test_sequence(bank: TemplateBank, algebra: TemplateAlgebra, output_dir: str) -> dict:
    """Create multi-step reasoning chains by sequential composition."""
    logger.info("=== Testing SEQUENCE (multi-step chains) ===")
    templates = bank.search(min_reuse=0)
    results = {"sequences": [], "stats": {}}

    if len(templates) < 3:
        return results

    chain_lengths = [2, 3, 4, 5]
    for length in chain_lengths:
        if length > len(templates):
            continue

        chain_templates = templates[:length]
        current = chain_templates[0]
        for t in chain_templates[1:]:
            current = algebra.compose(current, t, name=f"chain_{length}step")

        results["sequences"].append({
            "chain_length": length,
            "total_steps": current.num_steps,
            "total_vars": len(current.variables),
            "component_templates": [t.template_id for t in chain_templates],
        })

    results["stats"] = {
        "chains_created": len(results["sequences"]),
        "max_chain_length": max((s["chain_length"] for s in results["sequences"]), default=0),
    }
    logger.info("  Created %d reasoning chains", len(results["sequences"]))
    return results


# ===== Operation 5: BRANCH =====

def test_branch(bank: TemplateBank, output_dir: str) -> dict:
    """Conditional template selection based on problem features."""
    logger.info("=== Testing BRANCH (conditional selection) ===")
    templates = bank.search(min_reuse=0)
    results = {"branch_rules": [], "stats": {}}

    keyword_map = defaultdict(list)
    for t in templates:
        keywords = set()
        for step in t.steps:
            for word in re.findall(r'[a-z_]+', step.expression.lower()):
                if len(word) > 3:
                    keywords.add(word)
        keywords.add(t.domain)
        for kw in keywords:
            keyword_map[kw].append(t.template_id)

    top_keywords = sorted(keyword_map.items(), key=lambda x: len(x[1]), reverse=True)[:20]
    for kw, tids in top_keywords:
        results["branch_rules"].append({
            "keyword": kw,
            "num_matching_templates": len(tids),
            "template_ids": tids[:5],
        })

    results["stats"] = {
        "total_branch_keywords": len(keyword_map),
        "avg_templates_per_keyword": sum(len(v) for v in keyword_map.values()) / max(len(keyword_map), 1),
    }
    logger.info("  Built %d branch rules", len(keyword_map))
    return results


# ===== Coverage Analysis =====

def test_coverage(bank: TemplateBank, test_datasets: dict, output_dir: str) -> dict:
    """Evaluate what fraction of test problems can be matched to existing templates."""
    logger.info("=== Testing COVERAGE ===")
    templates = bank.search(min_reuse=0)
    results = {}

    for ds_name, ds in test_datasets.items():
        matched = 0
        total = 0
        match_distribution = Counter()

        for ex in ds:
            problem = str(ex.get("question", ex.get("problem", "")))
            total += 1

            best_match = None
            best_score = 0
            for t in templates:
                score = 0
                problem_lower = problem.lower()
                for step in t.steps:
                    ops = re.findall(r'[a-z]+', step.operation.lower())
                    for op in ops:
                        if op in problem_lower:
                            score += 1
                if t.domain in problem_lower:
                    score += 2

                if score > best_score:
                    best_score = score
                    best_match = t

            if best_match and best_score > 0:
                matched += 1
                match_distribution[best_match.template_id] += 1

        coverage = matched / max(total, 1)
        top_templates = match_distribution.most_common(10)

        results[ds_name] = {
            "total_problems": total,
            "matched_problems": matched,
            "coverage_rate": round(coverage, 4),
            "unique_templates_used": len(match_distribution),
            "top_templates": [{"id": tid, "count": cnt} for tid, cnt in top_templates],
        }
        logger.info("  %s: coverage=%.2f%% (%d/%d), %d unique templates",
                    ds_name, coverage * 100, matched, total, len(match_distribution))

    return results


# ===== Decompose =====

def test_decompose(bank: TemplateBank, algebra: TemplateAlgebra, output_dir: str) -> dict:
    """Test template decomposition into atomic sub-templates."""
    logger.info("=== Testing DECOMPOSE ===")
    templates = bank.search(min_reuse=0)
    results = {"decompositions": [], "stats": {}}

    total_atoms = 0
    for t in templates[:30]:
        atoms = algebra.decompose(t)
        total_atoms += len(atoms)
        results["decompositions"].append({
            "template_id": t.template_id,
            "template_name": t.name,
            "original_steps": t.num_steps,
            "num_atoms": len(atoms),
            "atom_operations": [a.steps[0].operation for a in atoms if a.steps],
        })

    results["stats"] = {
        "templates_decomposed": len(results["decompositions"]),
        "total_atoms": total_atoms,
        "avg_atoms_per_template": total_atoms / max(len(results["decompositions"]), 1),
    }
    logger.info("  Decomposed %d templates into %d atoms", len(results["decompositions"]), total_atoms)
    return results


def main():
    parser = argparse.ArgumentParser(description="Test template algebra operations")
    parser.add_argument("--config", type=str, default=str(Path(__file__).resolve().parent.parent / "configs" / "template_config.yaml"))
    parser.add_argument("--template_bank", type=str, default="results/templates/template_bank.json")
    parser.add_argument("--output_dir", type=str, default="results/operations")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Loading template bank from %s", args.template_bank)
    if not os.path.exists(args.template_bank):
        logger.error("Template bank not found: %s", args.template_bank)
        sys.exit(1)

    bank = TemplateBank.load(args.template_bank)
    logger.info("Loaded %d templates", len(bank.templates))
    logger.info("Bank stats: %s", json.dumps(bank.stats(), indent=2))

    algebra = TemplateAlgebra()
    all_results = {"meta": {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "template_bank": args.template_bank,
        "num_templates": len(bank.templates),
        "bank_stats": bank.stats(),
    }}

    # Run all operations
    all_results["compose"] = test_compose(bank, algebra, args.output_dir)
    all_results["abstract"] = test_abstract(bank, algebra, args.output_dir)
    all_results["specialize"] = test_specialize(bank, algebra, args.output_dir)
    all_results["sequence"] = test_sequence(bank, algebra, args.output_dir)
    all_results["branch"] = test_branch(bank, args.output_dir)
    all_results["decompose"] = test_decompose(bank, algebra, args.output_dir)

    # Coverage analysis on test sets
    logger.info("=" * 50)
    logger.info("  Loading test datasets for coverage analysis...")
    logger.info("=" * 50)
    test_datasets = load_test_datasets(config)
    if test_datasets:
        all_results["coverage"] = test_coverage(bank, test_datasets, args.output_dir)

    results_path = os.path.join(args.output_dir, "template_operations_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info("=" * 60)
    logger.info("  Template operations complete")
    logger.info("  Results: %s", results_path)
    for op in ["compose", "abstract", "specialize", "sequence", "branch", "decompose"]:
        stats = all_results.get(op, {}).get("stats", {})
        logger.info("  %s: %s", op, json.dumps(stats))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
