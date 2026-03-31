"""Maximum Compound Divergence (MCD) split builder.

Implements CFQ-style compositional split:
- Atoms: primitive ops, arity, type signatures
- Compounds: parent-child bigrams, sibling bigrams, top-call bigrams
- Maximize compound divergence while keeping atom divergence low.

Reference: Keysers et al. (2020) "Measuring Compositional Generalization"
"""

import json
import logging
import random
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.spatial.distance import jensenshannon

logger = logging.getLogger(__name__)


def extract_atoms(plan_data: dict) -> Counter:
    """Extract atomic features from a composition plan."""
    atoms = Counter()
    calls = plan_data.get("plan", [])
    for call in calls:
        sub_id = call.get("sub_id", "")
        atoms[f"sub:{sub_id}"] += 1
        bindings = call.get("bindings", {})
        atoms[f"arity:{len(bindings)}"] += 1
        for k, v in bindings.items():
            vtype = type(v).__name__
            atoms[f"type:{vtype}"] += 1
    atoms[f"num_calls:{len(calls)}"] += 1
    return atoms


def extract_compounds(plan_data: dict) -> Set[str]:
    """Extract compound features (bigrams, motifs) from a composition plan."""
    compounds = set()
    calls = plan_data.get("plan", [])
    sub_ids = [c.get("sub_id", "") for c in calls]

    for i in range(len(sub_ids) - 1):
        compounds.add(f"bigram:{sub_ids[i]}>{sub_ids[i+1]}")

    if len(sub_ids) >= 3:
        for i in range(len(sub_ids) - 2):
            compounds.add(f"trigram:{sub_ids[i]}>{sub_ids[i+1]}>{sub_ids[i+2]}")

    for i, call in enumerate(calls):
        sub_id = call.get("sub_id", "")
        bindings = call.get("bindings", {})
        for k in bindings:
            compounds.add(f"bind:{sub_id}.{k}")
        compounds.add(f"pos:{i}:{sub_id}")

    return compounds


def compute_atom_tvd(split_a: List[Counter], split_b: List[Counter]) -> float:
    """Total variation distance of atom distributions."""
    total_a, total_b = Counter(), Counter()
    for c in split_a:
        total_a += c
    for c in split_b:
        total_b += c

    all_keys = set(total_a.keys()) | set(total_b.keys())
    if not all_keys:
        return 0.0

    sum_a = sum(total_a.values()) or 1
    sum_b = sum(total_b.values()) or 1
    tvd = 0.0
    for k in all_keys:
        tvd += abs(total_a[k] / sum_a - total_b[k] / sum_b)
    return tvd / 2


def compute_compound_divergence(train_compounds: List[Set[str]], test_compounds: List[Set[str]]) -> float:
    """Fraction of test compounds unseen in train."""
    train_all = set()
    for cs in train_compounds:
        train_all |= cs

    test_all = set()
    for cs in test_compounds:
        test_all |= cs

    if not test_all:
        return 0.0
    unseen = test_all - train_all
    return len(unseen) / len(test_all)


def build_mcd_split(
    examples: List[dict],
    train_ratio: float = 0.6,
    dev_ratio: float = 0.2,
    test_ratio: float = 0.2,
    max_atom_tvd: float = 0.02,
    min_unseen_compounds: float = 0.40,
    num_trials: int = 500,
    seed: int = 42,
) -> Dict[str, List[int]]:
    """Build an MCD split that maximizes compound divergence.

    Args:
        examples: List of dicts with "plan" field (composition plan).
        train_ratio, dev_ratio, test_ratio: Split proportions.
        max_atom_tvd: Maximum atom total variation distance allowed.
        min_unseen_compounds: Minimum fraction of unseen compounds in test.
        num_trials: Number of random trials.
        seed: Random seed.

    Returns:
        Dict with "train", "dev", "test" keys mapping to lists of indices.
    """
    rng = random.Random(seed)
    n = len(examples)
    n_train = int(n * train_ratio)
    n_dev = int(n * dev_ratio)

    all_atoms = [extract_atoms(ex.get("plan_data", ex)) for ex in examples]
    all_compounds = [extract_compounds(ex.get("plan_data", ex)) for ex in examples]

    best_split = None
    best_score = -1.0

    indices = list(range(n))
    for trial in range(num_trials):
        rng.shuffle(indices)
        train_idx = indices[:n_train]
        dev_idx = indices[n_train:n_train + n_dev]
        test_idx = indices[n_train + n_dev:]

        train_atoms = [all_atoms[i] for i in train_idx]
        test_atoms = [all_atoms[i] for i in test_idx]
        atom_tvd = compute_atom_tvd(train_atoms, test_atoms)

        if atom_tvd > max_atom_tvd:
            continue

        train_compounds = [all_compounds[i] for i in train_idx]
        test_compounds = [all_compounds[i] for i in test_idx]
        compound_div = compute_compound_divergence(train_compounds, test_compounds)

        if compound_div < min_unseen_compounds:
            continue

        if compound_div > best_score:
            best_score = compound_div
            best_split = {
                "train": sorted(train_idx),
                "dev": sorted(dev_idx),
                "test": sorted(test_idx),
            }

        if trial % 50 == 0:
            logger.info("  MCD trial %d/%d: best_compound_div=%.4f, atom_tvd=%.4f",
                        trial, num_trials, best_score, atom_tvd)

    if best_split is None:
        logger.warning(
            "No split satisfies atom_tvd < %.3f AND min_unseen_compounds >= %.3f "
            "after %d trials. Relaxing min_unseen_compounds to find best available.",
            max_atom_tvd, min_unseen_compounds, num_trials,
        )
        for trial in range(num_trials):
            rng.shuffle(indices)
            train_idx = indices[:n_train]
            dev_idx = indices[n_train:n_train + n_dev]
            test_idx = indices[n_train + n_dev:]

            train_atoms = [all_atoms[i] for i in train_idx]
            test_atoms = [all_atoms[i] for i in test_idx]
            atom_tvd = compute_atom_tvd(train_atoms, test_atoms)
            if atom_tvd > max_atom_tvd * 2:
                continue

            train_compounds = [all_compounds[i] for i in train_idx]
            test_compounds = [all_compounds[i] for i in test_idx]
            compound_div = compute_compound_divergence(train_compounds, test_compounds)

            if compound_div > best_score:
                best_score = compound_div
                best_split = {
                    "train": sorted(train_idx),
                    "dev": sorted(dev_idx),
                    "test": sorted(test_idx),
                }

    if best_split is None:
        logger.warning("Fallback: random split (no constraint satisfied).")
        rng.shuffle(indices)
        best_split = {
            "train": sorted(indices[:n_train]),
            "dev": sorted(indices[n_train:n_train + n_dev]),
            "test": sorted(indices[n_train + n_dev:]),
        }

    train_atoms = [all_atoms[i] for i in best_split["train"]]
    test_atoms = [all_atoms[i] for i in best_split["test"]]
    train_compounds = [all_compounds[i] for i in best_split["train"]]
    test_compounds = [all_compounds[i] for i in best_split["test"]]

    stats = {
        "n_total": n,
        "n_train": len(best_split["train"]),
        "n_dev": len(best_split["dev"]),
        "n_test": len(best_split["test"]),
        "atom_tvd": round(compute_atom_tvd(train_atoms, test_atoms), 6),
        "compound_divergence": round(compute_compound_divergence(train_compounds, test_compounds), 4),
        "num_trials": num_trials,
        "train_unique_compounds": len(set().union(*train_compounds)) if train_compounds else 0,
        "test_unique_compounds": len(set().union(*test_compounds)) if test_compounds else 0,
    }

    unseen = (set().union(*test_compounds) if test_compounds else set()) - \
             (set().union(*train_compounds) if train_compounds else set())
    stats["unseen_test_compounds"] = len(unseen)
    stats["unseen_compound_ratio"] = round(len(unseen) / max(stats["test_unique_compounds"], 1), 4)

    best_split["stats"] = stats
    logger.info("MCD split built: %s", json.dumps(stats, indent=2))
    return best_split


def save_split(split: dict, path: str):
    with open(path, "w") as f:
        json.dump(split, f, indent=2)


def load_split(path: str) -> dict:
    with open(path) as f:
        return json.load(f)
