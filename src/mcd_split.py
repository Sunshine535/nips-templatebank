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
    # Call count atom (how many subroutine calls total)
    atoms[f"call_count:{len(calls)}"] += 1
    return atoms


def _infer_binding_types(bindings: dict) -> List[str]:
    """Return sorted list of type names for binding values."""
    return sorted(type(v).__name__ for v in bindings.values())


def extract_compounds(plan_data: dict) -> Set[str]:
    """Extract compound features (bigrams, motifs) from a composition plan.

    Extracts both simple sequential features and richer structural features
    based on actual call graph structure and binding flow:

    Simple (backward-compatible):
        - bigram: sequential subroutine ID pairs
        - trigram: sequential subroutine ID triples
        - bind: subroutine + binding key name
        - pos: position + subroutine ID

    Structural (call-graph-aware):
        - flow: parent-child data flow edges (output of call i feeds call i+1)
        - bind_flow: specific binding-level data flow (L01.x -> L03.y)
        - depth_trigram: depth-aware trigrams reflecting nesting structure
        - type_sig: subroutine + types of its bindings
        - arity_pair: consecutive call arity pairs
        - total_calls: total number of subroutine calls
    """
    compounds = set()
    calls = plan_data.get("plan", [])
    sub_ids = [c.get("sub_id", "") for c in calls]

    # --- Simple features (backward-compatible) ---

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

    # --- Structural features (call-graph-aware) ---

    # Build environment model: track which binding keys each call produces
    # and which keys each call consumes from the environment (not its own bindings).
    # A call's explicit bindings are "provided"; slots not in explicit bindings
    # are "inherited" from the environment (i.e., produced by earlier calls).
    call_explicit_keys: List[Set[str]] = []
    call_all_binding_keys: List[Set[str]] = []
    env_keys: Set[str] = set()  # keys available in environment after each call

    for i, call in enumerate(calls):
        bindings = call.get("bindings", {})
        explicit = set(bindings.keys())
        call_explicit_keys.append(explicit)
        call_all_binding_keys.append(explicit)
        # After execution, the call's binding keys + any computed keys enter env.
        # We approximate: all explicit binding keys plus "__last_output__" enter env.
        env_keys |= explicit

    # 1. Parent-child data flow bigrams
    #    When call i+1 has slots NOT in its explicit bindings (inherited from env),
    #    it depends on prior calls' outputs. We model this as a flow edge.
    inherited_keys_per_call: List[Set[str]] = []
    running_env: Set[str] = set()
    for i, call in enumerate(calls):
        bindings = call.get("bindings", {})
        explicit = set(bindings.keys())
        # Keys this call inherits from env (not explicitly provided)
        inherited = running_env - explicit
        inherited_keys_per_call.append(inherited)
        running_env |= explicit

    for i in range(len(calls) - 1):
        # If call i+1 inherits any keys (not all bindings are explicit),
        # there is a data flow from prior calls to call i+1.
        next_explicit = call_explicit_keys[i + 1]
        # Check if call i+1 could be consuming output from call i
        # (i.e., call i+1 has fewer explicit bindings than call i,
        #  or shares binding key names with call i)
        shared_keys = call_explicit_keys[i] & next_explicit
        has_flow = len(next_explicit) < len(call_explicit_keys[i]) or len(shared_keys) > 0
        # Always create flow edge for consecutive calls (conservative: any
        # sequential pair has potential data dependency via environment)
        compounds.add(f"flow:{sub_ids[i]}>{sub_ids[i+1]}")

    # 2. Binding flow patterns
    #    Track when a binding key from call i appears in call i+1 (or later),
    #    indicating data flows through that variable name.
    for i in range(len(calls)):
        for j in range(i + 1, len(calls)):
            keys_i = set(calls[i].get("bindings", {}).keys())
            keys_j = set(calls[j].get("bindings", {}).keys())
            shared = keys_i & keys_j
            for k in sorted(shared):
                compounds.add(
                    f"bind_flow:{sub_ids[i]}.{k}->{sub_ids[j]}.{k}"
                )

    # 3. Depth-aware trigrams
    #    Approximate nesting depth: call 0 is depth 0; if a call inherits
    #    bindings from env it is at depth >= 1 (depends on prior output).
    depths: List[int] = []
    running_env_depth: Set[str] = set()
    for i, call in enumerate(calls):
        bindings = call.get("bindings", {})
        explicit = set(bindings.keys())
        # Depth heuristic: how many prior env keys this call could inherit
        inherited_count = len(running_env_depth - explicit)
        depth = min(inherited_count, 3)  # cap at 3 for manageable feature space
        depths.append(depth)
        running_env_depth |= explicit

    if len(calls) >= 3:
        for i in range(len(calls) - 2):
            compounds.add(
                f"depth_trigram:d{depths[i]}:{sub_ids[i]}"
                f">d{depths[i+1]}:{sub_ids[i+1]}"
                f">d{depths[i+2]}:{sub_ids[i+2]}"
            )

    # 4. Type-signature compounds
    for call in calls:
        sub_id = call.get("sub_id", "")
        bindings = call.get("bindings", {})
        btypes = _infer_binding_types(bindings)
        if btypes:
            compounds.add(f"type_sig:{sub_id}({','.join(btypes)})")

    # 5. Arity-pair compounds
    for i in range(len(calls) - 1):
        arity_i = len(calls[i].get("bindings", {}))
        arity_j = len(calls[i + 1].get("bindings", {}))
        compounds.add(f"arity_pair:{arity_i}>{arity_j}")

    # 6. Call-count compound
    compounds.add(f"total_calls:{len(calls)}")

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
