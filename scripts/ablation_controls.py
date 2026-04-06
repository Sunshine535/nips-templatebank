#!/usr/bin/env python3
"""Ablation controls: compression-matched macros and typing variants.

Generates modified subroutine libraries + training data for:
  compression_matched  -- random macros with similar token savings (no semantics)
  untyped              -- all type annotations collapsed to FLOAT
  shuffled_types       -- type assignments randomly permuted across slots

Usage:
    python scripts/ablation_controls.py \
        --ablation compression_matched|untyped|shuffled_types \
        --library_path results/templates/subroutine_library.json \
        --programs_path results/templates/all_programs.json \
        --output_dir results/ablation/ABLATION_NAME
"""

import argparse
import copy
import json
import logging
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.template_dsl import (
    CompositionPlan,
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


# ---------------------------------------------------------------------------
# Compression-matched control: random macros with similar step counts
# ---------------------------------------------------------------------------

def _avg_plan_length(plans: list) -> float:
    """Average number of subroutine calls per plan."""
    lengths = [len(p.get("plan_data", {}).get("plan", [])) for p in plans]
    return sum(lengths) / max(len(lengths), 1)


def _avg_flat_length(programs: list) -> float:
    """Average number of steps in flat programs."""
    lengths = [len(p.get("program", {}).get("steps", [])) for p in programs]
    return sum(lengths) / max(len(lengths), 1)


def build_compression_matched(library: SubroutineLibrary, programs: list,
                               seed: int = 42) -> SubroutineLibrary:
    """Create random-macro library with similar step counts to real subroutines.

    For each real subroutine, we build a "macro" by sampling a random
    contiguous subsequence from a random program, matching the step count.
    The macro has no semantic coherence -- it isolates the compression effect.
    """
    rng = random.Random(seed)
    all_steps_pool = []
    for item in programs:
        prog = Program.from_dict(item["program"])
        all_steps_pool.append(prog.steps)

    if not all_steps_pool:
        logger.warning("No programs available; returning empty library")
        return SubroutineLibrary()

    new_lib = SubroutineLibrary()
    for sub_id, sub in library.subroutines.items():
        target_len = len(sub.program.steps)
        # Pick a random program with enough steps
        candidates = [s for s in all_steps_pool if len(s) >= target_len]
        if not candidates:
            candidates = all_steps_pool

        donor = rng.choice(candidates)
        start = rng.randint(0, max(0, len(donor) - target_len))
        chunk = copy.deepcopy(donor[start:start + target_len])

        # Build random macro with same slot count but arbitrary slots
        slot_names = set()
        for step in chunk:
            slot_names.update(step.inputs)
        slots = [Slot(name=n, dtype=DType.FLOAT, description="random") for n in sorted(slot_names)]

        macro_prog = Program(
            program_id=f"rand_{sub_id}",
            slots=slots,
            steps=chunk,
            source="random_macro",
        )
        macro_sub = Subroutine(
            sub_id=sub_id,  # keep same ID so plans still reference it
            program=macro_prog,
            support=sub.support,
            mdl_gain=0.0,
        )
        new_lib.subroutines[sub_id] = macro_sub
        new_lib._fp_index[macro_prog.fingerprint()] = sub_id

    logger.info("Compression-matched library: %d random macros (same IDs as real)",
                new_lib.size)
    return new_lib


# ---------------------------------------------------------------------------
# Typing ablations: untyped / shuffled_types
# ---------------------------------------------------------------------------

def build_untyped(library: SubroutineLibrary) -> SubroutineLibrary:
    """Strip all type annotations -- every slot becomes FLOAT."""
    new_lib = SubroutineLibrary()
    for sub_id, sub in library.subroutines.items():
        new_prog = copy.deepcopy(sub.program)
        for slot in new_prog.slots:
            slot.dtype = DType.FLOAT
        for step in new_prog.steps:
            step.target_dtype = DType.FLOAT
        new_sub = Subroutine(
            sub_id=sub_id,
            program=new_prog,
            support=sub.support,
            mdl_gain=sub.mdl_gain,
        )
        new_lib.subroutines[sub_id] = new_sub
        new_lib._fp_index[new_prog.fingerprint()] = sub_id

    logger.info("Untyped library: %d subroutines (all FLOAT)", new_lib.size)
    return new_lib


def build_shuffled_types(library: SubroutineLibrary, seed: int = 42) -> SubroutineLibrary:
    """Randomly permute type assignments across slots within each subroutine."""
    rng = random.Random(seed)
    new_lib = SubroutineLibrary()
    for sub_id, sub in library.subroutines.items():
        new_prog = copy.deepcopy(sub.program)
        if new_prog.slots:
            dtypes = [slot.dtype for slot in new_prog.slots]
            rng.shuffle(dtypes)
            for slot, dt in zip(new_prog.slots, dtypes):
                slot.dtype = dt
        new_sub = Subroutine(
            sub_id=sub_id,
            program=new_prog,
            support=sub.support,
            mdl_gain=sub.mdl_gain,
        )
        new_lib.subroutines[sub_id] = new_sub
        new_lib._fp_index[new_prog.fingerprint()] = sub_id

    logger.info("Shuffled-types library: %d subroutines", new_lib.size)
    return new_lib


# ---------------------------------------------------------------------------
# Rebuild training data from modified library
# ---------------------------------------------------------------------------

def rebuild_training_data(modified_lib: SubroutineLibrary, programs: list,
                          output_dir: str):
    """Regenerate compose + flat training data using the modified library.

    Plans keep the same subroutine IDs, so the modified library slots in
    transparently.  We just regenerate the instruction strings with the
    new signatures.
    """
    lib_sigs = "\n".join(modified_lib.signatures())

    compose_data = []
    flat_data = []

    for item in programs:
        problem = item["problem"]
        prog = Program.from_dict(item["program"])
        steps_sig = tuple(s.op.value for s in prog.steps)

        # Find matching subroutine (same logic as extract_templates.py)
        best_sub = None
        for sub in modified_lib.subroutines.values():
            sub_sig = tuple(s.op.value for s in sub.program.steps)
            if sub_sig == steps_sig:
                best_sub = sub
                break
        if best_sub is None:
            for sub in modified_lib.subroutines.values():
                sub_sig = tuple(s.op.value for s in sub.program.steps)
                if len(sub_sig) <= len(steps_sig):
                    best_sub = sub
                    break
        if best_sub is None and modified_lib.subroutines:
            best_sub = list(modified_lib.subroutines.values())[0]

        if best_sub is None:
            continue

        plan = CompositionPlan(calls=[{
            "sub_id": best_sub.sub_id,
            "bindings": item.get("bindings", {}),
        }])
        plan_json = json.dumps(plan.to_dict())

        compose_data.append({
            "instruction": f"Available subroutines:\n{lib_sigs}\n\nProblem: {problem}\n\nGenerate a composition plan (JSON):",
            "output": plan_json,
            "source": item.get("source", ""),
        })

        flat_data.append({
            "instruction": f"Problem: {problem}\n\nGenerate an executable program (JSON):",
            "output": json.dumps(prog.to_dict()),
            "source": item.get("source", ""),
        })

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "compose_train.json"), "w") as f:
        json.dump(compose_data, f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, "flat_train.json"), "w") as f:
        json.dump(flat_data, f, indent=2, ensure_ascii=False)

    logger.info("Training data saved to %s: compose=%d, flat=%d",
                output_dir, len(compose_data), len(flat_data))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Ablation controls: compression-matched macros and typing variants")
    parser.add_argument("--ablation", required=True,
                        choices=["compression_matched", "untyped", "shuffled_types"],
                        help="Which ablation to run")
    parser.add_argument("--library_path", type=str, required=True,
                        help="Path to real subroutine_library.json")
    parser.add_argument("--programs_path", type=str, required=True,
                        help="Path to all_programs.json")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for modified library + training data")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for stochastic ablations")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("  Ablation: %s", args.ablation)
    logger.info("=" * 60)

    # Load real library
    library = SubroutineLibrary.load(args.library_path)
    logger.info("Loaded library: %d subroutines from %s", library.size, args.library_path)

    # Load programs
    with open(args.programs_path) as f:
        programs = json.load(f)
    logger.info("Loaded %d programs from %s", len(programs), args.programs_path)

    # Compute and log compression stats
    avg_flat = _avg_flat_length(programs)
    logger.info("Average flat program length: %.1f steps", avg_flat)

    # Build modified library
    if args.ablation == "compression_matched":
        modified_lib = build_compression_matched(library, programs, seed=args.seed)
    elif args.ablation == "untyped":
        modified_lib = build_untyped(library)
    elif args.ablation == "shuffled_types":
        modified_lib = build_shuffled_types(library, seed=args.seed)
    else:
        raise ValueError(f"Unknown ablation: {args.ablation}")

    # Save modified library
    os.makedirs(args.output_dir, exist_ok=True)
    lib_path = os.path.join(args.output_dir, "subroutine_library.json")
    modified_lib.save(lib_path)
    logger.info("Modified library saved to %s", lib_path)

    # Rebuild training data
    rebuild_training_data(modified_lib, programs, args.output_dir)

    # Save ablation metadata
    meta = {
        "ablation": args.ablation,
        "seed": args.seed,
        "original_library_size": library.size,
        "modified_library_size": modified_lib.size,
        "num_programs": len(programs),
        "avg_flat_steps": avg_flat,
    }
    with open(os.path.join(args.output_dir, "ablation_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Ablation '%s' complete. Output in %s", args.ablation, args.output_dir)


if __name__ == "__main__":
    main()
