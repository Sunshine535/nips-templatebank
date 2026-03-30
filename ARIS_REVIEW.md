# ARIS Research Review Report — Template Algebra

**Date**: 2026-03-30
**Reviewer**: Codex GPT-5.4 (xhigh reasoning)
**Current Score**: 3/10 (Clear Reject)
**Target Score**: ≥6/10 (Borderline Accept)

## Critical Issues Identified

### 1. Novelty Overstated
- ReasonFlux, SELF-DISCOVER, Buffer of Thoughts, Retrieval-of-Thought, Learning Composable CoT all address similar problems
- Only defensible novelty: **learned, typed, executable template composition with rigorous unseen-composition benchmark**

### 2. Implementation-Proposal Mismatch (SEVERE)
- Proposal says 32B teacher → code uses 9B for everything
- Proposal says JSON-AST composition plans → code uses keyword matching
- No schema-constrained decoding implemented
- No proper compositional split

### 3. Missing Critical Baseline
- **Flat-program baseline**: Same DSL, same executor, same constrained decoding, NO composition
- Without this, cannot prove composition helps beyond having a good DSL

## Kill List

### CUT
- Any claim about "general reasoning", "human-like abstraction", or SOTA
- Any primitive that is effectively a whole-task solver
- Descriptive library names (use opaque IDs like L01-L16)
- Any baseline not sharing the same primitive DSL

### RENAME
- `compiler` → `planner`
- `library` → `subroutine library`
- `compositional generalization` → `program-compound generalization under MCD split`

### KEEP
- Same primitive DSL for composed and flat
- Deterministic executor
- Fallback pipeline
- CFQ-style MCD split
- Oracle ceilings and error decomposition

## Experiment Plan (~1970 GPU-hours)

| Priority | Run | Purpose | GPU-h |
|----------|-----|---------|-------|
| 1 | canonicalize_all | 32B teacher generates flat executable templates | 220 |
| 2 | build_mcd | Build MCD splits from canonical programs | 10 |
| 3 | pilot_gsm8k_mcd | compose_L16 vs flat_inline, seed 0 | 70 |
| 4 | main_mcd_3seed | Main experiments, 3 seeds | 600 |
| 5 | cot_budget + retrieval_compose | Inference baselines | 260 |
| 6 | library_ablation | L={4,8,16,32} | 220 |
| 7 | iid_oracles_latency | Official IID tests + oracles | 240 |
| 8 | buffer | Reruns, failed jobs | 360 |

## Paper Structure

1. **Intro**: Single claim — subroutine composition helps beyond strong primitive DSL
2. **Setup**: Datasets, canonical DSL, executor, MCD split, metrics
3. **Method**: Teacher canonicalization, library mining by MDL, planner training, fallback
4. **Baselines**: Flat-inline, compute-matched CoT, retrieval-compose, oracle ceilings
5. **Main Results**: Table with accuracy, valid-plan, exec success, fallback, tokens, latency
6. **Analysis**: Library size sweep, oracle ceilings, error waterfall
7. **Discussion/Limits**: Scope, DSL coverage, teacher dependency

## Stop/Go Rules
- Stop if flat-DSL ceiling < 98% GSM8K or < 80% MATH
- Stop if MCD split atom TVD > 0.02
- Stop composition story if compose_L16 - flat_inline < 2pts with CI crossing 0

## Key Competitors
- ReasonFlux (Feb 2025): ~500 templates + hierarchical RL, 91.2% MATH
- Learning Composable CoT (2025): Composable CoT format for compositional generalization
- Retrieval-of-Thought: Composable thought steps, 40% token reduction
- Buffer of Thoughts: Template distillation + instantiation
- SELF-DISCOVER: Composes atomic reasoning modules
