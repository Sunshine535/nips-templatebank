# Project Direction Lock

PROJECT_DIRECTION_LOCK:
Build a rigorous, reproducible, positive NeurIPS main-track method for execution-verified, compositional mathematical reasoning, where the core contribution is a mechanism-level improvement in reusable / structured reasoning.

Allowed changes:
Mechanism, plan representation, loss/objective, training curriculum, verifier, intermediate supervision, inference procedure, ablation design, logging, reproducibility, baseline fairness.

Forbidden pivots:
Negative-result paper, workshop-only diagnosis, GSM8K-only preprocessing trick, benchmark-specific shortcut, oracle-at-test method, baseline weakening, metric/split/preprocessing change to create gains, hiding negative results, or claiming SOTA without official baselines.

Scope Compliance Status:
PASS if successor method uses model-generated intermediate value hints and executable consistency checks without test-time gold leakage.
FAIL if any implementation uses gold test intermediates, weakens flat baselines, changes metric/split/preprocessing, or reframes the work as only an oracle-analysis paper.

## Current Successor Path
V-GIFT: Value-Grounded Interface-Flow Template Composition

## Evidence
- Pure symbolic GIFT C=19% < flat A=30% (failed)
- Value-supervised E=41.5% > flat D=29.5% (weak positive signal)
- Dataflow mechanism is causally active: B1=0% vs C=19%
- Edge activity: 94.3% call_output edges causally active

## Novelty Gate Status: NOVELTY_RISK (conditional pass)

Closest works: HintMR, Chain of Code, PAL/PoT/ToRA, Faithful CoT, MARIO.
Required differentiation: typed reusable subroutine DAG + model-generated
value hints + executable consistency verifier + active edge audit.
Code provenance: NO_PROVENANCE_RISK_DETECTED — all main method code is original.
