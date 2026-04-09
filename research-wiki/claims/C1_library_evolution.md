---
type: claim
node_id: claim:C1
title: "Library Evolution > Frozen Library"
status: proposed
testable: true
created_at: 2026-04-09T00:00:00Z
---

# C1: RLVR-evolved library outperforms frozen library by ≥10 pts on MATH MCD-hard

**Formal statement**: On MATH MCD-hard split, Qwen3.5-9B compose-planner using RLVR-evolved library L_final achieves ≥10 absolute accuracy points higher than same model using frozen mined library L₀, with 95% bootstrap CI excluding 0, averaged across 3 seeds.

**Evidence needed**: Full Phase 0 + Phase 1 results, 3 seeds, MCD-hard evaluation.

**If fails**: Reframe to "library evolution improves efficiency" (fewer GRPO steps needed) rather than absolute accuracy.
