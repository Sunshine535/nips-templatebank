---
type: claim
node_id: claim:C3
title: "Test-time tool building recovers ≥20% failures"
status: proposed
testable: true
created_at: 2026-04-09T00:00:00Z
---

# C3: Per-problem tool building recovers ≥20% of failed compositions

**Formal statement**: On MATH MCD-hard problems where all top-K library compositions fail, per-problem test-time tool building (max 3 new tools, max 10 verification attempts) recovers ≥20% of failures, under matched compute budget vs search-without-building baseline.

**Evidence needed**: Phase 2 evaluation with equalized compute accounting.

**If fails**: Keep as analysis section showing tool building potential, remove from headline claims.
