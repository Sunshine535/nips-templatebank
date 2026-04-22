# SEVAL Auto Review Loop

## Status: Round 1 — Fresh Start
**Date**: 2026-04-23
**Difficulty**: nightmare
**Prior state**: Stale (2026-04-04, discarded)

## Pre-Review Diagnosis

### Experimental Results (as of 2026-04-23)
All results are **catastrophically negative** (0-7% accuracy on GSM8K, SOTA >95%):

| Method | GSM8K Acc | MCD-test Acc |
|--------|-----------|--------------|
| compose (SEVAL core) | 2% | 0% |
| flat_inline | 0% | 5% |
| direct_cot | 0% | 1% |
| cot_budget@3 | 3% | 7% |

### Root Cause Analysis

**Three critical bugs identified and fixed:**

1. **Bug: Empty bindings in training data (98% of examples)**
   - `extract_templates.py:364-370`: Subroutine slot names don't match program variable names
   - Fix: Brute-force permutation search to find correct slot-to-value mapping
   - Verified: permutation search correctly maps values (e.g., finds 7*5=35 from [2,7,5])

2. **Bug: Catastrophic binding fallback in CompositionExecutor**
   - `template_dsl.py:408-412`: All slots bound to LAST candidate value when bindings empty
   - Fix: Reject ambiguous multi-candidate bindings instead of silent wrong assignment
   - Verified: executor now fails clearly on ambiguous bindings

3. **Bug: Library mining over-collapses diversity**
   - 697 diverse programs collapsed to 3 usable templates
   - Zero multi-subroutine plans (no actual composition)
   - Fix: Include slot count in signature + subsequence mining for multi-call plans
   - Fix: Increased target_size from 16 to 32

### Pipeline Status
- **Phase 0 extraction**: Running on pod (n88, 4x H200, 100% GPU util)
- **Code fixes**: Pushed to git and pulled on pod
- **Fixes will take effect**: When extraction completes and enters library mining phase

### Path to Positive Results

The teacher extraction WORKS (697/697 correct programs). The approach is viable.
With bug fixes, the pipeline should produce:
1. Meaningful bindings (>50% permutation-matched)
2. Diverse library (32 subroutines with slot-count differentiation)
3. Multi-call composition plans (from subsequence mining)
4. Higher plan faithfulness (from correct bindings)

Next: wait for Phase 0 to complete, verify fixed pipeline, then proceed to Phase 1 (GRPO training).
