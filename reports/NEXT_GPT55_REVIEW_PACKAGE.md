# Next GPT-5.5 Pro Review Package (R4 Update)

## Summary of Changes Since Last Review

### New Variants Trained and Evaluated (seed 42, GSM8K test 200)
- **D (flat_matched_565)**: 29.5% — matched-data flat baseline confirms data size not the issue
- **E (value_supervised_plan)**: 41.5% — GIFT plan format with oracle intermediate values as constants
- A/B1/B2/C retrained under frozen commit 75329c4

### Bug Fixes
- Audit counting bug fixed: `true_dataflow_rate` now ≤1.0 (was 1.03)
- Edge-level call_output activity audit added: 94.3% edges causally active
- Duplicate YAML key fixed: `library` → `library_path` + `library_config`
- Config integrity tests added (9 tests, duplicate key detection)

### Key Finding
**E = 41.5% reveals the successor path**: GIFT DataflowPlan format + explicit intermediate values beats flat programs by +12pt. The plan STRUCTURE helps, but the model cannot infer call_output values at this scale (C=19%).

## Result Table

| Variant | Accuracy | Notes |
|---------|----------|-------|
| A: old_fragment_only (697) | 30.0% | Flat baseline |
| D: flat_matched_565 (565) | 29.5% | Data-matched flat |
| **E: value_supervised_plan** | **41.5%** | **Best — GIFT + value constants** |
| C: full_gift_step | 19.0% | Symbolic refs only |
| B2: gift_no_active_gate | 18.5% | ≈ C |
| B1: gift_no_call_output | 0.0% | Mechanism causal |

## What Supports the Diagnosis
1. Dataflow mechanism is causally active (B1=0% vs C=19%, edge audit 94.3%)
2. Plan structure helps when values are grounded (E=41.5% > D=29.5%)
3. Pure symbolic inference is the bottleneck, not the architecture

## What Contradicts or Weakens
1. C < A: symbolic GIFT still loses to flat
2. B2 ≈ C: active-binding gate doesn't help
3. E is one seed / 200 samples / no leakage audit yet

## Unresolved Questions
1. Does E have test-time gold leakage? (likely no, audit pending)
2. Can V-GIFT (model-generated value hints) match E without oracle?
3. Is the +12pt signal stable across seeds?
4. Can it generalize beyond GSM8K?

## Decision
Implement V-GIFT successor path: value-grounded dataflow plans with model-generated intermediate value annotations and consistency checks.

## What GPT-5.5 Should Review Next
1. V-GIFT schema implementation (ValueAnnotatedDataflowPlan)
2. Leakage audit result for E
3. V-GIFT seed 42 gate results (F vs A vs E vs no-mechanism)
4. Multi-seed stability if gate passes
