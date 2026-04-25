# Minimal Experiment Results (Updated R4)

## Complete 6-Variant Ablation — GSM8K Test 200, seed 42, commit 75329c4

| # | Variant | Format | Data | Accuracy | Parse | Exec | Correct | Interpretation |
|---|---------|--------|------|----------|-------|------|---------|----------------|
| A | old_fragment_only | flat JSON | 697 | **30.0%** | 95% | 88% | 60/200 | Flat baseline (697) |
| D | flat_matched_565 | flat JSON | 565 | **29.5%** | 97% | 91% | 59/200 | Matched-data flat |
| **E** | **value_supervised_plan** | **GIFT+constants** | **565** | **41.5%** | **94%** | **88%** | **83/200** | **Best: plan structure + value supervision** |
| C | full_gift_step | GIFT+refs | 565 | 19.0% | 98% | 90% | 38/200 | Symbolic refs too hard |
| B2 | gift_no_active_gate | GIFT (no gate) | 565 | 18.5% | 98% | 94% | 37/200 | Gate minimal effect |
| B1 | gift_no_call_output | GIFT (no refs) | 565 | 0.0% | 94% | 90% | 0/200 | Confirms refs are causal |

## Mechanism Audit (v2, step-primitive data)

| Metric | Value |
|--------|-------|
| GIFT plan coverage | 81.1% (565/697) |
| true_dataflow_rate (correct plans) | 100% |
| Quantity binding active rate | 97.3% |
| Call-output edge active rate | 94.3% |

## Limitations
- Seed 42 only (no 123/456)
- First 200 GSM8K test samples (not full 1319)
- No official baselines (PAL/BoT/Faithful CoT)
- E is training-time value supervision, not test-time oracle — but leakage audit pending
- All results are weak signal, not paper claims
