# Core A/B/C Comparison — GSM8K Test (200 samples, seed 42)

## Results (UPDATED with step-primitive GIFT)

| Variant | Config | Accuracy | Parsed | Exec | Correct/Total |
|---------|--------|----------|--------|------|---------------|
| **A. old_fragment_only** | flat SFT on 697 verified programs | **30.0%** | 95.0% | 88.0% | 60/200 |
| **B1. gift_no_call_output** | GIFT schema, no call_output refs | 0.0% | 94.0% | 90.0% | 0/200 |
| **B2. gift_no_active_gate** | GIFT, no active-binding filter | 18.5% | 97.5% | 93.5% | 37/200 |
| **C. full_gift_step** | step-primitive GIFT, all gates ON | **19.0%** | 97.5% | 90.0% | 38/200 |

## Interpretation

### C vs A (19% vs 30%): GIFT does NOT beat existing best fragment
Per GPT-5.5 stop criteria, the new mechanism should beat the existing best
positive fragment. This is not achieved on seed 42 with the current 9B
student + 697 verified programs.

### C vs B1 (19% vs 0%): Call-output refs are CRITICAL (+19pt)
Stripping call_output refs from training data collapses accuracy to zero.
Confirms the dataflow mechanism is causally doing work — but the magnitude
of work it does is not enough to overtake flat SFT.

### C vs B2 (19.0% vs 18.5%): Active-binding gate has minimal effect at this scale
The active-binding filter on training data gave only +0.5pt on seed 42.
Either the filter was redundant (most plans already had active bindings) or
the gate is too weak at current scale.

### Why A still wins:
1. **Data quantity**: 697 flat programs vs 565 GIFT plans
2. **Generation simplicity**: complete program vs DAG with explicit refs
3. **Model capacity**: 9B may need more parameters to learn GIFT schema
4. **GIFT scale curve**: 141 plans → 7%, 565 plans → 19%, suggests ~1500
   faithful plans needed to reach 30%

## Training Stats

| Variant | Records | Steps | Train Loss | Token Acc | Runtime |
|---------|---------|-------|-----------|-----------|---------|
| A       | 697     | 351   | 0.115     | 96.9%     | 8.2 min |
| B1      | 565     | 285   | 0.106     | 98.9%     | 6.3 min |
| B2      | 565     | 285   | 0.108     | 98.9%     | 6.1 min |
| C       | 565     | 213   | 0.111     | 99.1%     | 4.5 min |

All trained with manifest (data hash, config hash, git commit, seed=42).

## GPT-5.5 Pro Gate Check

| Gate | Required | Actual (seed 42) | Status |
|------|----------|------------------|--------|
| GIFT coverage >30% | Yes | 81.1% | ✅ |
| true_dataflow > 0 | Yes | 582 plans | ✅ |
| active_binding_rate >70% | Yes | 97.3% | ✅ |
| Mechanism causally active | C > B1 | 19% > 0% | ✅ |
| **Full GIFT > Existing Best** | **C > A** | **19% < 30%** | **❌** |

## Multi-Seed Status

Only seed 42 evaluated. Seeds 123 and 456 not yet run (~5 GPU-hours each).
Given C < A by 11 points on seed 42 with similar parse/exec rates, the
gap is unlikely to flip with seed variation alone.

## Honest Decision

**STOP. DO NOT ENTER FULL BENCHMARK.**

The mechanism IS causally active (B1=0% confirms call_output refs matter,
+19pt over chance). But Full GIFT does not beat the flat SFT baseline at
this scale. Three options forward:

1. **Accept flat SFT as primary path**: 30% on GSM8K test is the strongest
   result. Re-frame GIFT as an analysis tool / ablation lens for understanding
   composition mechanism.
2. **Scale up training data**: extract MATH + more GSM8K to reach ~2000+
   faithful GIFT plans. Pre-extraction speedup needed.
3. **Larger student model**: 9B may be capacity-limited for GIFT schema.

This is the final A/B/C result. Reporting honestly per GPT-5.5 integrity rules.
