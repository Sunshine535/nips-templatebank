# Core Comparison: A vs C

## GSM8K Test Set (200 samples, seed 42)

| Variant | Config | Dataset | Accuracy | Parse | Exec | Training Data |
|---------|--------|---------|----------|-------|------|---------------|
| A. Flat SFT (existing best) | train_seval sft | GSM8K test | **29.5%** (59/200) | 95% | 89% | 697 flat programs |
| C. Full GIFT | gift_minimal sft | GSM8K test | **7.0%** (14/200) | 99.5% | 77% | 141 GIFT plans |

## Interpretation

C < A: GIFT does NOT beat the existing best positive fragment.

Root cause: insufficient GIFT training data (141 vs 697, 5x gap).
GIFT parse rate (99.5%) proves the format is highly learnable.
GIFT exec rate (77%) shows explicit bindings work but library coverage is too narrow.

## Per GPT-5.5 Diagnosis Stop/Continue Criteria

- "Full GIFT beats both old fragment and no-mechanism ablation" → **NOT MET**
- "faithful GIFT coverage >30%" → **NOT MET** (20.2%)
- This triggers: "If faithful coverage <30%, stop and report library mining unusable"

## Decision

**DEBUG MORE** — GIFT format is learnable (99.5% parse) but library mining is too coarse.
Need: (1) finer-grained subroutine mining, or (2) more verified programs, or (3) step-level primitives instead of whole-program subroutines.

The flat SFT baseline at 29.5% remains the strongest result.
