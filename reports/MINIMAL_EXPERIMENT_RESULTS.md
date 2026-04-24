# Minimal Experiment Results

| Experiment | Config | Dataset | Seed | Metric | Result | Expected | Pass/Fail | Interpretation |
|------------|--------|---------|------|--------|--------|----------|-----------|----------------|
| Smoke test (pytest) | — | — | — | 41/41 pass | PASS | all pass | PASS | Codebase stable |
| Plan audit | — | templates_verified | — | empty_binding_rate | 98.1% | ~98% | PASS | Confirms GPT-5.5 diagnosis |
| Plan audit: single_call | — | templates_verified | — | single_call_rate | 100% | ~100% | PASS | No composition in old data |
| Plan audit: subs used | — | templates_verified | — | unique_subs | 3/16 | few | PASS | Library collapse confirmed |
| SFT training | train_seval sft | 697 GSM8K programs | 42 | train_loss | 0.060 | <0.1 | PASS | Model learns format |
| SFT train eval | train_seval sft | GSM8K train subset | 42 | accuracy | 84.0% | >50% | PASS | Overfitting expected |
| GRPO training | train_seval grpo | 697 GSM8K programs | 42 | mean_reward | 0.80-0.94 | >0.5 | PASS | Reward saturated |
| GRPO train eval | train_seval grpo | GSM8K train subset | 42 | accuracy | 84.0% | ≥SFT | PASS | No improvement (saturated) |
| Base GSM8K test | — | GSM8K test 200 | 42 | accuracy | 0.0% | ~0% | PASS | Base cannot generate JSON |
| **SFT GSM8K test** | train_seval sft | GSM8K test 200 | 42 | accuracy | **29.5%** | >0% | PASS | First positive result |
| SFT parse rate | train_seval sft | GSM8K test 200 | 42 | parse_rate | 95.0% | >50% | PASS | Format learnable |
| SFT exec rate | train_seval sft | GSM8K test 200 | 42 | exec_rate | 89.0% | >50% | PASS | Programs structurally valid |
| GRPO GSM8K test | train_seval grpo | GSM8K test 200 | 42 | accuracy | 29.5% | ≥SFT | PASS | = SFT, no improvement |
| GIFT data build | — | 697 programs | — | faithful_coverage | 20.2% | >30% | **WARN** | Below threshold |
| GIFT data: faithful plans | — | 697 programs | — | gift_plans | 141 | >200 | **WARN** | Library too coarse |
| GIFT data: two-call flow | — | 697 programs | — | true_dataflow | 0 | >0 | **FAIL** | No multi-call plans |
| GIFT SFT training | gift_minimal | 141 GIFT plans | 42 | train_loss | 0.194 | <0.3 | PASS | Learns GIFT format |
| **GIFT GSM8K test** | gift_minimal | GSM8K test 200 | 42 | accuracy | **7.0%** | >29.5% | **FAIL** | C < A |
| GIFT parse rate | gift_minimal | GSM8K test 200 | 42 | parse_rate | 99.5% | >90% | PASS | Format highly learnable |
| GIFT exec rate | gift_minimal | GSM8K test 200 | 42 | exec_rate | 77.0% | >80% | WARN | Binding issues |
| A vs C comparison | — | GSM8K test 200 | 42 | A=29.5% C=7.0% | C < A | C > A | **FAIL** | GIFT not competitive yet |
