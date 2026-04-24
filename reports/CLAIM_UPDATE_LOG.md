# Claim Update Log

| Claim | Old Text | New Text | Evidence | Status |
|-------|----------|----------|----------|--------|
| Library gain | "Frozen library yields ≥15% gain" | REMOVED — unsupported hypothesis | No positive result for compose vs flat | Contradicted |
| Compose vs flat | "Compose improves over flat on MCD-hard" | REMOVED — contradicted | flat 5/100 > compose 0/100 (eval_v2) | Contradicted |
| SEVAL evolution | "RLVR-evolved library +10pts on MATH" | Demoted to future hypothesis | No SEVAL experiment completed | Unsupported |
| Test-time tools | "Recovers ≥20% failures" | Demoted to future hypothesis | No test-time eval completed | Unsupported |
| Cross-model transfer | "Transfer ≥8pts to 4B" | Demoted to future hypothesis | No transfer experiment | Unsupported |
| NEW: SFT teaches programs | — | "SFT on 697 verified programs teaches typed program generation: 0% → 29.5% on GSM8K test" | `results/seval/gsm8k_test/results.json` | Supported |
| NEW: Format learnable | — | "Typed DSL JSON-AST format is highly learnable (95% parse rate after SFT)" | Same | Supported |
| NEW: GIFT format learnable | — | "GIFT DataflowPlan format is learnable (99.5% parse rate)" | `results/gift/gsm8k_test/results.json` | Supported |
| NEW: Overfitting | — | "697 programs insufficient — 84% train vs 29.5% test indicates severe overfitting" | Train vs test comparison | Supported |
| NEW: GRPO saturated | — | "GRPO shows no improvement over SFT on 697 programs (reward saturated at 0.94)" | GRPO = SFT = 29.5% | Supported |
| NEW: GIFT coverage | — | "GIFT faithful coverage 20.2% with current library — below 30% threshold" | `results/gift/plan_audit.json` | Supported |
| NEW: Binding diagnosis | — | "98% empty bindings in old compose data confirms GPT-5.5 missing mechanism diagnosis" | Plan audit | Supported |
