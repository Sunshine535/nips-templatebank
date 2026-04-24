# Current Result Audit

## Result Table

| Result | File | Dataset | Config | Seed | Metric | Value | Compared Against | Supports Diagnosis? | Notes |
|--------|------|---------|--------|------|--------|-------|------------------|---------------------|-------|
| Old compose eval | `results/eval_v2/eval_results_seed42.json` | GSM8K | template_config | 42 | accuracy | 2/100 (2%) | flat 0%, cot 0% | YES — high exec, zero semantics | Historical negative |
| Old MCD compose | same | MCD_test | same | 42 | accuracy | 0/100 (0%) | flat 5/100 | YES — composition fails | flat > compose |
| SFT train eval | `results/seval/sft_seed42/eval_results.json` | GSM8K (train) | train_seval | 42 | accuracy | 42/50 (84%) | — | NEUTRAL — overfitting | Train set, not test |
| GRPO train eval | `results/seval/grpo_seed42/eval_results.json` | GSM8K (train) | train_seval | 42 | accuracy | 42/50 (84%) | SFT | YES — no GRPO improvement | Reward saturated |
| Base GSM8K test | `results/seval/gsm8k_test/results.json` | GSM8K test | — | 42 | accuracy | 0/200 (0%) | SFT, GRPO | YES — base cannot generate programs | Expected |
| SFT GSM8K test | same | GSM8K test | train_seval | 42 | accuracy | 59/200 (29.5%) | base, GRPO | YES — SFT teaches format, overfits | Recent valid result |
| GRPO GSM8K test | same | GSM8K test | train_seval | 42 | accuracy | 59/200 (29.5%) | SFT | YES — GRPO = SFT exactly | Need more data |
| SFT parse rate | same | GSM8K test | — | 42 | parse_rate | 95% | base 0% | NEUTRAL | Format learnable |
| SFT exec rate | same | GSM8K test | — | 42 | exec_rate | 89% | base 0% | YES — exec ≠ correct answer | Confirms PHE-19 |
| Verified programs | `results/templates_verified/all_programs_stats.json` | GSM8K | — | — | count | 697/697 | — | YES — source data valid | Only GSM8K |
| compose_train quality | `results/templates_verified/compose_train.json` | — | — | — | empty_binding_rate | ~98% | — | YES — broken labels | Core problem |
| Library | `results/templates_verified/subroutine_library.json` | — | — | — | size | 16 | — | NEUTRAL | Weak interfaces |

## Variant Existence Check

| Variant | Exists? | Location | Notes |
|---------|---------|----------|-------|
| A. Existing Best Positive Fragment Only | PARTIAL | SFT flat 29.5% (`results/seval/gsm8k_test/`) | Flat SFT, not old compose |
| B. New MAIN METHOD Without New Mechanism | NO | — | Not implemented yet |
| C. Full New MAIN METHOD (GIFT) | NO | — | Not implemented yet |

## Result-Based Execution Decision

**PROCEED**

Reason: Diagnosis is fully supported by current evidence. No contradictions. Missing variants B and C must be implemented. Bug fixes (empty bindings, eval reliability) already partially done but GIFT core not yet implemented. The SFT flat baseline at 29.5% sets the bar GIFT must beat.
