# Claude Execution Plan

## 1. GPT55_DIAGNOSIS.md Location
`./GPT55_DIAGNOSIS.md` (repository root, 1324 lines)

## 2. MAIN METHOD PATH
**GIFT — Grounded Interface-Flow Template Composition**

Replace implicit env/slot/type binding with explicit grounded interface-flow execution: every subroutine input must bind to a problem quantity or a previous call output via `BindingRef`. Plans are typed DAGs, not flat call lists with empty bindings.

## 3. Missing Mechanism
**Grounded Interface-Flow Binding and Faithfulness Verification**

Current `CompositionExecutor` resolves slots via implicit env/name/type heuristic. Plans allow empty bindings. Training labels map unrelated problems to mismatched templates. MCD compounds are from call adjacency, not real dataflow.

## 4. Current Evidence Supporting the Diagnosis

| Evidence | Supports? | Source |
|----------|-----------|--------|
| compose GSM8K 2/100, MCD 0/100 with 96% valid, 81% exec | YES — high syntax, zero semantics | `results/eval_v2/eval_results_seed42.json` |
| 410/418 compose_train have empty bindings | YES — no grounding | `results/templates_verified/compose_train.json` |
| SFT 84% train / 29.5% test (recent run) | YES — overfitting on ungrounded labels | `results/seval/sft_seed42/eval_results.json` |
| GRPO = SFT (29.5%) | YES — reward saturated on bad data | `results/seval/gsm8k_test/results.json` |
| base model 0% on program generation | YES — format is learnable only via SFT | `results/seval/gsm8k_test/results.json` |

## 5. Current Evidence Contradicting/Weakening the Diagnosis

| Evidence | Contradicts? | Interpretation |
|----------|-------------|----------------|
| SFT achieves 29.5% on GSM8K test with flat programs (no composition) | PARTIALLY — flat programs work without GIFT | Flat SFT is a strong baseline; GIFT must beat it |
| 697 verified programs all correct | NEUTRAL — supports verified execution, doesn't contradict GIFT need | Source data is good; label construction is the problem |

## 6. Files to Inspect
- `src/template_dsl.py` — CompositionExecutor, BindingRef candidates
- `src/mcd_split.py` — compound extraction logic
- `scripts/extract_templates.py` — plan label construction
- `scripts/eval_template_reasoning.py` — fallback/metrics
- `scripts/train_template_compiler.py` — training reproducibility
- `tests/test_templatebank.py` — existing tests
- `results/templates_verified/compose_train.json` — current labels
- `results/seval/gsm8k_test/results.json` — recent results
- `configs/template_config.yaml` — current config

## 7. Files to Edit
- NEW: `src/dataflow_plan.py` — BindingRef, DataflowPlan, DataflowExecutor
- NEW: `tests/test_dataflow_plan.py` — explicit dataflow tests
- NEW: `scripts/audit_dataflow_plans.py` — plan faithfulness auditor
- NEW: `scripts/build_gift_data.py` — faithful GIFT data builder
- NEW: `configs/gift_minimal.yaml` — GIFT config
- EDIT: `src/mcd_split.py` — true dataflow compounds
- EDIT: `scripts/eval_template_reasoning.py` — method-only metrics
- EDIT: `scripts/train_template_compiler.py` — reproducibility

## 8. Files to Archive
- `results/extract_27b_500/` → archive as historical
- `results/mcd_split_27b.json` → archive (synthetic, bad atom_tvd)
- `results/templatebank_pilot_*.json` → archive as mixed/weak evidence
- `PIPELINE_REPORT.md` → archive as historical 4% synthetic
- `src/template_algebra.py` → archive as aspirational/unused

## 9. Files NOT to Touch
- `results/templates_verified/all_programs.json` — verified source data, keep
- `results/eval_v2/eval_results_seed42.json` — historical negative, keep as diagnostic
- `results/seval/gsm8k_test/results.json` — recent valid results, keep
- `src/rlvr_evolution.py` — freeze, keep as future ablation
- `src/test_time_tools.py` — freeze, keep as future ablation

## 10. Tests Before/After
**Before**: `pytest tests/test_templatebank.py` — verify existing tests pass
**After**: `pytest tests/test_dataflow_plan.py tests/test_templatebank.py`
**New test**: ADD→MUL explicit dataflow (expected (2+3)*4=20)

## 11. Rollback Conditions
- If DataflowExecutor breaks primitive Program execution → isolate in separate file
- If faithful GIFT coverage <30% → stop, report library mining unusable
- If one-batch overfit fails → stop, report schema/training broken
- If Full GIFT does not beat old fragment across 3 seeds → stop, report diagnosis may be wrong
