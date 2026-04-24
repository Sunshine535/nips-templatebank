# GPT-5.5 Pro Report Extraction

## Diagnosis File Used
`./GPT55_DIAGNOSIS.md` (root, 1324 lines, sections 0-20)

## Recommended MAIN METHOD PATH
**GIFT — Grounded Interface-Flow Template Composition**

Train and evaluate a planner that emits executable subroutine-call DAGs with explicit typed bindings from problem quantities and previous call outputs, and accept only plans that pass dataflow faithfulness checks.

## Missing Mechanism
**Grounded Interface-Flow Binding and Faithfulness Verification**

The repository has verified single-problem programs and recurring arithmetic skeletons, but lacks explicit mechanism that binds problem quantities to subroutine interfaces and connects subroutine outputs to later subroutine inputs. "Composition" degenerates into ungrounded template selection or concatenation.

## Evidence From Positive Results
- 697 verified programs all execute correctly and match answers (PHE-05)
- SFT teaches program generation: 0% → 29.5% on GSM8K test (recent)
- 95% parse rate shows typed DSL format is learnable

## Evidence From Negative Results
- compose GSM8K 2/100, MCD 0/100 with 96% valid, 81% exec (PHE-01/02)
- flat > compose on MCD: 5/100 vs 0/100 (PHE-03)
- 410/418 training labels have empty bindings (PHE-06)
- Problem-template semantic mismatch in labels (PHE-07)
- GRPO = SFT (29.5%), reward already saturated (recent)

## Evidence From Unstable Results
- Pilot dynamic memory: reuse up but accuracy flat (PHE-09/10)
- Only seed42 visible in old results (PHE-18)

## Evidence From Failed Ablations
- SEVAL review: pattern abstraction vacuous/bigram-like (PHE-16)
- MCTS gold reward cheating flagged in self-review
- Synthetic contamination in 27B split (PHE-11)

## Why Existing Best Positive Fragment Is Insufficient
The best fragments (pilot reuse, synthetic 4%, verified programs) do not add explicit grounded interface-flow semantics. SFT 29.5% on flat programs is a strong baseline that GIFT must beat.

## Files to Inspect
- `src/template_dsl.py` — CompositionExecutor binding logic
- `src/mcd_split.py` — compound extraction
- `scripts/extract_templates.py` — label construction
- `scripts/eval_template_reasoning.py` — metrics/fallback
- `results/templates_verified/compose_train.json` — broken labels

## Files to Edit
- NEW: `src/dataflow_plan.py`
- NEW: `tests/test_dataflow_plan.py`
- NEW: `scripts/audit_dataflow_plans.py`
- NEW: `scripts/build_gift_data.py`
- NEW: `configs/gift_minimal.yaml`
- EDIT: `src/mcd_split.py`
- EDIT: `scripts/eval_template_reasoning.py`

## Files to Archive
- `results/extract_27b_500/`, `results/mcd_split_27b.json` — synthetic
- `results/templatebank_pilot_*.json` — weak signal
- `src/template_algebra.py` — dead code

## Files to Keep
- `results/templates_verified/all_programs.json` — verified source
- `results/eval_v2/eval_results_seed42.json` — historical negative
- `results/seval/gsm8k_test/results.json` — recent valid result

## Files to Keep Only as Baseline
- `scripts/train_seval.py` — flat SFT/GRPO baseline (29.5%)
- Direct CoT, cot_budget baselines in eval

## Files to Keep Only as Ablation
- `src/rlvr_evolution.py` — resume after GIFT core
- `src/test_time_tools.py` — convert to dataflow gap repair later

## Suspected Bugs
1. P0: CompositionExecutor implicit binding (silent wrong answers)
2. P0: compose_train empty bindings (polluted supervision)
3. P0: flat_train mismatched templates (contaminated baseline)
4. P0: MCD compounds from adjacency, not real flow
5. P1: Eval fallback masking method accuracy
6. P1: Missing adapter loading validation

## Required Logging
- slot_coverage, typecheck_ok, exec_ok, answer_correct per plan
- active_binding_rate (perturb input, verify output changes)
- true_multicall_flow_rate
- empty_binding_rate
- checkpoint hash, data hash, seed, command

## Required Minimal Experiments
1. Smoke test (pytest)
2. Data sanity (empty binding audit on old labels)
3. One-batch overfit (GIFT schema)
4. A/B/C comparison (old fragment / GIFT no mechanism / full GIFT)
5. 3-seed stability

## Required Core Comparison
A. Existing Best Positive Fragment Only (old compose or flat SFT 29.5%)
B. GIFT Without Explicit Dataflow (plan calls without BindingRef)
C. Full GIFT (explicit BindingRef + faithfulness)

## Required Baselines
- Direct CoT, cot_budget, faithful flat inline, PAL (if feasible)

## Stop / Continue / Pivot Criteria
- **Continue**: faithful coverage >30%, one-batch overfit passes, GIFT > old fragment across 3 seeds
- **Stop**: cannot overfit, plans rarely execute, C does not beat A/B
- **Pivot**: faithful coverage too low, PAL/BoT dominates at matched compute
