# Patch Summary

## Files Added
- `src/dataflow_plan.py` — GIFT core: BindingRef, DataflowPlan, PlanCall, DataflowExecutor
- `tests/test_dataflow_plan.py` — 10 tests for explicit dataflow semantics
- `scripts/build_gift_data.py` — Faithful GIFT data builder with permutation search
- `scripts/extract_vllm.py` — Batched HF extraction (vLLM fallback)
- `scripts/gpu_keepalive.py` — Python GPU keep-alive for pod
- `scripts/pod_bootstrap.sh` — Pod deployment automation
- `scripts/nccl_test.sh` — NCCL keep-alive supervisor
- `configs/gift_minimal.yaml` — GIFT experiment config
- `archive/README_unreliable_results.md` — Unreliable results manifest
- `reports/` — 13 execution reports (this file + 12 others)
- `idea-stage/IDEA_REPORT.md` — 5 improvement directions
- `review-stage/` — Auto-review loop state and logs
- `results/seval/` — SFT/GRPO training and eval results
- `results/gift/` — GIFT data and eval results

## Files Modified
- `scripts/extract_templates.py` — Fixed binding permutation search, checkpointing, flush
- `scripts/train_seval.py` — Complete rewrite with correct schema and reward
- `src/template_dsl.py` — Fixed executor ambiguous binding fallback
- `configs/template_config.yaml` — K=1, max_new_tokens=512, library_size=32

## Files Archived (manifest only, not moved)
- `results/mcd_split_27b.json` — synthetic, bad atom_tvd
- `results/templatebank_pilot_*.json` — weak signal
- `src/template_algebra.py` — dead code
- `PIPELINE_REPORT.md` — historical 4% synthetic

## Bugs Fixed
1. Empty bindings in compose_train (98%) → permutation search
2. CompositionExecutor silent wrong binding → reject ambiguous
3. Library mining collapse (697→3) → slot count + subsequence mining
4. train_seval.py schema mismatch → complete rewrite
5. Container OOM (16GB) → 128GB
6. Python log buffering → -u flag + explicit flush

## Configs Added
- `configs/gift_minimal.yaml`

## Tests Added
- 10 new GIFT dataflow tests (all pass)
- Total: 41/41 pass on pod

## Key Commands Run
- `pytest tests/ -v` → 41/41 pass
- `python3 scripts/build_gift_data.py` → 141/697 faithful (20.2%)
- `torchrun --nproc_per_node=4 scripts/train_seval.py --mode sft` → 29.5% GSM8K test
- `torchrun --nproc_per_node=4 scripts/train_seval.py --mode grpo` → 29.5% (= SFT)
- `python3 eval_gift.py` → 7% GSM8K test
- `python3 eval_gsm8k_test.py` → base 0%, SFT 29.5%, GRPO 29.5%

## Results Observed
| Model | GSM8K Test (200) | Parse | Exec |
|-------|-----------------|-------|------|
| Base Qwen3.5-9B | 0.0% | 0% | 0% |
| Flat SFT | 29.5% | 95% | 89% |
| Flat GRPO | 29.5% | 95% | 89% |
| GIFT SFT | 7.0% | 99.5% | 77% |

## Failed Checks
- GIFT coverage < 30% threshold (20.2%)
- Zero two-call dataflow plans
- GIFT (7%) < Flat SFT (29.5%) — C < A
- GRPO no improvement over SFT

## Unresolved Risks
See reports/REMAINING_RISKS.md
