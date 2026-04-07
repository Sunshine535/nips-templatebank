# Experiment Plan

## Scope
Repo: `/workspace/nips-templatebank`
Teacher: Qwen3.5-32B
Student A: Qwen3.5-9B
Student B: Qwen3.5-3B, else Llama-3-8B-Instruct
Primary benchmark: GSM8K
Stress test: MATH
Hardware: 4x H800 80GB for 5 weeks
Budget ceiling: 3360 GPUh
Planned budget: 1950 GPUh
Contingency: 330 GPUh
Reserved slack: 1080 GPUh

## Experiment Blocks

### Block 1: Smoke path validation
- Depends on: none
- GPUh: 2
- Command sketch:
```bash
cd /workspace/nips-templatebank
source .venv/bin/activate
SMOKE=1 bash run.sh --smoke
```
- Success: all scripts finish; outputs appear under `results`; no catastrophic executor failure
- If fail: fix repo/config issues before any GPU-heavy run

### Block 2: GSM8K teacher extraction with step verification
- Depends on: 1
- GPUh: 180
- Required code changes: step-level trace logging and rejection reasons in `scripts/extract_templates.py`; CLI overrides for dataset, output dir, samples/example
- Command sketch:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 scripts/extract_templates.py \
  --config configs/template_config.yaml \
  --teacher_model Qwen/Qwen3.5-32B \
  --dataset gsm8k \
  --samples_per_example 4 \
  --output_dir results/gsm8k_teacher_verified/
```
- Success: >=1500 verified programs; parse-valid >=55%; executable among parse-valid >=60%; final acceptance >=20%; median verified length <=10 steps
- If fail: raise `samples_per_example` to 8; if still <1500, narrow paper scope before training students

### Block 3: MATH teacher extraction stress pass
- Depends on: 1
- GPUh: 260
- Command sketch: same script, MATH subsets balanced, `samples_per_example=6`, first pass capped at 2000 train examples
- Success: >=1000 verified programs from the 2000-example pass; acceptance >=12%; median length <=16
- If fail: keep MATH as smaller stress test; spend at most 120 extra GPUh trying to improve it

### Block 4: GSM8K library mining and causal variants
- Depends on: 2
- GPUh: 12
- Required code changes: export MDL gains, support, reuse; build MDL library, random library, frequency-matched library, matched-size uncompressed bank
- Artifacts: `library_mdl_16.json`, `library_random_16_seed{42,123,456}.json`, `library_freq_16.json`, `program_bank_matched_16.json`, `plans_with_programs.json`
- Success: size=16 exactly; mean support >=6; top-quartile MDL gains positive; redundancy <=15%; coherence >=70%
- If fail: raise min support and prune noisy subroutines

### Block 5: Audited GSM8K MCD split
- Depends on: 2, 4
- GPUh: 6
- Command sketch:
```bash
python3 scripts/build_mcd_split.py \
  --programs results/gsm8k_teacher_verified/all_programs.json \
  --output results/gsm8k_mcd_split_seed42.json \
  --train_ratio 0.6 --dev_ratio 0.2 --test_ratio 0.2 \
  --max_atom_tvd 0.02 --min_unseen_compounds 0.40 \
  --num_trials 5000 --seed 42
```
- Repeat for seeds 123 and 456
- Success: all 3 seeds satisfy TVD <=0.02 and unseen compounds >=0.40; template unseen >=0.40; op-trigram unseen >=0.35; solution-graph edge unseen >=0.30
- If fail: add local swap search; if still weak after 12 extra hours, lower hardness claim explicitly

### Block 6: Student A compose and flat training
- Depends on: 4, 5
- GPUh: 240
- Command sketch:
```bash
torchrun --nproc_per_node=4 scripts/train_template_compiler.py \
  --mode compose \
  --training_data results/gsm8k_compose_train_seed42.json \
  --output_dir results/planner_qwen9b/compose_seed42

torchrun --nproc_per_node=4 scripts/train_template_compiler.py \
  --mode flat \
  --training_data results/gsm8k_flat_train_seed42.json \
  --output_dir results/planner_qwen9b/flat_seed42
```
- Repeat for 3 seeds
- Success: all runs complete; train loss drops >=30% from step 100 to end; compose dev valid-plan >=45%; flat dev valid-program >=35%
- If fail: reduce max sequence length first; if needed fall back to stricter QLoRA-only settings

### Block 7: Student A CoT-distilled baseline
- Depends on: 2, 5
- GPUh: 240
- Required code changes: `generate_cot_distill_data.py`, `train_cot_student.py`, matched eval path
- Command sketch:
```bash
python3 scripts/generate_cot_distill_data.py \
  --teacher_model Qwen/Qwen3.5-32B \
  --dataset gsm8k \
  --split results/gsm8k_mcd_split_seed42.json \
  --output results/gsm8k_cot_distill_seed42.json

torchrun --nproc_per_node=4 scripts/train_cot_student.py \
  --model Qwen/Qwen3.5-9B \
  --train_file results/gsm8k_cot_distill_seed42.json \
  --output_dir results/cot_qwen9b/seed42
```
- Repeat for 3 seeds
- Success: distilled student lands within 5 points of direct CoT decoding on random split; all seeds train successfully
- If fail: shorten CoT cap; if still unstable, downgrade to answer-only distillation

### Block 8: Retrieval and memory-control baselines
- Depends on: 4, 5, 6
- GPUh: 60
- Required code changes: raw trace retrieval mode, matched-size exemplar bank mode, compute accounting
- Variants: `raw_trace_retrieval`, `retrieval_compose`, `uncompressed_program_bank`, `random_library`, `frequency_matched_library`
- Success: all baselines run on identical splits; all log accuracy, execution, tokens, latency; matched-size uncompressed bank trails compressed library by >=5 points if causal compression claim is retained
- If fail: keep the baseline result and reduce compression language if needed

### Block 9: Student A main GSM8K evaluation
- Depends on: 6, 7, 8
- GPUh: 120
- Evaluate: random, MCD-medium, MCD-hard; 3 training seeds; 3 split seeds when feasible
- Success: `compose` beats `32B-CoT-distilled 9B` by >=15 points on GSM8K MCD-hard for at least 2/3 split seeds; bootstrap 95% CI excludes 0; also beats `flat_inline` by >=8 points
- If fail: reframe thesis to narrower structure-aware gains

### Block 10: Student B portability
- Depends on: 4, 5, 7
- GPUh: 250
- Same pipeline as Block 6 but with Student B, plus matching CoT-distilled baseline
- Success: frozen library improves Student B over its CoT-distilled baseline by >=8 points on GSM8K MCD-hard; positive sign on all split seeds
- If fail: reduce portability claim from "portable" to "partially transferable"

### Block 11: Compression-as-diagnostic sweep
- Depends on: 4, 6, 9
- GPUh: 220
- Required code changes: `run_compression_sweep.py`
- Command sketch:
```bash
python3 scripts/run_compression_sweep.py \
  --dataset gsm8k \
  --library_sizes 4 8 16 32 \
  --split_seeds 42 123 456 \
  --student_model Qwen/Qwen3.5-9B
```
- Success: compression ratio has the largest standardized positive regression coefficient; `p<0.05`; adjusted `R^2>=0.35`
- If fail: keep only descriptive correlation language

### Block 12: Search-time repair
- Depends on: 6, 8, 9
- GPUh: 160
- Required code changes: typed repair operators and equal-budget accounting in `src/mcts_search.py`; equal-budget flat/retrieval search baselines
- Command sketch:
```bash
python3 scripts/eval_repair_search.py \
  --student_ckpt results/planner_qwen9b/compose_seed42 \
  --library results/gsm8k_teacher_verified/library_mdl_16.json \
  --split results/gsm8k_mcd_split_seed42.json \
  --max_simulations 64 \
  --max_calls 6
```
- Success: >=25% of initially failed plans repaired; equal-budget flat and retrieval search each at least 5 points worse; identical median executor budget across compared methods
- If fail: keep search only as analysis, remove it from title/abstract claims

### Block 13: Library audit and failure analysis
- Depends on: 4, 9, 12
- GPUh: 20
- Required code changes: `audit_library.py`, `analyze_failures.py`
- Command sketch:
```bash
python3 scripts/audit_library.py \
  --library results/gsm8k_teacher_verified/library_mdl_16.json \
  --plans results/gsm8k_teacher_verified/plans_with_programs.json

python3 scripts/analyze_failures.py \
  --predictions results/eval/gsm8k_mcd_hard_predictions.json \
  --n_sample 150
```
- Success: audit table generated; >=150 GSM8K failures labeled; no hidden failure mode >20% of sampled failures
- If fail: extend annotation by 2 days; do not claim interpretability without it

### Block 14: MATH stress-test transfer
- Depends on: 3, 6, 7, 9
- GPUh: 180
- Evaluate: frozen GSM8K library transfer, optional MATH-specific library upper bound, Student A compose vs CoT-distilled vs flat
- Success: frozen-library compose is not worse than CoT-distilled by more than 3 points; MATH-specific library beats flat
- If fail: keep MATH strictly as stress-test analysis

## Critical Path
```
Block 1 → Block 2 → Block 4 → Block 5 → Block 6 → Block 9 → Block 12 → Block 13
                                          Block 7 ↗        ↗
                                          Block 8 ↗
```

Parallel side paths after GSM8K stabilizes:
- Block 3 (MATH extraction) — parallel with Block 2
- Block 10 (Student B) — parallel with Block 9
- Block 11 (compression sweep) — parallel with Block 9
- Block 14 (MATH stress) — after Block 3 + Block 9

## Week-by-Week Schedule

### Week 1 (Apr 7-13)
- Block 1: smoke test
- Block 2: GSM8K teacher extraction (start)
- Block 3: MATH extraction (start, parallel)
- Implement step-level verification
- **Gate A**: >=1000 GSM8K verified traces by week end

### Week 2 (Apr 14-20)
- Finish Block 2, Block 3
- Block 4: library mining + causal variants
- Block 5: audited MCD split
- Generate random/frequency/uncompressed controls
- **Gate B**: at least one split seed meets hardness thresholds AND library coherence >=70%

### Week 3 (Apr 21-27)
- Block 6: Student A compose/flat training
- Block 7: CoT-distilled baseline training
- Start Block 8: baseline evals
- **Gate C**: compose dev valid-plan >=45% AND CoT-distilled training stable

### Week 4 (Apr 28-May 4)
- Block 9: main GSM8K eval
- Block 10: Student B portability
- Block 11: compression sweep
- **Gate D**: compose beats CoT-distilled by >=15 on first seed
- **Gate E**: Student B gain >=8

### Week 5 (May 5-11)
- Block 12: search-time repair
- Block 13: library audit + failure analysis
- Block 14: MATH stress test
- Freeze plots and paper text
- **Gate F**: repair >=25% and wins equal-budget comparison

## Stop/Go Gates Summary

| Gate | When | Criterion | If Fail |
|------|------|-----------|---------|
| A | End Week 1 | >=1000 GSM8K verified traces | Narrow scope, raise samples |
| B | Mid Week 2 | TVD<=0.02, unseen>=0.40, coherence>=70% | Weaken split claim |
| C | End Week 3 | Compose dev valid-plan>=45% | Reduce seq length, QLoRA |
| D | End Week 4 | Compose beats CoT-distill by >=15 | Reframe thesis |
| E | End Week 4 | Student B gain >=8 | Narrow portability claim |
| F | Mid Week 5 | Repair >=25%, wins equal-budget | Remove search claim |

## GPU Budget Summary

| Block | GPUh | Critical Path? |
|-------|------|---------------|
| 1. Smoke | 2 | Yes |
| 2. GSM8K extraction | 180 | Yes |
| 3. MATH extraction | 260 | No |
| 4. Library mining | 12 | Yes |
| 5. MCD split | 6 | Yes |
| 6. Student A training | 240 | Yes |
| 7. CoT-distilled baseline | 240 | Yes |
| 8. Retrieval baselines | 60 | Yes |
| 9. Main GSM8K eval | 120 | Yes |
| 10. Student B portability | 250 | No |
| 11. Compression sweep | 220 | No |
| 12. Search repair | 160 | No |
| 13. Audit + failure analysis | 20 | Yes |
| 14. MATH stress test | 180 | No |
| **Total planned** | **1950** | |
| Contingency | 330 | |
| **Total with contingency** | **2280** | |

## Risk Matrix

| Risk | Prob | Impact | Signal | Mitigation |
|------|------|--------|--------|------------|
| Low verified extraction | Medium | Critical | acceptance <15% after 1k | increase samples, tighten prompt, pause MATH |
| Weak MCD split | Medium | Critical | unseen compounds <0.30 after 5k trials | local swap search, revised compounds |
| Poor library coherence | Medium | High | subroutines rely on default bindings | raise support threshold, prune |
| Strong CoT baseline | Medium | Critical | compose gap <5 on dev | narrow claim to controlled splits |
| Student B instability | Medium | Medium | repeated OOM/invalid outputs | switch to Llama-3-8B-Instruct |
| Search leakage | Low | Critical | unrealistically high dev gains | hard budget accounting, code review |
| Compression nonsignificant | Medium | High | unstable coefficient across seeds | downgrade to descriptive diagnostic |
| MATH overconsumes budget | High | Medium | >120 GPUh with <500 verified | freeze as limited stress test |
| Train/test leakage | Low | Critical | split built after full-data mining | rebuild library from train only |

## Submission Deliverables
1. `results/gsm8k_teacher_verified/` — verified traces + rejection logs
2. `results/gsm8k_mcd_split_seed{42,123,456}.json` — splits + audit summaries
3. Student A compose, flat, CoT-distilled checkpoints
4. Student B compose and CoT-distilled checkpoints
5. Retrieval, uncompressed bank, random/frequency-matched library outputs
6. Repair-search outputs with equal-budget logs
7. Library audit tables and failure annotations
8. Compression-regression table

If any of deliverables 1-6 are missing, submission should be delayed or claims reduced.
