# SEVAL Experiment Plan V2

## Scope
**Title**: Self-Evolving Verified Abstraction Libraries (SEVAL)
**Teacher**: Qwen3.5-27B
**Student A**: Qwen3.5-9B (RLVR evolution target)
**Student B**: Qwen3.5-4B (transfer test)
**Benchmark**: MATH (primary), GSM8K (secondary)
**Hardware**: 4× H100 80GB for 5 weeks
**Budget**: 1828 GPUh planned, 200 GPUh contingency

## Experiment Blocks

### Block 0: Smoke Test
- Depends: none
- GPUh: 2
- Command: `SMOKE=1 bash run.sh --smoke`
- Success: all scripts finish; no import/execution errors

### Block 1: MATH Teacher Extraction + Step Verification
- Depends: 0
- GPUh: 260
- Command:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/extract_templates.py \
  --config configs/template_config.yaml \
  --teacher_model Qwen/Qwen3.5-27B \
  --dataset math \
  --samples_per_example 6 \
  --output_dir results/math_teacher_verified/
```
- Success: ≥1500 verified MATH programs; acceptance ≥15%
- If fail: raise samples to 8; cap MATH at 2000 examples

### Block 2: Library Mining + MCD Split
- Depends: 1
- GPUh: 18
- Command:
```bash
# Library mining
python scripts/extract_templates.py --post_split \
  --programs results/math_teacher_verified/all_programs.json \
  --output_dir results/math_teacher_verified/

# MCD split (3 seeds)
for SEED in 42 123 456; do
  python scripts/build_mcd_split.py \
    --programs results/math_teacher_verified/all_programs.json \
    --output results/math_mcd_split_seed${SEED}.json \
    --max_atom_tvd 0.02 --min_unseen_compounds 0.40 \
    --num_trials 5000 --seed $SEED
done
```
- Success: library ≥16 subs, coherence ≥70%; all 3 splits satisfy thresholds
- **Gate A (Week 1 end)**: ≥1000 verified traces + split passes

### Block 3: SEVAL GRPO Training + Library Evolution (3 seeds)
- Depends: 2
- GPUh: 500
- Command:
```bash
for SEED in 42 123 456; do
  torchrun --nproc_per_node=4 scripts/train_seval.py \
    --config configs/template_config.yaml \
    --library results/math_teacher_verified/subroutine_library.json \
    --train_data results/math_teacher_verified/compose_train.json \
    --eval_data results/math_teacher_verified/flat_train.json \
    --output_dir results/seval/seed${SEED} \
    --grpo_num_steps 2000 \
    --evolution_interval 200 \
    --evolution_rounds 5 \
    --grpo_learning_rate 5e-6
done
```
- Success: at least 1 evolution round per seed; library grows; GRPO loss decreases
- Track: library snapshots L₀→L₁→L₂→..., CoT-Pass@K at each eval interval
- **Gate B (Week 2 mid)**: library evolves at least once; no training collapse

### Block 4: CoT-Distilled Baselines
- Depends: 1, 2
- GPUh: 240
- Command:
```bash
# Generate CoT distillation data
python scripts/generate_cot_distill_data.py \
  --teacher_model Qwen/Qwen3.5-27B \
  --dataset math \
  --split results/math_mcd_split_seed42.json \
  --output results/math_cot_distill_seed42.json

# Train 9B CoT baseline (3 seeds)
for SEED in 42 123 456; do
  torchrun --nproc_per_node=4 scripts/train_cot_student.py \
    --model Qwen/Qwen3.5-9B \
    --train_file results/math_cot_distill_seed${SEED}.json \
    --output_dir results/cot_9b/seed${SEED}
done

# Train 3B CoT baseline (3 seeds)
for SEED in 42 123 456; do
  torchrun --nproc_per_node=4 scripts/train_cot_student.py \
    --model Qwen/Qwen3.5-4B \
    --train_file results/math_cot_distill_seed${SEED}.json \
    --output_dir results/cot_3b/seed${SEED}
done
```
- Success: all baselines train successfully

### Block 5: Frozen Library Compose Baseline
- Depends: 2
- GPUh: 120
- Same as original Block 6 but with frozen L₀ only (no evolution)
- Provides the C1 comparison target

### Block 6: Main Evaluation — Claims C1 & C2
- Depends: 3, 4, 5
- GPUh: 150
- Evaluate ALL methods on MCD-hard/medium/random:
  - SEVAL-evolved compose (each seed)
  - Frozen L₀ compose
  - CoT-distilled 9B
  - flat_inline
  - All 12 baselines
- CoT-Pass@K evaluation for C2:
```bash
python scripts/eval_cot_passk.py \
  --model_evolved results/seval/seed42/model_final \
  --model_base Qwen/Qwen3.5-9B \
  --library_evolved results/seval/seed42/library_final.json \
  --library_frozen results/math_teacher_verified/subroutine_library.json \
  --eval_data results/math_mcd_split_seed42.json \
  --output results/seval/cot_passk_eval.json
```
- Success C1: evolved beats frozen by ≥10 pts on MCD-hard (2/3 seeds)
- Success C2: CoT-Pass@64 evolved > base model (2/3 seeds)
- **Gate C (Week 3 end)**: C1 positive signal on at least 1 seed

### Block 7: Test-Time Tool Building Evaluation — Claim C3
- Depends: 3, 6
- GPUh: 200
```bash
python scripts/eval_test_time_tools.py \
  --library results/seval/seed42/library_final.json \
  --model_dir results/seval/seed42/model_final \
  --eval_data results/math_mcd_split_seed42.json \
  --output_dir results/seval/test_time_eval_seed42 \
  --max_new_tools 3 \
  --max_verify_attempts 10
```
- Success C3: recovery rate ≥20% under matched budget
- Also run: MCTS search without building (budget-matched control)
- **Gate D (Week 4)**: recovery rate ≥15% (relaxed)

### Block 8: Transfer — Claim C4
- Depends: 3
- GPUh: 200
```bash
# Transfer evolved library to 3B
for SEED in 42 123 456; do
  torchrun --nproc_per_node=4 scripts/train_template_compiler.py \
    --mode compose \
    --library results/seval/seed${SEED}/library_final.json \
    --student_model Qwen/Qwen3.5-4B \
    --output_dir results/transfer_3b/seed${SEED}
done
```
- Success C4: transfer beats CoT-distilled 3B by ≥8 pts (2/3 seeds)

### Block 9: Ablations
- Depends: 3, 6
- GPUh: 150
- Ablation matrix:
  1. Evolution rounds: 0 (frozen) vs 1 vs 3 vs 5
  2. Library size cap: 16 vs 32 vs 64
  3. Evolution interval: 100 vs 200 vs 500
  4. Remove test-time building (isolation)
  5. Random evolution (add random subroutines instead of MDL-selected)

### Block 10: Library Audit + Failure Analysis
- Depends: 3, 6, 7
- GPUh: 20
```bash
python scripts/audit_subroutines.py \
  --library_initial results/math_teacher_verified/subroutine_library.json \
  --library_evolved results/seval/seed42/library_final.json \
  --plans results/seval/seed42/

python scripts/analyze_failures.py \
  --predictions results/seval/test_time_eval_seed42/ \
  --n_sample 150
```
- Output: evolution dynamics plot, subroutine quality table, failure taxonomy

## Critical Path
```
Block 0 → Block 1 → Block 2 → Block 3 → Block 6 → Block 7 → Block 10
                                Block 4 ↗        ↗
                                Block 5 ↗
```
Parallel: Block 4, 5 (baselines) with Block 3 (SEVAL training)
After Block 6: Block 7 (test-time) and Block 8 (transfer) in parallel

## Week-by-Week Schedule

### Week 1 (Apr 9-13)
- Block 0: smoke test
- Block 1: MATH teacher extraction (start)
- **Gate A**: ≥1000 verified traces

### Week 2 (Apr 14-20)
- Block 1: finish extraction
- Block 2: library mining + MCD split
- Block 3: start SEVAL GRPO training
- Block 4: start CoT-distilled baselines (parallel)
- **Gate B**: library evolves + GRPO stable

### Week 3 (Apr 21-27)
- Block 3: finish SEVAL training (all seeds)
- Block 5: frozen compose baseline
- Block 6: main evaluation (C1 + C2)
- **Gate C**: C1 positive signal

### Week 4 (Apr 28-May 4)
- Block 7: test-time tool building eval (C3)
- Block 8: transfer to 3B (C4)
- Block 9: ablations
- **Gate D**: C3 recovery ≥15%

### Week 5 (May 5-11)
- Block 10: audit + failure analysis
- Figures and tables
- Paper writing
- **Gate E**: all 4 claims assessed

## Stop/Go Gates
| Gate | When | Criterion | If Fail |
|------|------|-----------|---------|
| A | End W1 | ≥1000 MATH verified | Fallback to GSM8K |
| B | Mid W2 | Library evolves | Debug evolution logic |
| C | End W3 | C1 positive (≥5 pts) | Reframe to efficiency |
| D | W4 | C3 recovery ≥15% | Demote to analysis |
| E | W5 | ≥2/4 claims hold | Narrow paper scope |
