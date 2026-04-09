# Verified Procedural Abstractions for Compositional Math Reasoning

Compression-mined, execution-verified subroutine libraries that transfer across model sizes and predict compositional generalization via MDL compression ratio.

## Setup

```bash
git clone <this-repo> && cd nips-templatebank
bash setup.sh            # venv + PyTorch (CUDA 12.8) + all deps
source .venv/bin/activate
```

**Requirements**: Python >= 3.10, PyTorch (CUDA 12.8), 1+ NVIDIA GPU (auto-detected).

## GPU Auto-Detection

All scripts automatically detect available GPUs:
- **Training**: `torchrun` via `scripts/gpu_utils.sh` — auto-sets `--nproc_per_node`
- **Inference**: `device_map="auto"` — shards large models across all GPUs
- Override: `CUDA_VISIBLE_DEVICES=0,1 bash run.sh`

## Quick Start (Smoke Test)

```bash
SMOKE=1 bash run.sh --smoke
```

## Full Experiment Pipeline

The pipeline has 7 stages. GPU hours estimated for 4x H100 80GB.

### Stage 1: Teacher Program Extraction (~180 GPU-h)

Extract verified DSL programs from GSM8K using Qwen3.5-27B:

```bash
python scripts/extract_templates.py \
  --config configs/template_config.yaml \
  --output_dir results/gsm8k_teacher_verified/
```

Programs are step-level verified: parse-valid, type-valid, each step re-executes, final answer matches gold.

### Stage 2: Library Mining + MCD Split (~18 GPU-h)

Rebuild library from train partition only (prevents test leakage), then construct MCD splits:

```bash
# Build MCD split
python scripts/build_mcd_split.py \
  --programs results/gsm8k_teacher_verified/all_programs.json \
  --output results/gsm8k_mcd_split_seed42.json \
  --max_atom_tvd 0.02 --min_unseen_compounds 0.40 \
  --num_trials 5000 --seed 42

# Rebuild library + training data from train partition
python scripts/extract_templates.py \
  --config configs/template_config.yaml \
  --post_split \
  --split_path results/gsm8k_mcd_split_seed42.json \
  --output_dir results/gsm8k_teacher_verified/
```

Generate ablation control libraries:

```bash
# Frequency-matched (top-K by support, no MDL)
python scripts/ablation_controls.py --ablation frequency_matched \
  --library_path results/gsm8k_teacher_verified/subroutine_library.json \
  --programs_path results/gsm8k_teacher_verified/all_programs.json \
  --output_dir results/ablation/frequency_matched

# Uncompressed program bank (matched-size, no compression)
python scripts/ablation_controls.py --ablation uncompressed_bank \
  --library_path results/gsm8k_teacher_verified/subroutine_library.json \
  --programs_path results/gsm8k_teacher_verified/all_programs.json \
  --output_dir results/ablation/uncompressed_bank
```

### Stage 3: Train Student Planners (~480 GPU-h)

Train compose planner + flat baseline + CoT-distilled baseline (3 seeds each):

```bash
NUM_GPUS=$(nvidia-smi -L | wc -l)

# Compose planner (ours)
torchrun --nproc_per_node=$NUM_GPUS scripts/train_template_compiler.py \
  --mode compose \
  --training_data results/gsm8k_teacher_verified/compose_train.json \
  --output_dir results/planner_qwen9b/compose_seed42

# Flat baseline (same DSL, no library)
torchrun --nproc_per_node=$NUM_GPUS scripts/train_template_compiler.py \
  --mode flat \
  --training_data results/gsm8k_teacher_verified/flat_train.json \
  --output_dir results/planner_qwen9b/flat_seed42

# CoT-distilled baseline (primary comparison)
python scripts/generate_cot_distill_data.py \
  --teacher_model Qwen/Qwen3.5-27B \
  --dataset gsm8k \
  --split results/gsm8k_mcd_split_seed42.json \
  --output results/gsm8k_cot_distill_seed42.json

torchrun --nproc_per_node=$NUM_GPUS scripts/train_cot_student.py \
  --model Qwen/Qwen3.5-9B \
  --train_file results/gsm8k_cot_distill_seed42.json \
  --output_dir results/cot_qwen9b/seed42
```

Repeat for seeds 123, 456. Training auto-resumes from latest checkpoint.

### Stage 4: Evaluation (~120 GPU-h)

```bash
python scripts/eval_template_reasoning.py \
  --config configs/template_config.yaml
```

Evaluates: compose, flat_inline, cot_budget, retrieval_compose, cot_distilled, raw_trace_retrieval, uncompressed_bank, random_library, frequency_matched, compose_with_repair.

### Stage 5: Compression Diagnostic Sweep (~220 GPU-h)

```bash
python scripts/run_compression_sweep.py \
  --dataset gsm8k \
  --library_sizes 4 8 16 32 \
  --split_seeds 42 123 456 \
  --results_dir results/compression_sweep

# After all conditions complete:
python scripts/run_compression_sweep.py --analyze_only \
  --results_dir results/compression_sweep
```

### Stage 6: Library Audit + Failure Analysis

```bash
# Library semantic audit
python scripts/audit_subroutines.py \
  --library results/gsm8k_teacher_verified/subroutine_library.json \
  --plans results/gsm8k_teacher_verified/plans_with_programs.json

# Failure categorization (8 bins, >=150 samples)
python scripts/analyze_failures.py \
  --predictions results/eval/gsm8k_mcd_hard_predictions.json \
  --library results/gsm8k_teacher_verified/subroutine_library.json \
  --n_sample 150
```

### Stage 7: One-Command Full Pipeline

```bash
bash run.sh
```

Runs Stages 1-4 sequentially with auto-checkpointing. Skips completed phases via `results/.phase_markers/`.

## Checkpoint Resume

- **Training**: `--resume auto` (recovers from latest checkpoint)
- **Pipeline**: phase markers in `results/.phase_markers/`
- **Force re-run**: `FORCE_RERUN=1 bash run.sh`

## Project Structure

```
src/
  template_dsl.py              # Core DSL: Program, Step, Subroutine, Executor
  mcd_split.py                 # CFQ-style MCD split construction
  mcts_search.py               # MCTS search + RepairSearcher

scripts/
  extract_templates.py         # Stage 1: Teacher → verified programs → library
  build_mcd_split.py           # Stage 2: MCD split
  train_template_compiler.py   # Stage 3a: Compose/flat planner training
  train_cot_student.py         # Stage 3b: CoT-distilled baseline
  generate_cot_distill_data.py # Generate CoT distillation data
  eval_template_reasoning.py   # Stage 4: Full evaluation
  run_compression_sweep.py     # Stage 5: Compression diagnostic regression
  audit_subroutines.py         # Stage 6a: Library semantic audit
  analyze_failures.py          # Stage 6b: Failure categorization
  ablation_controls.py         # Ablation: frequency_matched, uncompressed_bank, etc.
  gpu_utils.sh                 # Auto GPU detection + torchrun helpers

configs/
  template_config.yaml         # All hyperparameters

refine-logs/
  FINAL_PROPOSAL.md            # Refined research proposal
  EXPERIMENT_PLAN.md           # 14-block experiment plan with stop/go gates
```

## Key Models

| Role | Model | Usage |
|------|-------|-------|
| Teacher | Qwen/Qwen3.5-27B | Program extraction + library mining |
| Student A | Qwen/Qwen3.5-9B | Compose/flat planner (LoRA r=32, α=64) |
| Student B | Qwen/Qwen3.5-4B | Portability test |

## Claims

1. **Portable Verified Library**: frozen 32B-mined library → 9B beats CoT-distilled by ≥15pts on MCD-hard
2. **Compression Diagnostic**: MDL ratio outpredicts library size, trace length, teacher accuracy
3. **Search-Time Repair**: typed MCTS repair recovers ≥25% failed plans under matched budget
