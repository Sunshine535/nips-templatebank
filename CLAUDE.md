# Project: nips-templatebank

## Project goal

**SEVAL: Self-Evolving Verified Abstraction Libraries for Compositional Math Reasoning.** Four claims: (1) RLVR-evolved library outperforms frozen library by >=10pts on MATH MCD-hard, (2) RLVR expands (not just optimizes) compositional reasoning (CoT-Pass@K), (3) test-time tool building recovers >=20% failures, (4) evolved library transfers cross-model by >=8pts. See refine-logs/FINAL_PROPOSAL_V2.md and refine-logs/EXPERIMENT_PLAN_V2.md.

## Key models

- `Qwen/Qwen3.5-32B` — Teacher (program extraction + library mining)
- `Qwen/Qwen3.5-9B` — Student A / Planner (LoRA r=32, α=64)
- `Qwen/Qwen3.5-3B` or `Llama-3-8B-Instruct` — Student B (portability test)

## Key datasets

- GSM8K — Train (7473) + Test (1319)
- MATH — Train (7500) + Test (5000)

## Repo map

- `src/template_dsl.py` — Core typed DSL: Program, Subroutine, Executor, CompositionPlan + Library Evolution API
- `src/rlvr_evolution.py` — **SEVAL core**: GRPO reward, LibraryEvolver, CoT-Pass@K, SEVALTrainer
- `src/test_time_tools.py` — **Test-time tool building**: failure analysis, tool generation, verification
- `src/mcd_split.py` — CFQ-style Maximum Compound Divergence split builder
- `src/mcts_search.py` — MCTS over composition plans + beam search repair
- `src/template_algebra.py` — Legacy template algebra (not used in main experiments)
- `scripts/extract_templates.py` — Phase 0: teacher -> programs -> library -> plans
- `scripts/build_mcd_split.py` — Phase 0: MCD split construction
- `scripts/train_seval.py` — **Phase 1: GRPO + library evolution training**
- `scripts/eval_test_time_tools.py` — **Phase 2: test-time tool building evaluation**
- `scripts/train_template_compiler.py` — Compose/flat planner training (baselines)
- `scripts/eval_template_reasoning.py` — Full evaluation with all baselines
- `scripts/run_all_experiments.sh` — Master pipeline orchestration
- `scripts/gpu_utils.sh` — GPU auto-detection and torchrun setup
- `configs/template_config.yaml` — All experiment configuration (incl. SEVAL section)

## Common commands

```bash
bash setup.sh
source .venv/bin/activate

# Smoke test
SMOKE=1 bash run.sh --smoke

# Full pipeline (~1830 GPU-hours)
bash run.sh

# SEVAL GRPO + evolution training
torchrun --nproc_per_node=$(nvidia-smi -L | wc -l) scripts/train_seval.py \
  --config configs/template_config.yaml \
  --library results/math_teacher_verified/subroutine_library.json \
  --train_data results/math_teacher_verified/compose_train.json \
  --output_dir results/seval/seed42

# Test-time tool building eval
python scripts/eval_test_time_tools.py \
  --library results/seval/seed42/library_final.json \
  --model_dir results/seval/seed42/model_final \
  --eval_data results/math_mcd_split_seed42.json

# Evaluate all baselines
python scripts/eval_template_reasoning.py --max_samples 100
```

## Experiment phases

| Phase | Description | Est. GPU-h |
|-------|-------------|-----------|
| 0 | MATH teacher extraction + library mining + MCD split | ~278 |
| 1 | **SEVAL GRPO + library evolution** (3 seeds) | ~500 |
| 2 | **Test-time tool building evaluation** | ~200 |
| 3 | Transfer to 3B + CoT-distilled baselines | ~440 |
| 4 | Main evaluation (12 methods × 3 splits) | ~150 |
| 5 | Ablations + audit + failure analysis | ~170 |

## Data and outputs

- Programs: `results/templates/all_programs.json`
- Library: `results/templates/subroutine_library.json`
- Training data: `results/templates/compose_train.json`, `flat_train.json`
- MCD split: `results/mcd_split.json`
- Compose planner: `results/planner/compose/`
- Flat baseline: `results/planner/flat/`
- Evaluation: `results/eval/`
- Ablations: `results/ablation_L{4,8,16,32}/`
- Logs: `logs/`

## Environment

- Python 3.10+, PyTorch 2.10+ (CUDA 12.8)
- Key deps: transformers, datasets, accelerate, trl, peft, wandb
- Optional: outlines (constrained decoding), sentence-transformers (retrieval baseline)
- Training uses `torchrun` for multi-GPU

## Checkpoint resume

- Training automatically resumes from latest checkpoint (--resume auto)
- Pipeline phases skip if marker file exists in `results/.phase_markers/`
- Force re-run: `FORCE_RERUN=1`

