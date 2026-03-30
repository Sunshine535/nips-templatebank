# Project: nips-templatebank

## Project goal

Subroutine Composition for Mathematical Reasoning — prove that reusable subroutines improve program-compound generalization beyond a strong primitive DSL. Mine typed subroutine library from GSM8K+MATH, train a planner (Qwen3.5-9B + LoRA) to compose subroutines, evaluate on CFQ-style MCD split.

## Key models

- `Qwen/Qwen3.5-32B` — Teacher (program extraction only)
- `Qwen/Qwen3.5-9B` — Planner / Flat baseline (LoRA r=64, α=128)

## Key datasets

- GSM8K — Train (7473) + Test (1319)
- MATH — Train (7500) + Test (5000)

## Repo map

- `src/template_dsl.py` — Core typed DSL: Program, Subroutine, Executor, CompositionPlan
- `src/mcd_split.py` — CFQ-style Maximum Compound Divergence split builder
- `src/template_algebra.py` — Legacy template algebra (not used in main experiments)
- `scripts/extract_templates.py` — Stage 1: teacher -> programs -> library -> plans
- `scripts/build_mcd_split.py` — Stage 2: MCD split construction
- `scripts/train_template_compiler.py` — Stage 3: planner (compose/flat) training
- `scripts/eval_template_reasoning.py` — Stage 4: full evaluation with all baselines
- `scripts/run_all_experiments.sh` — Master pipeline orchestration
- `scripts/gpu_utils.sh` — GPU auto-detection and torchrun setup
- `configs/template_config.yaml` — All experiment configuration

## Common commands

```bash
bash setup.sh
source .venv/bin/activate

# Smoke test (synthetic data, student model)
SMOKE=1 bash run.sh --smoke

# Full pipeline (~1970 GPU-hours)
bash run.sh

# Background
nohup bash run.sh > run.log 2>&1 &

# Force re-run all phases
FORCE_RERUN=1 bash run.sh

# Train only compose planner
torchrun --nproc_per_node=4 scripts/train_template_compiler.py --mode compose

# Train only flat baseline
torchrun --nproc_per_node=4 scripts/train_template_compiler.py --mode flat

# Evaluate only
python scripts/eval_template_reasoning.py --max_samples 100
```

## Experiment phases

| Phase | Description | Est. GPU-h |
|-------|-------------|-----------|
| 1 | Extract programs + library (32B teacher) | ~220 |
| 2 | Build MCD split | ~10 |
| 3a | Train compose planner (torchrun) | ~300 |
| 3b | Train flat baseline (torchrun) | ~300 |
| 4 | Full evaluation | ~240 |
| 5 | Library size ablation L={4,8,16,32} | ~220 |

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

## Remote server

- SSH: `sshpass -p '123456' ssh -p 30022 -o PubkeyAuthentication=no "wujn@root@ssh-362.default"@222.223.106.147`
- GPU: 4x NVIDIA H800 80GB
- Code dir: `/gfs/space/private/wujn/Research/nips-templatebank`
- Background: `nohup bash run.sh > run.log 2>&1 &`
