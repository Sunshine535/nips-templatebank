# Subroutine Composition for Mathematical Reasoning

Can reusable subroutines improve program-compound generalization beyond a strong primitive DSL?

We build a typed subroutine library from math CoT traces and train a planner that composes subroutines for new problems. On a CFQ-style MCD split, composition outperforms flat programs on unseen subroutine combinations while using fewer tokens.

## Quick Start

```bash
git clone https://github.com/Sunshine535/nips-templatebank.git
cd nips-templatebank

# Setup environment
bash setup.sh

# Smoke test (fast, uses student model, ~50 examples)
bash run.sh --smoke

# Full pipeline (~1970 GPU-hours on 8xA100)
bash run.sh
```

### Resume After Interruption

Re-run `bash run.sh` — completed phases are automatically skipped.
Force re-run: `FORCE_RERUN=1 bash run.sh`

### Check Progress

```bash
ls results/.phase_markers/     # See completed phases
cat results/.pipeline_done     # Shows PIPELINE_COMPLETE when done
```

## Method Overview

```
Phase 1: Teacher (32B) generates executable JSON-AST programs for GSM8K/MATH
Phase 2: Mine subroutine library (L=16) by structural clustering + MDL
Phase 3: Build MCD split (maximize compound divergence, low atom TVD)
Phase 4a: Train compose planner (9B + LoRA): problem -> composition plan
Phase 4b: Train flat-program baseline (same DSL, no library calls)
Phase 5: Evaluate on MCD split + IID split with all baselines
Phase 6: Library size ablation (L = 4, 8, 16, 32)
```

## Project Structure

```
nips-templatebank/
├── src/
│   ├── template_dsl.py          # Typed DSL: programs, subroutines, executor
│   ├── mcd_split.py             # CFQ-style MCD split builder
│   └── template_algebra.py      # Legacy template algebra operations
├── scripts/
│   ├── run_all_experiments.sh   # Master pipeline (6 stages)
│   ├── extract_templates.py     # Stage 1: teacher -> programs -> library
│   ├── build_mcd_split.py       # Stage 2: MCD split construction
│   ├── train_template_compiler.py  # Stage 3: planner + flat baseline training
│   ├── eval_template_reasoning.py  # Stage 4: full evaluation
│   └── gpu_utils.sh             # GPU auto-detection
├── configs/
│   └── template_config.yaml     # All experiment configuration
├── results/                     # Experiment outputs
├── logs/                        # Training/eval logs
└── ARIS_REVIEW.md               # Research review and experiment plan
```

## Evaluation Methods

| Method | Description |
|--------|-------------|
| compose | Our method: planner outputs composition plan using subroutine library |
| flat_inline | Critical baseline: same DSL, no library calls |
| direct_cot | Standard chain-of-thought prompting |
| cot_budget | Compute-matched CoT with majority vote |

## Metrics

- `accuracy`: End-to-end correctness
- `valid_plan_rate`: Fraction of outputs that parse as valid plans/programs
- `execution_success`: Fraction that execute without errors
- `fallback_rate`: Fraction routed to CoT fallback
- `fallback_free_accuracy`: Accuracy excluding fallback cases
- `total_tokens`: Average tokens per problem
- `latency`: Wall-clock time per problem

## Key Design Decisions

1. **Same DSL for compose and flat** — isolates the composition benefit
2. **Opaque subroutine IDs** (L00-L15) — no information leakage from names
3. **MCD split** — maximizes unseen compound divergence while keeping atom distribution similar
4. **Deterministic executor** — all programs are executable, not just text
5. **Fallback pipeline** — CoT fallback when plans fail (tracked separately)

## Compute Budget

| Stage | Description | Est. GPU-hours |
|-------|-------------|----------------|
| 1 | Program extraction (32B teacher) | ~220 |
| 2 | MCD split construction | ~10 |
| 3a | Compose planner training | ~300 |
| 3b | Flat baseline training | ~300 |
| 4 | Full evaluation | ~240 |
| 5 | Library size ablation | ~220 |
| | **Total** | **~1290** |

## Models

- **Teacher**: Qwen/Qwen3.5-32B (program extraction only)
- **Planner/Baseline**: Qwen/Qwen3.5-9B + LoRA (r=64, α=128)

## Citation

```bibtex
@inproceedings{subroutinecomposition2026neurips,
  title     = {Subroutine Composition for Mathematical Reasoning},
  author    = {Anonymous},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

## License

MIT License — see [LICENSE](LICENSE) for details.
