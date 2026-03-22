# Template Algebra: Formal Reasoning Template Composition for Mathematical Problem Solving

---

## Quick Start

```bash
# 1. Clone and enter project
git clone https://github.com/Sunshine535/nips-templatebank.git
cd nips-templatebank

# 2. One-command setup + run all experiments
bash run.sh

# 3. (Optional) Run in background for long experiments
nohup bash run.sh > run.log 2>&1 &
tail -f run.log
```

### Check Completion

```bash
cat results/.pipeline_done   # Shows PIPELINE_COMPLETE when all phases finish
ls results/.phase_markers/   # See which individual phases completed
```

### Save and Send Results

```bash
# Option A: Push to GitHub
git add results/ logs/
git commit -m "Experiment results"
git push origin main

# Option B: Package as tarball
bash collect_results.sh
# Output: results_archive/nips-templatebank_results_YYYYMMDD_HHMMSS.tar.gz
```

### Resume After Interruption

Re-run `bash run.sh` — completed phases are automatically skipped.
To force re-run all phases: `FORCE_RERUN=1 bash run.sh`

## Project Structure

```
nips-templatebank/
├── README.md
├── LICENSE                              # MIT License
├── setup.sh                             # One-command environment setup
├── requirements.txt                     # Pinned dependencies
├── configs/
│   └── template_config.yaml             # Template extraction + training config
├── scripts/
│   ├── gpu_utils.sh                     # Shared GPU auto-detection
│   ├── run_all_experiments.sh           # Master pipeline (5 stages)
│   ├── extract_templates.py             # Stage 1: CoT → template bank
│   ├── run_template_operations.py       # Stage 2: 6 algebra operations
│   ├── train_template_compiler.py       # Stage 3: Two-stage SFT compiler
│   └── eval_template_reasoning.py       # Stage 4: GSM8K + MATH evaluation
├── src/                                 # Core library modules
├── results/                             # Experiment outputs
├── logs/                                # Training logs
└── docs/                                # Additional documentation
```

## Experiments

| # | Stage | Description | Est. Time (8×A100) |
|---|-------|-------------|-------------------|
| 1 | Template Extraction | Extract reasoning templates from GSM8K + MATH CoT traces | ~24 hrs |
| 2 | Template Operations | Test 6 algebraic operations on extracted template bank | ~36 hrs |
| 3 | Compiler Training | Two-stage SFT: (a) template selection, (b) variable filling | ~120 hrs |
| 4 | Evaluation | GSM8K + MATH accuracy vs. CoT baselines + ablations | ~48 hrs |
| 5 | Ablation Studies | Bank size sweep (10–300 templates), operation-type ablation | ~24 hrs |

## Timeline & GPU Hours

- **Model**: Qwen/Qwen3.5-9B
- **Total estimated GPU-hours**: ~4920 (8× A100-80GB)
- **Wall-clock time**: ~25–27 days on 8× A100

## Citation

```bibtex
@inproceedings{templatealgebra2026neurips,
  title     = {Template Algebra: Formal Reasoning Template Composition for Mathematical Problem Solving},
  author    = {Anonymous},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
