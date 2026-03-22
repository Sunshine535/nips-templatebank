# Template Algebra: Formal Reasoning Template Composition for Mathematical Problem Solving

---

## How to Run (Complete Guide)

### Requirements

- Linux server with NVIDIA GPU (4-8x A100 80GB recommended)
- CUDA 12.8 compatible driver
- `git`, `curl` installed
- ~200GB disk space (model weights + checkpoints)

### Step 1: Clone and Run (One Command)

```bash
git clone https://github.com/Sunshine535/nips-templatebank.git
cd nips-templatebank
bash run.sh
```

`run.sh` will automatically:
1. Install `uv` package manager (if not present)
2. Create Python 3.10 virtual environment
3. Install PyTorch 2.10 + CUDA 12.8
4. Install all dependencies
5. Run **all experiments** in full production mode
6. Display real-time progress in terminal and save to `run.log`

### Step 2: Monitor Progress

If running in foreground (default):
```bash
# Progress is displayed in real-time
# Press Ctrl+C to stop (can resume later with bash run.sh)
```

If running in background (recommended for long experiments):
```bash
nohup bash run.sh > run.log 2>&1 &
tail -f run.log          # Watch progress
```

### Step 3: Check Completion

```bash
cat results/.pipeline_done
# If this file exists and shows "PIPELINE_COMPLETE", all experiments finished successfully
```

### Step 4: Package and Send Results

```bash
# Option A: Push to GitHub (recommended)
git add results/ logs/
git commit -m "Experiment results $(date +%Y%m%d)"
git push origin main

# Option B: Create tarball for manual transfer
bash collect_results.sh
# Creates: results_archive/nips-templatebank_results_YYYYMMDD_HHMMSS.tar.gz
# Send this file via scp/email/cloud drive
```

### Troubleshooting

| Problem | Solution |
|---------|----------|
| Experiment interrupted | Re-run `bash run.sh` — completed phases are automatically skipped |
| Want to re-run everything from scratch | `FORCE_RERUN=1 bash run.sh` |
| GPU out of memory | The script auto-detects GPUs; ensure CUDA drivers are installed |
| Network issues downloading models | Set `HF_ENDPOINT=https://hf-mirror.com` before running |
| Check which phases completed | `ls results/.phase_markers/` |

### Output Structure

After completion, key results are in:

```
nips-templatebank/
├── results/              # All experiment outputs (JSON, figures, metrics)
│   └── .pipeline_done    # Completion marker
├── logs/                 # Per-phase log files
├── run.log               # Full pipeline log
└── results_archive/      # Packaged tarballs (after collect_results.sh)
```

---

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
