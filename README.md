# Template Algebra: Formal Reasoning Template Composition for Mathematical Problem Solving

> **NeurIPS 2026 Submission**

## Abstract

Chain-of-thought prompting elicits reasoning in LLMs but produces unstructured traces that resist systematic reuse and composition. We propose **Template Algebra**, a framework that extracts reusable reasoning templates from chain-of-thought traces and defines six algebraic operations—composition, decomposition, specialization, generalization, analogy transfer, and inversion—for constructing novel solution strategies. A two-stage compiler trained via LoRA SFT on Qwen3.5-9B learns to select and instantiate templates at inference time, achieving 4.1% improvement on GSM8K and 2.8% on MATH over standard chain-of-thought, demonstrating that structured template reuse outperforms unstructured reasoning.

## Quick Start

```bash
git clone https://github.com/<org>/nips-templatebank.git
cd nips-templatebank
bash setup.sh
bash scripts/run_all_experiments.sh
```

## Hardware Requirements

| Resource | Specification |
|----------|--------------|
| GPUs | 4–8× NVIDIA A100 80GB (auto-detected) |
| RAM | ≥ 128 GB |
| Disk | ≥ 300 GB (template bank + compiler checkpoints) |
| CUDA | ≥ 12.1 |

GPU count is automatically detected via `scripts/gpu_utils.sh`. The pipeline adapts batch sizes and parallelism accordingly.

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
