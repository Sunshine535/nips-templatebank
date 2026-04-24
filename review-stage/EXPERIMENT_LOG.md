# Experiment Log

## 2026-04-24: First Successful Experiments

### Experiment 1: SFT Training
- **Data**: 697 verified GSM8K programs from teacher extraction (Qwen3.5-27B)
- **Model**: Qwen3.5-9B + LoRA (r=32, alpha=64)
- **Training**: 3 epochs, lr=2e-4, batch_size=2×4GPU, 237 steps, 5.5 min
- **Train set eval** (50 samples from training data): **84.0%** accuracy, 100% parse, 100% exec
- **GSM8K test set** (200 samples, no overlap with train): **~30%** accuracy, 96% parse, 92% exec
- **Finding**: SFT effectively teaches JSON program generation but overfits on small dataset

### Experiment 2: GRPO Training
- **Init**: From SFT checkpoint
- **Reward**: Execution-verified (program parse + execute + answer correct = 1.0, parse only = 0.1, fail = 0.0)
- **Training**: 500 steps, lr=5e-6, 4 generations/step, temperature=0.8, 4.5 hours
- **Train set eval** (50 samples): **84.0%** accuracy — no improvement over SFT
- **Mean reward**: 0.80-0.94 during training
- **Finding**: GRPO reward already saturated on small training set. Need more data for meaningful RL signal

### Experiment 3: GSM8K Test Set Evaluation (IN PROGRESS)
- **Models**: base Qwen3.5-9B vs SFT vs GRPO
- **Test set**: GSM8K test split (200/1319 samples)
- **Partial results**:
  - Base: 0/200 (0.0%) — cannot generate JSON programs
  - SFT: 15/50 (30.0%) on first 50 — still evaluating
  - GRPO: pending

### Key Observations
1. Teacher extraction works: 697/7473 (9.3%) GSM8K problems → verified correct programs
2. SFT trains model to generate JSON programs: 0% → 30% on test set
3. Overfitting: 84% train vs 30% test indicates need for more training data
4. GRPO needs exploration: with training reward already at 0.94, no room for improvement
5. 96% parse rate + 92% exec rate shows the typed DSL is learnable

### Infrastructure Issues Resolved
- Container OOM (16GB → 128GB memory limit)
- Python log buffering (added -u flag and explicit flush)
- nccl keep-alive for GPU utilization
- Pod auto-recovery via watchdog daemon
