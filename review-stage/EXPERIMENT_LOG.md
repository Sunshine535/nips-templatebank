# Experiment Log

## 2026-04-24: First Successful Experiments

### Experiment 1: SFT Training
- **Data**: 697 verified GSM8K programs from teacher extraction (Qwen3.5-27B)
- **Model**: Qwen3.5-9B + LoRA (r=32, alpha=64)
- **Training**: 3 epochs, lr=2e-4, batch_size=2×4GPU, 237 steps, 5.5 min
- **Final train loss**: 0.060, token accuracy: 98.3%

### Experiment 2: GRPO Training
- **Init**: From SFT checkpoint
- **Reward**: Execution-verified (parse+exec+answer=1.0, parse_only=0.1, fail=0.0)
- **Training**: 500 steps, lr=5e-6, 4 generations/step, temp=0.8, 4.5 hours
- **Final mean reward**: 0.80-0.94

### Experiment 3: GSM8K Test Set Evaluation (200 samples)

| Model | Accuracy | Parse Rate | Exec Rate | Notes |
|-------|----------|------------|-----------|-------|
| **Base (Qwen3.5-9B)** | **0.0%** (0/200) | 0.0% | 0.0% | Cannot generate JSON programs |
| **SFT** | **29.5%** (59/200) | 95.0% | 89.0% | Learns program structure well |
| **GRPO** | **~30%** (15/50 partial) | 96% | 92% | Still evaluating, similar to SFT |

### Key Findings
1. Base model 0% → SFT 29.5%: **SFT successfully teaches JSON program generation**
2. 95% parse rate: typed DSL format is highly learnable
3. 89% execution rate: generated programs are structurally valid
4. 29.5% answer accuracy: programs execute but often compute wrong answer
5. Train 84% vs Test 29.5%: **severe overfitting** due to only 697 training programs
6. GRPO shows no improvement over SFT: reward already saturated on small dataset

### Next Steps
- Extract more programs (GSM8K + MATH) for larger training set
- Implement Type-Local GRPO (step-level credit assignment)
- Evaluate on MATH test set
- Compare with baselines (direct CoT, PAL)
