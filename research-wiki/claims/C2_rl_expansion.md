---
type: claim
node_id: claim:C2
title: "RLVR expands compositional reasoning beyond base model"
status: proposed
testable: true
created_at: 2026-04-09T00:00:00Z
---

# C2: RLVR expands (not just optimizes) compositional reasoning

**Formal statement**: In library-composition settings, RLVR-evolved Qwen3.5-9B (GRPO-trained with composition-execution rewards) achieves strictly higher CoT-Pass@64 than an SFT-trained Qwen3.5-9B (supervised on the same composition data, matched compute budget) on MATH MCD-hard, for at least 2/3 training seeds. This provides first evidence that RL expands compositional reasoning capability beyond what supervised learning achieves.

**Critical comparison**: RLVR-evolved vs **SFT-trained** (NOT vs raw base model). Comparing against base model is trivially true and scientifically meaningless — any fine-tuned model beats a raw base at pass@1. The NeurIPS 2025 Runner-Up showed base models may match RL at large pass@K; we test whether this holds in compositional settings.

**Three-way comparison needed**:
1. Raw base model (lower bound)
2. SFT-trained (matched data + compute)
3. RLVR-evolved (our method)

If RLVR > SFT at pass@64 → expansion evidence
If RLVR ≈ SFT at pass@64 but RLVR > SFT at pass@1 → optimization only

**Evidence needed**: CoT-Pass@K curves for K=1,4,16,64 comparing all three.

**If fails**: Report as "RLVR improves efficiency but not frontier" — still publishable as it extends the debate to compositional settings with a definitive answer.
