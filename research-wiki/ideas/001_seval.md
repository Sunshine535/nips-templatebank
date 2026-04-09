---
type: idea
node_id: idea:001_seval
title: "Self-Evolving Verified Abstraction Libraries with Test-Time Tool Building"
stage: proposed
outcome: null
created_at: 2026-04-09T00:00:00Z
updated_at: 2026-04-09T00:00:00Z
target_gaps: [G1, G3, G4, G5, G7, G8]
based_on: [paper:deepseek2025_r1, paper:t3rl2026_tool_verification, paper:lilo2024_library_learning, paper:du2025_compositional_energy, paper:yang2025_reasonflux, paper:rlvr_incentivizes_2025, paper:neurips2025_rl_reasoning_debate, paper:theorycoder2_2026]
---

# Self-Evolving Verified Abstraction Libraries with Test-Time Tool Building (SEVAL)

## One-Line Thesis
A verified executable subroutine library that evolves through RLVR composition rewards and grows new tools at test time provides the first evidence that reinforcement learning expands — not merely optimizes — compositional mathematical reasoning.

## Problem Anchor
Question: Can a verified executable library of typed DSL subroutines, mined from a teacher LLM and evolved via RLVR, create a self-reinforcing "abstraction flywheel" where new verified compositions become new tools, expanding the compositional reasoning frontier beyond what frozen libraries, unverified skills, or flat CoT distillation achieve?

## Why This Is Revolutionary
1. **Connects two major disconnected threads** (RLVR + library learning) for the first time
2. **Answers the hottest open question** (does RL expand reasoning?) for compositional settings — the NeurIPS 2025 Runner-Up left this open
3. **First self-evolving VERIFIED tool library** — beyond fixed templates (ReasonFlux), unverified skills (Agentic Proposing), and NL hints (RLAD)
4. **First test-time tool building** — creates NEW tools during inference, not just verifies existing ones
5. **Strong theoretical spine**: MDL compression → RLVR evolution → energy-based composition → test-time growth

## Method

### Phase 0: Seed Library Mining (existing infrastructure)
- Teacher (Qwen3.5-27B) generates typed DSL programs for MATH
- Step-level verification: parse → type-check → execute → answer-check → perturbation-robustness
- MDL-based subroutine extraction → initial frozen library L₀ (K=16 subroutines)
- MCD split construction for compositional evaluation

### Phase 1: RLVR-Driven Library Evolution (NEW — addresses G1, G3, G5)
- Student (Qwen3.5-9B) learns to compose library functions via GRPO
- Verifiable reward = binary composition execution success (no reward model needed)
- Evolution protocol (every N GRPO steps):
  1. Collect successful novel composition patterns
  2. If pattern appears ≥K times with ≥P% success AND MDL_gain > threshold:
     → Abstract into new verified subroutine
     → Verify on held-out examples
     → Add to library: L₀ → L₁ → L₂ → ...
- "Abstraction flywheel": more tools → richer compositions → more abstraction candidates → more tools
- Track CoT-Pass@K throughout to measure capability EXPANSION (not just optimization)
- Anti-collapse safeguards: minimum library diversity, MDL gain floor, redundancy check

### Phase 2: Test-Time Tool Building (NEW — addresses G4)
- At inference, when top-K compositions all fail verification:
  1. Analyze failure patterns (missing operation type, wrong composition order)
  2. Generate candidate new tool from failed attempt fragments
  3. Verify candidate via execution on simple synthetic cases
  4. If verified: add to per-problem tool cache
  5. Re-attempt composition with expanded tool set
- Budget-controlled: max M=3 new tools, max T=10 verification attempts per problem
- T3RL-style verification weighting for candidate ranking

### Phase 3: Transfer & Generalization (enhanced from original)
- Freeze evolved library L_final
- Transfer to Student B (Qwen3.5-4B) — NO student-specific reminting
- Evaluate on MCD-hard splits (compositional generalization)
- Evaluate on AIME 2024/2025 subset (competitive math, stretch goal)

## Four Falsifiable Claims

### Claim C1: Library Evolution > Frozen Library
RLVR-evolved library L_final outperforms frozen L₀ by ≥10 accuracy points on MATH MCD-hard (Qwen3.5-9B), with 95% bootstrap CI excluding 0.

### Claim C2: RLVR Expands Compositional Reasoning
In library-composition settings, RLVR-trained student achieves higher CoT-Pass@K than base model at K=64, providing first evidence that RL expands (not just optimizes) compositional reasoning. The gap must be positive for at least 2/3 training seeds.

### Claim C3: Test-Time Tool Building Recovers Failures
Per-problem tool building at inference recovers ≥20% of initially failed compositions under matched compute budget (equalized forward passes + executor calls). Must beat test-time search without tool building by ≥5 pts.

### Claim C4: Evolved Library Transfers Cross-Model
Frozen evolved library L_final, when transferred to Qwen3.5-4B without reminting, improves over CoT-distilled 3B baseline by ≥8 pts on MATH MCD-hard.

## Key Differentiators from Competitors

| Feature | ReasonFlux | Agentic Prop. | RLAD | rStar-Math | SEVAL (ours) |
|---------|-----------|---------------|------|------------|-------------|
| Abstraction type | NL templates | Unverified skills | NL hints | None | **Verified DSL** |
| Self-evolving | No (frozen) | No | No | Self-play | **RLVR-driven** |
| Execution verified | No | No | No | Code-augmented | **Step-level** |
| Library growth | No | No | No | No | **RLVR evolution** |
| Test-time growth | No | No | No | MCTS search | **Tool building** |
| Cross-model transfer | No | No | No | No | **Yes** |
| MCD evaluation | No | No | No | No | **Yes** |
| RL expansion evidence | N/A | N/A | Partial | N/A | **CoT-Pass@K** |

## GPU Budget Estimate
| Phase | GPUh | Critical Path? |
|-------|------|---------------|
| 0: Seed mining (MATH) | 200 | Yes |
| 1: RLVR evolution (3 rounds × 3 seeds) | 500 | Yes |
| 2: Test-time eval | 200 | Yes |
| 3: Transfer + baselines | 300 | Yes |
| Ablations (library size, evolution rounds) | 200 | No |
| Contingency | 200 | - |
| **Total** | **1600** | |

## Risk Matrix
| Risk | Prob | Impact | Mitigation |
|------|------|--------|------------|
| Library collapse to trivial abstractions | Medium | Critical | MDL gain floor, diversity constraint |
| RLVR doesn't show expansion at K=64 | Medium | High | Reframe to "more efficient" rather than "expanded" |
| Test-time tool building too slow | Medium | Medium | Reduce budget, keep as analysis only |
| Teacher extraction too few verified | Low | Critical | Increase samples, fallback to GSM8K |
| MATH too hard for 9B student | Medium | High | Focus on Level 1-3, keep 4-5 as stress test |

## Failure Notes
*(to be filled after experiments)*
