# Idea Discovery Report

**Direction**: Subroutine Composition for Mathematical Reasoning (nips-templatebank)
**Date**: 2026-04-07
**Pipeline**: research-lit -> idea-creator -> novelty-check -> research-review -> research-refine-pipeline

## Executive Summary
Best direction identified: **Verified Procedural Abstractions Enable Transferable Compositional Math Reasoning**. A compression-mined, step-verified subroutine library extracted from Qwen3.5-32B is frozen and transferred to smaller models (9B, 3B), achieving large gains on rigorously controlled MCD compositional splits. MDL compression ratio serves as a diagnostic predictor of compositional transfer. MCTS search-time repair recovers failed plans. Three headline claims, all novelty-confirmed, with a clear 5-week experiment plan.

## Literature Landscape

### Tier 1: Direct Competitors (Template/Library-based Reasoning)
| Paper | Venue | Mechanism | Result |
|-------|-------|-----------|--------|
| ReasonFlux | NeurIPS 2025 Spotlight | ~500 hand-designed templates + hierarchical RL | 91.2% MATH |
| Buffer of Thoughts | NeurIPS 2024 Spotlight | Meta-buffer + template retrieval | +11-51% on hard reasoning |
| SELF-DISCOVER | 2024 | Self-compose atomic reasoning modules | +32% over CoT |
| Retrieval-of-Thought | Sep 2025 | Thought graph + composable steps | 40% token reduction |

### Tier 2: Compositional Generalization Theory
| Paper | Venue | Finding |
|-------|-------|---------|
| Compositional Gen. from Learned Skills via CoT | Feb 2025 | CoT enables 2-stage compositional circuit |
| Learning Composable Chains-of-Thought | May 2025 | Composable CoT format for atomic skills |
| MCD splits (Keysers et al.) | ICLR 2020 | Max compound divergence evaluation |

### Tier 3: Program Synthesis / DSL / Library Learning
| Paper | Venue | Mechanism |
|-------|-------|-----------|
| LILO | ICLR 2024 | LLM + Stitch compression, toy domains |
| DreamCoder/Stitch | PLDI 2021 | Wake-sleep library learning |
| MathDSL | NeurIPS 2025 workshop | DSL axioms for equation solving |
| RV-Syn | EACL 2026 | Structured function library for data synthesis |
| Proof of Thought | NeurIPS 2024 workshop | Neurosymbolic reasoning + Z3 |

### Tier 4: Code-based Math Reasoning
| Paper | Venue | Mechanism |
|-------|-------|-----------|
| SBSC | ICLR 2025 | Multi-turn step-by-step code |
| rStar-Math | ICML 2025 | MCTS + self-play, 90% MATH |
| AgentMath | Dec 2025 | Tool-augmented agent + agentic RL |
| SymCode | EACL 2026 | SymPy-based verifiable code gen |

### Structural Gaps
1. No compression-based library learning for math reasoning (LILO only toy domains)
2. No MCD-split evaluation in any template/library reasoning paper
3. No reusable subroutine libraries for reasoning (RV-Syn only for data synthesis)
4. Theory proves CoT enables compositional gen but no practical system exists
5. No portable frozen reasoning libraries transferred across model sizes

## Ranked Ideas

### Idea 1: Program Distillation With Verified Library Transfer — SELECTED
- **Thesis**: Frozen 32B-mined verified subroutine library enables 9B to achieve 3x compositional accuracy on MCD-hard splits
- **Novelty**: CONFIRMED — no prior work freezes mined typed subroutine library for cross-model transfer
- **Reviewer score**: 4/10 as initial proposal, path to 7/10 and 9/10 defined
- **Feasibility**: ~560 GPUh, builds on existing codebase

### Idea 2: Difficulty-Scaled Compression Diagnostic — MERGED INTO #1
- **Thesis**: MDL compression ratio of verified traces is the strongest diagnostic predictor of MCD compositional generalization
- **Novelty**: CONFIRMED — no prior connection between trace compressibility and compositional gen prediction
- **Merged as**: Claim 2 of the combined proposal

### Idea 3: Library-First Inference + Search-Time Repair — MERGED INTO #1
- **Thesis**: MCTS over typed compositions with execution-guided repair recovers >=25% of failed plans
- **Novelty**: CONFIRMED — no prior typed subroutine repair for math reasoning
- **Merged as**: Claim 3 of the combined proposal

## Eliminated Ideas
None — all three ideas were merged into a unified proposal.

## Combined Proposal: Three Claims
1. **Portable Verified Library**: compose + frozen 32B library beats CoT-distilled 9B by >=15pts on MCD-hard
2. **Compression as Diagnostic**: MDL ratio outpredicts library size, trace length, teacher accuracy (p<0.05)
3. **Search-Time Repair**: typed MCTS repair recovers >=25% of failed plans under matched budget

## Reviewer Feedback Integrated
- Narrowed thesis from "reasoning abstractions are transferable" to domain-specific claim
- Added 10 baselines including 32B-CoT distillation, raw trace retrieval, uncompressed bank
- Step-level verification protocol (not just final answer)
- Formal atom/compound definitions with overlap audits
- Library quality audit with coherence threshold
- Causal compression test: compressed vs matched-size uncompressed
- Anti-cheating guarantees for search
- Failure analysis: 150+ labeled failures in 8 categories
- Second student model for portability
- Stop/go gates at each week

## Refined Proposal
- Proposal: `refine-logs/FINAL_PROPOSAL.md`
- Experiment plan: `refine-logs/EXPERIMENT_PLAN.md`

## GPU Budget
- Total planned: 1950 GPUh
- Contingency: 330 GPUh
- Hardware: 4x H800 80GB (5 weeks = 3360 GPUh ceiling)

## 5-Week Timeline
| Week | Focus | Gate |
|------|-------|------|
| 1 | Teacher extraction + step verification | >=1000 verified traces |
| 2 | Library mining + MCD split | TVD<=0.02, unseen>=0.40 |
| 3 | Student training (compose + CoT-distilled) | valid-plan>=45% |
| 4 | Main eval + portability + compression sweep | compose beats CoT-distill by >=15 |
| 5 | Search repair + audit + MATH stress + paper | repair >=25% |

## Next Steps
- [ ] Implement step-level verification in extract_templates.py
- [ ] Run Block 1 smoke test
- [ ] Run Block 2 GSM8K teacher extraction
- [ ] Or invoke /experiment-bridge to start implementation
