# Query Pack (auto-generated, max 8000 chars)

**Generated**: 2026-04-09 | **Papers**: 11 | **Ideas**: 0 | **Gaps**: 8 unresolved

## Project Direction (300 chars)
Self-evolving verified abstraction libraries for compositional math reasoning. Mine verified subroutines from teacher LLM, evolve via RLVR, transfer to smaller models. Target: NeurIPS 2026 best paper. Must differentiate from ReasonFlux (Spotlight) and Agentic Proposing (91.6% AIME25).

## Top 5 Gaps (ranked by impact × opportunity)
1. **G3**: RLVR + library learning are two separate hot threads with ZERO intersection. GRPO exists, library learning exists, nobody connected them. [6 linked papers, 0 ideas]
2. **G5**: "Does RL expand reasoning?" only studied for flat CoT. Compositional/library settings COMPLETELY untested. NeurIPS 2025 Runner-Up paper left this open. [3 linked papers, 0 ideas]
3. **G1**: No self-evolving verified library for math. All existing (ReasonFlux, LILO) use frozen/fixed libraries. [2 linked papers, 0 ideas]
4. **G7**: MCD evaluation absent from ALL template/library reasoning papers. ReasonFlux, Buffer of Thoughts, SELF-DISCOVER — none control for compositional generalization. [1 linked paper, 0 ideas]
5. **G8**: Agentic Proposing does compositional skills but NO step-level execution verification. Skills are unverified black boxes. [1 linked paper, 0 ideas]

## Paper Clusters

**Cluster A: Template/Library Reasoning** (yang2025_reasonflux, zhang2026_agentic_proposing, lilo2024_library_learning)
Hand-designed templates (ReasonFlux) or compositional skills (Agentic Proposing) or program compression (LILO). All use FROZEN libraries. None verify execution. None use MCD.

**Cluster B: RLVR & Reasoning Expansion** (deepseek2025_r1, rlvr_incentivizes_2025, neurips2025_rl_reasoning_debate)
GRPO/RLVR shown effective for flat CoT. Fundamental debate: expand vs optimize. Both sides only study FLAT reasoning — compositional settings untouched.

**Cluster C: Test-Time Compute** (t3rl2026_tool_verification, ttrl2025_test_time_rl, guan2025_rstar_math)
MCTS/TTRL/T3RL push reasoning to test time. T3RL adds tool verification. None incorporate library structure or abstraction building.

**Cluster D: Compositional Theory** (du2025_compositional_energy, theorycoder2_2026)
Energy-based composition (NeurIPS Spotlight) and auto-abstraction learning validated on toy domains. Neither applied to math reasoning.

## Failed Ideas
*(none yet — highest anti-repetition value section)*

## Active Chains (limitation → opportunity)
1. ReasonFlux frozen templates → SEVAL self-evolving library (G1)
2. RLVR flat-only debate → Library-RLVR settles it for compositional (G3+G5)
3. Agentic Proposing unverified skills → Verified compositional execution (G8)
4. T3RL verification without building → Test-time tool building (G4)
5. LILO toy domains → Real math reasoning with our DSL (G1)

## Open Unknowns
- Can RLVR evolve a library without collapsing to trivial abstractions?
- Does energy-based composition scale to real math problems?
- Is the compositional RL-expansion question answerable with current compute?
- Will MCD-hard remain meaningful when models get 95%+ on GSM8K standard?
