# Self-Evolving Verified Abstraction Libraries: When Library Learning Meets RLVR for Compositional Mathematical Reasoning

## Problem Anchor
Can a verified executable library of typed DSL subroutines, mined from a teacher LLM and evolved via RLVR, create a self-reinforcing "abstraction flywheel" where new verified compositions become new tools—expanding the compositional reasoning frontier beyond what frozen libraries, unverified skills, or flat CoT distillation achieve?

## Falsifiable Thesis (4 Claims)
1. **C1 — Library Evolution**: RLVR-evolved library L_final outperforms frozen L₀ by ≥10 accuracy points on MATH MCD-hard (Qwen3.5-9B), 95% bootstrap CI excluding 0.
2. **C2 — RLVR Expands Compositional Reasoning**: CoT-Pass@64 of RLVR-evolved student exceeds **SFT-trained student** (matched data + compute) on MATH MCD-hard, providing first evidence that RL expands (not just optimizes) compositional reasoning. Three-way comparison: base vs SFT vs RLVR. Positive for ≥2/3 seeds.
3. **C3 — Test-Time Tool Building**: Per-problem tool building recovers ≥20% of initially failed compositions under matched compute budget.
4. **C4 — Cross-Model Transfer**: Frozen evolved library transferred to Qwen3.5-3B (no reminting) beats CoT-distilled 3B baseline by ≥8 pts on MATH MCD-hard.

Not claimed: universal reasoning; a "compression law"; search with privileged labels.

## Method

### Core Objects (existing, in src/template_dsl.py)
- `Program(program_id, slots, steps)`: typed DSL steps
- `Subroutine(sub_id, program, support, mdl_gain)`: verified library entry
- `SubroutineLibrary`: collection with evolution API (`mint_subroutine`, `evolve`)
- `CompositionPlan(calls)`: ordered subroutine calls with bindings
- `CompositionExecutor`: executes plans over the library
- `Executor`: deterministic step-by-step execution with type checking

### Phase 0: Seed Library Mining
Same as v1: Qwen3.5-32B generates DSL programs → step-level verification → MDL-based subroutine extraction → frozen initial library L₀ (K=16).

### Phase 1: RLVR-Driven Library Evolution (NEW — src/rlvr_evolution.py)
1. Student (Qwen3.5-9B) generates composition plans via GRPO
2. Verifiable reward = `CompositionExecutor` execution success (binary, no reward model)
3. Every `evolution_interval` (200) GRPO steps:
   a. Collect all successful composition patterns from buffer
   b. Find recurring subroutine bigrams (count ≥5, success rate ≥70%)
   c. Estimate MDL gain for each pattern
   d. Abstract qualifying patterns into new subroutine programs
   e. Verify candidates on held-out examples (≥80% execution success)
   f. Add verified subroutines to library: L₀ → L₁ → L₂ → ...
4. Anti-collapse safeguards: MDL gain floor, diversity check, max library size cap
5. Track CoT-Pass@K throughout training for capability expansion measurement

### Phase 2: Test-Time Tool Building (NEW — src/test_time_tools.py)
When top-K compositions fail at inference:
1. Analyze failure patterns (missing sub, type mismatch, execution error, wrong answer)
2. Generate candidate tools via 4 strategies:
   - Combine partial results from different failed attempts
   - Bridge from available values to target
   - Create type adapters for mismatches
   - Model-generated tools (if planner model available)
3. Verify candidates via execution (deterministic, perturbation-robust)
4. Add verified tools to per-problem cache
5. Re-attempt composition with expanded tool set
Budget: max 3 new tools, max 10 verification attempts per problem.

### Phase 3: Transfer & Generalization
- Freeze evolved library L_final
- Transfer to Student B (Qwen3.5-3B) — no reminting
- Evaluate on MCD splits with full protocol
- Compare: evolved vs frozen vs CoT-distilled

## Formal MCD Split Construction
Same as v1: 3-layer validation (structural, distributional, leakage), TVD ≤ 0.02, unseen compounds ≥ 0.40.

## Verification Protocol
Same as v1: step-level (parse → type → execute → answer → perturbation).

## Baselines (12 methods)
1. Frozen L₀ compose (our Phase 0 output)
2. CoT-distilled 9B (primary comparison for C1)
3. CoT-distilled 3B (primary comparison for C4)
4. flat_inline
5. raw_trace_retrieval
6. uncompressed_program_bank
7. random_library
8. frequency_matched_library
9. retrieval_compose
10. cot_budget (majority vote)
11. MCTS search (without tool building, for C3 budget match)
12. base model (untuned, for C2 expansion measurement)

## Evaluation Metrics
- **Accuracy** (answer correctness on MCD-hard/medium/random)
- **CoT-Pass@K** (K=1,4,16,64) — capability expansion measurement
- **Valid plan rate** — fraction of parseable, executable plans
- **Recovery rate** — fraction of failures recovered by tool building
- **MDL compression ratio** — library evolution quality tracking
- **Library size over training** — evolution dynamics

## 8-Page Paper Outline
1. **Introduction** (0.75p): Library learning meets RLVR; abstraction flywheel; 4 contributions
2. **Background** (0.5p): DSL programs, composition plans, MCD splits, GRPO
3. **SEVAL Method** (1.5p): Phase 0 mining, Phase 1 RLVR evolution, Phase 2 test-time building
4. **MCD Split & Verification** (0.75p): Formal construction, step-level verification
5. **Experiments** (1.5p): Setup, baselines, evaluation protocol
6. **Results** (1.5p): C1-C4, evolution dynamics, CoT-Pass@K curves, ablations
7. **Analysis** (0.75p): What evolves? Library audit, failure analysis, when building helps
8. **Related Work & Conclusion** (0.75p): ReasonFlux, RLAD, T3RL, LILO differentiation

## GPU Budget
| Phase | GPUh | Critical Path? |
|-------|------|---------------|
| 0: MATH teacher extraction | 260 | Yes |
| 0: Library mining + MCD split | 18 | Yes |
| 1: GRPO evolution (3 seeds) | 500 | Yes |
| 2: Test-time eval | 200 | No |
| 3: Transfer (3B, 3 seeds) | 200 | No |
| Baselines (12 methods) | 200 | Yes |
| CoT-Pass@K evaluation | 100 | Yes |
| Ablations (library size, rounds) | 150 | No |
| Contingency | 200 | - |
| **Total** | **1828** | |

## Risk Matrix
| Risk | Prob | Impact | Mitigation |
|------|------|--------|------------|
| Library collapse | Medium | Critical | MDL gain floor, diversity constraint, max size cap |
| RLVR no expansion at K=64 | Medium | High | Reframe to "more efficient" (better pass@1) |
| Test-time building too slow | Medium | Medium | Reduce to analysis-only, remove from claims |
| Teacher extraction <1000 | Low | Critical | Increase samples, fallback to GSM8K |
| MATH too hard for 9B | Medium | High | Focus Level 1-3, keep 4-5 as stress |
| GRPO training unstable | Low | Medium | Reduce LR, increase warmup |
