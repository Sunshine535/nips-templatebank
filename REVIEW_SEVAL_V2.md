# SEVAL V2 Review — Nightmare Difficulty (Internal)

**Reviewer**: Claude Opus 4.6 (self-adversarial, no external MCP available)
**Date**: 2026-04-09
**Target**: NeurIPS 2026 Best Paper
**Status**: Pre-experiment review of proposal + code

---

## Mock NeurIPS Review

### Summary
This paper proposes SEVAL, a system that mines verified typed DSL subroutines from a teacher LLM, then evolves the subroutine library via GRPO (verifiable execution rewards), and builds new tools at test time. Four claims: library evolution > frozen (C1), RLVR expands compositional reasoning (C2), test-time tools recover failures (C3), cross-model transfer (C4). Evaluated on MATH with MCD compositional splits.

### Strengths
1. **Novel combination**: First paper to connect RLVR (GRPO) with library learning. This is a genuine gap (G3 in their gap map) — RLVR papers ignore abstraction structure, library learning papers ignore RL.
2. **Addresses hot open question**: The "does RL expand reasoning?" debate (NeurIPS 2025 Runner-Up) is only studied for flat CoT. Testing in compositional settings is novel and timely.
3. **Strong evaluation design**: MCD splits with 3-layer validation, 12 baselines, 3 seeds, CoT-Pass@K metric — substantially more rigorous than ReasonFlux or Agentic Proposing.
4. **Test-time tool building is creative**: Going beyond T3RL (verification) to actual tool CONSTRUCTION at inference is a novel concept.
5. **Codebase quality**: 6500+ LOC, tested DSL execution, production-ready scripts.

### Weaknesses (CRITICAL — would block acceptance)

**W1: The "abstraction flywheel" may be theoretically vacuous.**
The claimed flywheel (new compositions → new tools → richer compositions) is appealing but the actual mechanism in `rlvr_evolution.py:LibraryEvolver.find_patterns()` only looks at subroutine bigrams. A bigram like (L00, L01) being common doesn't mean the sequence is a meaningful new operation — it could just mean those are the two most popular subroutines. The abstraction step (`abstract_pattern`) literally concatenates two subroutine programs with variable renaming. This is the SAME thing as `inline_program()` in `template_dsl.py`. **The "evolved" subroutine is just a pre-concatenated version of what the planner was already doing.** This provides a constant-factor speedup (one call vs two), not a new capability. For the flywheel to actually create new capability, the abstracted subroutine must represent something the planner COULDN'T compose before — but since it's derived from successful compositions, by definition it already could.

**FIX NEEDED**: The evolution mechanism needs to produce genuinely new abstractions — e.g., parameterized generalizations of bigrams (where constants become slots), not just concatenations. Or: evolved subroutines should demonstrate capability on problems that the original library CANNOT solve (not just compress existing solutions).

**W2: CoT-Pass@K for "capability expansion" is confounded.**
Claim C2 says: if CoT-Pass@64 of RLVR-evolved student > base model, then RL "expands" reasoning. But this conflates two things:
- The student was trained (GRPO fine-tuned) — of course it generates better plans than the untrained base model
- The comparison should be RLVR-evolved vs SFT-trained (same data, same compute) — not vs untrained base

The NeurIPS 2025 Runner-Up paper compared RL vs base at large pass@K specifically because RL is known to improve pass@1. If you compare a FINE-TUNED model against a RAW base model, you're measuring training effect, not RL expansion. **This renders C2 as stated trivially true and scientifically meaningless.**

**FIX NEEDED**: C2 must compare RLVR-evolved against SFT-trained (same data budget, same compute). If RLVR-evolved still has higher pass@64 than SFT-trained, THAT would be evidence of expansion. Even better: show that at large K, SFT catches up (as in the debate paper) but RLVR-evolved does not — this would be strong evidence.

**W3: Test-time tool building strategies are heuristic and weak.**
Looking at `test_time_tools.py`, the 4 tool generation strategies are:
1. `_combine_partial_results`: tries sum/product/difference of available values — this is random guessing
2. `_create_bridge_tool`: multiplies base_value × factor — even more random
3. `_create_type_adapter`: converts types — trivial, doesn't help with reasoning
4. `_model_generate_tools`: asks the model to write a new subroutine — this is the only non-trivial strategy, but it's basically just program synthesis (not library learning)

**The first 3 strategies are hacks, not principled tool building.** For a best-paper-level contribution, test-time tool building should be more principled — e.g., identifying WHICH operation is missing from the library via error analysis, then synthesizing specifically THAT operation.

**FIX NEEDED**: Replace heuristic strategies with systematic gap detection: analyze the failure to identify what TYPE of computation is needed (e.g., "need a quadratic formula subroutine"), then either retrieve from a larger bank or synthesize specifically that.

**W4: No competitive benchmarking.**
ReasonFlux: 91.2% MATH. Agentic Proposing: 91.6% AIME25. rStar-Math: 90% MATH. The proposal doesn't target these benchmarks head-to-head. MCD-hard is a controlled split (good for scientific claims) but reviewers WILL ask "what's your overall MATH accuracy?" If it's <50% (likely for a 9B model with a small library), the paper looks weak regardless of how rigorous the evaluation is.

**FIX NEEDED**: Include standard benchmark comparisons (full MATH test set, AIME subset). Even if the absolute numbers are lower (smaller model, constrained method), show the gap to baselines is meaningful. Or explicitly argue why MCD-hard accuracy is more meaningful than full MATH accuracy (compositional generalization vs memorization).

**W5: The DSL is too restrictive for MATH.**
The current DSL (`template_dsl.py`) supports: assign, compute, compare, aggregate, condition, output. Expressions are Python eval. But MATH includes geometry (need diagrams/spatial reasoning), number theory (need modular arithmetic), combinatorics (need factorial/binomial), and precalculus (need trig). **Can the DSL actually express solutions to these problem types?** If the verified extraction rate on MATH is <20% (which is the target), that means 80% of MATH is outside the DSL's reach. The library can only evolve on the 20% it can handle, which severely limits generality.

**FIX NEEDED**: Expand SAFE_BUILTINS in the Executor to include: math.factorial, math.comb, math.gcd, modular arithmetic, trig functions. Report per-category extraction rates (algebra vs geometry vs number theory). Acknowledge the DSL's limitations explicitly.

### Weaknesses (MINOR — addressable)

**W6**: The `min_pattern_count=5` threshold seems arbitrary. With only ~500 training examples, bigrams that appear 5 times may be coincidental.

**W7**: Evolution interval of 200 GRPO steps — no justification. Too frequent = noisy; too infrequent = miss patterns.

**W8**: `answer_tolerance=1e-3` may be wrong for MATH — many answers are exact integers.

**W9**: No analysis of what types of subroutines the library evolves into. The audit plan exists but is Block 10 (last).

**W10**: The paper outline gives 1.5 pages to "SEVAL Method" which is too compressed for 3 novel phases.

### Questions for Authors

Q1: Can you show an example of a non-trivial subroutine that would be EVOLVED (not just mined) by the flywheel? Walk through the full pipeline: initial library → GRPO training → pattern detected → abstracted → verified → used to solve a problem the initial library couldn't.

Q2: For C2, what happens if you compare RLVR-evolved vs SFT-trained (matched data/compute) at pass@64? If both are equal, C2 falls.

Q3: What is the projected extraction rate on MATH? If <20%, how many of the 5000 test problems can the system even attempt?

Q4: The test-time tool building generates tools via `x + y`, `x * y`, `x - y` (line 235). How is this different from a calculator?

### Scores

| Criterion | Score | Comment |
|-----------|-------|---------|
| Novelty | 7/10 | RLVR + library learning is genuinely new. Test-time building is novel but under-developed |
| Rigor | 5/10 | C2 is confounded as stated. Flywheel mechanism is questionable |
| Significance | 6/10 | Answers a hot question (RL expansion) but may not achieve competitive accuracy |
| Clarity | 7/10 | Good structure, but method density is high |
| Reproducibility | 8/10 | Strong codebase, configs, detailed plan |
| **Overall** | **5/10** | **Borderline reject** — fix W1-W4 to reach accept territory |
| Confidence | 4/5 | Familiar with all cited work |

### What Would Move Toward Accept

1. **Fix C2 comparison** (RLVR vs SFT, not vs base) — this alone is worth 1.5 points
2. **Make evolution non-trivial** (parameterized generalization, not concatenation) — worth 1 point
3. **Show at least one category of MATH where evolved library beats frozen by ≥15 pts** — worth 0.5 points
4. **Replace heuristic test-time strategies with principled gap detection** — worth 0.5 points
5. **Include full MATH benchmark numbers** (even if lower than SOTA) — worth 0.5 points
6. **Expand DSL to cover more MATH categories** — prerequisite for credibility

With fixes 1-4 implemented: estimated score **7/10 (weak accept)**.
With fixes 1-6 and strong results: estimated score **8/10 (accept)**.
Best paper requires a clear "this changes how the field thinks" result on C2 — proving RL expansion in compositional settings would do it.

---

## Prioritized Fix List

| Priority | Fix | Effort | Impact |
|----------|-----|--------|--------|
| P0 | Fix C2: compare RLVR vs SFT (matched) | Script change | +1.5 pts |
| P0 | Fix evolution: parameterized generalization | Core algorithm | +1.0 pts |
| P1 | Expand DSL builtins for MATH | Small code change | Prerequisite |
| P1 | Add full MATH benchmark eval | Script addition | +0.5 pts |
| P1 | Replace heuristic test-time strategies | Moderate rewrite | +0.5 pts |
| P2 | Per-category analysis (algebra/geometry/etc) | Eval extension | +0.3 pts |
| P2 | Justify evolution hyperparameters | Ablation | +0.2 pts |
| P2 | Answer tolerance = exact for integers | Config fix | Bug fix |

## Claims Matrix (Possible Outcomes)

| C1 (evolution>frozen) | C2 (RL expands) | C3 (test-time) | C4 (transfer) | Paper status |
|------------------------|------------------|-----------------|----------------|-------------|
| ≥10 pts ✓ | Expansion ✓ | ≥20% ✓ | ≥8 pts ✓ | **Strong accept** |
| ≥10 pts ✓ | Expansion ✓ | <20% ✗ | ≥8 pts ✓ | Accept (demote C3 to analysis) |
| ≥10 pts ✓ | Optimization only | ≥20% ✓ | ≥8 pts ✓ | Accept (reframe C2 to "efficiency") |
| ≥5 pts ✓ | Expansion ✓ | Any | Any | Weak accept (C2 carries the paper) |
| ≥10 pts ✓ | Any | Any | <8 pts ✗ | Weak accept (narrow scope) |
| <5 pts ✗ | Any | Any | Any | **Reject** (core method doesn't work) |
| Any | Any | Any | All fail | **Reject** |

**C2 is the make-or-break claim.** If C2 holds (properly measured), the paper is publishable regardless of C3/C4 outcomes. If C2 fails, C1 alone is incremental over frozen library work.
