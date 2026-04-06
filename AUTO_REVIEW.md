# ATLAS Auto Review Log

## Round 1 (2026-04-04) — Post-ATLAS Redesign

### Assessment (Summary)
- Score: **2/10**
- Verdict: **NOT READY**
- Key criticisms:
  1. MCTS uses gold answer as reward (cheating)
  2. MCD split has zero unseen compounds (benchmark invalid)
  3. Results too weak (4% on N=50)
  4. Real data artifacts marked as synthetic (contamination)
  5. Programs not verified against gold answer
  6. Composition fallback to generic subroutine (brittle)
  7. Novelty claims overstated relative to evidence

<details>
<summary>Click to expand full reviewer response</summary>

Score: 2/10 for a top venue. The idea is interesting, and some core machinery exists, but the current paper story is not supported by valid or strong enough evidence.

Critical Weaknesses:

1. The MCTS evaluation is not valid as presented. The eval code passes the gold answer into search, and the rollout reward scores closeness to that gold answer. Without gold, the fallback reward just prefers longer plans. Minimum fix: remove answer-conditioned reward entirely, replace it with a label-free search objective, and rerun the full MCTS comparison from scratch.

2. The claimed compositional-generalization benchmark is not actually established. The saved split has zero unseen test compounds, despite the code targeting nontrivial unseen-compound divergence. Minimum fix: produce a real MCD split with substantial unseen compounds and low atom drift, then report the main compose-vs-flat result on that split.

3. The empirical result is far too weak for NeurIPS/ICML. 4% on N=50 is not persuasive, and the "real" setting is 0%. Minimum fix: audit the metric pipeline, regenerate all results from one reproducible command, and show statistically reliable gains on real data over the flat baseline and CoT.

4. The "real-data" pipeline appears contaminated or at least ambiguously tracked. The supposed real extraction artifact still says synthetic_used: true. Minimum fix: remove synthetic fallback from all main experiments, fail hard if real data are unavailable, and regenerate clean artifacts with provenance per sample.

5. The supervision pipeline is not semantically trustworthy enough. Program extraction accepts any executable program without checking it matches the gold answer. Minimum fix: only keep extracted programs whose execution matches the gold answer, add a stronger binding/grounding mechanism, and report extraction correctness separately from execution rate.

6. The composition mechanism is still brittle and partly heuristic. Failed decompositions can fall back to a generic best-supported subroutine. Minimum fix: add constrained decoding and stricter binding semantics, then report oracle decomposition coverage vs learned planner accuracy.

7. The novelty framing is overstated relative to the evidence. "First" claims are not justified by the current results. Minimum fix: either substantially strengthen experiments or reframe as a preliminary systems/prototype study.

Verdict: READY for submission: No. This is not close to NeurIPS/ICML-ready.

</details>

### Actions Taken

1. **Fixed MCTS reward** (weakness #1): Removed gold-answer-conditioned reward entirely. MCTS now uses label-free criteria only: execution success (0.4), multi-call plan structure (0.3), non-trivial result (0.2), reasonable numeric range (0.1). `gold_answer` parameter removed from all APIs.

2. **Fixed extraction verification** (weakness #5): Programs now only kept if execution output matches gold answer (`gold_verified: True`). Ensures training data semantic correctness.

3. **Removed generic fallback** (weakness #6): Programs that can't be decomposed are skipped (no more falling back to "best-supported" subroutine).

4. **Updated all APIs and tests**: 22/22 tests pass with new label-free MCTS.

### Remaining for Round 2
- MCD split validity (needs diverse real programs)
- Empirical strength (needs full-scale extraction + training)
- Data provenance (need clean separation of synthetic/real)
- Novelty framing (needs evidence to back claims)

### Status
- Continuing to round 2
- Difficulty: medium

---

## Round 2 (2026-04-04)

### Assessment (Summary)
- Score: **3/10** (up from 2)
- Verdict: **NOT READY** (but engineering credibility improved)
- Reviewer verified: MCTS label-free fix, gold verification, fallback removal all correct

<details>
<summary>Click to expand full reviewer response</summary>

Rescore: 3/10. That is an improvement from 2/10, because the most serious validity bug is fixed in the main path. The engineering credibility is better. The scientific case is still weak.

Why It's Still Low:
- No fresh post-fix evidence yet. results/eval/ still contains the earlier weak/inconsistent artifacts.
- The core compositional benchmark is still not real: the saved split still has zero unseen compounds.
- The "real" pipeline is still provenance-contaminated: synthetic_used: true.
- The current real-plan pool is still mostly single-call: 26 multi-call out of 146.
- MCTS is now valid, but still heuristic. Rewarding "finite nontrivial numbers" is not a publishable justification by itself.

Minimum Viable Path To 6/10:
1. Narrow the paper. Fragment mining + composition planning as main claim. MCTS optional.
2. Produce one clean dataset story. GSM8K only, no synthetic fallback, >= 1k gold-verified programs.
3. Build a real compositional split. Multi-call plans, meaningful unseen compound divergence.
4. Hit one credible empirical result. Compose beats flat by >= 5 absolute points, 3 seeds, CIs.
5. Run only the ablations that matter. Fragment vs whole-program, compose vs flat, compose+MCTS.
6. Clean the artifact story. One command regenerates the final tables.

</details>

### Actions Taken
Round 2 is code-fix-only (no new experiments due to compute constraints).

The reviewer's path to 6/10 requires:
- **~4-14 hours of GPU time** for full GSM8K extraction (sequential LLM inference)
- This is the blocking constraint — all other fixes depend on having sufficient real data

### Status
- Round 2 complete (code fixes verified, path to 6/10 identified)
- **Next action**: Run full GSM8K extraction overnight (~3000 problems → ~1000 gold-verified)
- Difficulty: medium
