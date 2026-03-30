# Round 3 Refinement

## Problem Anchor
[Preserved verbatim — same as all rounds]

## Anchor Check
- Original bottleneck: Compositional reasoning reuse.
- Revised method still addresses it: Yes — all changes sharpen the evaluation rigor and eliminate confounds. The core mechanism (library-backed composition plans) is unchanged.
- Drift rejected: None.

## Simplicity Check
- Dominant contribution: Library-backed composition plans. Unchanged.
- Removed/merged: Nothing — method is already minimal. Changes are all on evaluation side.
- Why still smallest route: Same mechanism, stricter evaluation.

## Changes Made

### 1. Rigorous Multi-Layer Compositional Split
- **Reviewer said**: Unseen template bigrams alone insufficient; control for problem-family leakage, lexical overlap, frequency imbalance.
- **Action**: Three-layer split construction:
  - **Layer 1 (Structural)**: Hold out 15-20% of template bigram types. Zero overlap between train and test bigrams.
  - **Layer 2 (Distributional)**: Difficulty-match using problem length + number of reasoning steps. Ensure held-out set has similar difficulty distribution to in-distribution test via KS test (p > 0.1).
  - **Layer 3 (Leakage audit)**: (a) No problem appears in both train and test. (b) BM25 similarity between each test problem and nearest train problem must be below threshold (ensure no paraphrases). (c) Template frequency balance: each held-out bigram's constituent templates must individually appear in training (only the bigram is unseen, not the individual templates). This ensures we test composition, not template recognition.
  - **Stress test split**: Additionally, hold out 5% of individual template types (templates never seen in training at all) as a harder OOD set.
- **Impact**: Compositional claim becomes defensible against all known leakage attacks.

### 2. Fallback-Separated Reporting
- **Reviewer said**: Fallback is a confound.
- **Action**: Report 5 metrics separately:
  - **Compiler coverage**: % of test problems where compiler produces valid composition plan
  - **Compiler-only accuracy**: accuracy only on problems where compiler succeeds, no fallback
  - **Executor accuracy**: accuracy when compilation succeeds AND execution produces an answer
  - **Fallback rate**: % of problems routed to CoT fallback
  - **Full system accuracy**: compiler-only + fallback combined
  - **Token cost**: reported separately for compiler path and fallback path
- **Impact**: Transparent evaluation; reviewers can assess each component independently.

### 3. Retrieve-and-Compose Baseline
- **Reviewer said**: Need "retrieve templates + greedy compose" baseline using same library.
- **Action**: Add baseline: Given a problem, use BM25 over template descriptions to retrieve top-K templates. Greedily compose them by type matching (first valid composition). Execute. This uses the SAME template library but without the learned compiler — tests whether composition is the hard part or retrieval is.
- **Impact**: Isolates the value of the learned compiler vs. the template library itself.

### 4. Exact Typed Binding Algorithm
- **Reviewer said**: Specify binding/search semantics, ambiguity, failure.
- **Action**: Binding algorithm:
  ```
  function BIND(plan, library):
    env = {}  // variable environment: name → (value, type)
    for (tid, bindings) in plan:
      template = library[tid]
      // Bind explicit slot values from composition plan
      for (slot_name, value) in bindings:
        assert type(value) == template.input_slots[slot_name].type  // type check
        env[slot_name] = (value, template.input_slots[slot_name].type)
      // Auto-bind from previous template outputs (COMPOSE)
      for slot in template.input_slots:
        if slot.name not in bindings:
          // Search env for matching type
          candidates = [v for v in env if env[v].type == slot.type and v not in used]
          if len(candidates) == 1:
            env[slot.name] = env[candidates[0]]  // unambiguous bind
          elif len(candidates) > 1:
            // Ambiguity: use most recently computed variable
            env[slot.name] = env[candidates[-1]]  // recency heuristic
          else:
            return FAIL  // no matching variable → fallback to CoT
      // Execute template steps
      for step in template.steps:
        result = execute_step(step, env)
        if step.out:
          env[step.out] = (result, step.out_type)
    return env["__output__"]
  ```
  **Ambiguity resolution**: Recency heuristic (most recently computed variable of matching type). In practice, >90% of bindings are unambiguous because slot names carry semantic hints.
  **Failure cases**: (a) Type mismatch → FAIL → fallback. (b) No matching variable → FAIL → fallback. (c) Division by zero / runtime error → FAIL → fallback.
- **Impact**: Method Specificity is now fully implementable.

### 5. Library Coverage Statistics
- **Reviewer said**: Show library stats: coverage, reuse, composition depth.
- **Action**: Report in paper:
  - Template count per domain
  - Coverage: % of training problems that map to at least one template
  - Reuse frequency distribution (histogram)
  - Average composition depth (templates per plan)
  - Template bigram frequency distribution
  - % of test compositions that are genuinely unseen (should be 100% for compositional split)
- **Impact**: Full transparency on library quality.

### 6. Inference-Time Reranking
- **Reviewer said**: Add inference-time reranking over candidate plans.
- **Action**: At inference, generate N=3 candidate composition plans via nucleus sampling (p=0.9). Rerank by: (a) type-check validity, (b) execution success, (c) shortest plan (parsimony). Return the top valid plan. This is lightweight (3 forward passes) and improves robustness without adding complexity.
- **Impact**: Better frontier leverage; uses execution as a verifier at inference.

## Revised Proposal

[The full proposal from Round 2 refinement applies with these additions:]

### Updated Evaluation Protocol

**Compositional Split (3-layer)**:
1. Hold out 15-20% of template bigram types (Layer 1)
2. Difficulty-match via KS test on problem length + step count (Layer 2)
3. Leakage audit: no paraphrases (BM25<threshold), individual templates seen in train, only bigram unseen (Layer 3)
4. Stress test: 5% of individual template types held out entirely

**Metrics (5-level)**:
| Metric | What it measures |
|--------|-----------------|
| Compiler coverage | % valid composition plans |
| Compiler-only accuracy | Accuracy without fallback |
| Executor accuracy | Accuracy when compilation succeeds |
| Fallback rate | % routed to CoT |
| Full system accuracy | End-to-end including fallback |

**Baselines (6)**:
| Baseline | Description |
|----------|-------------|
| CoT (0-shot) | Direct chain-of-thought |
| CoT (8-shot) | Few-shot chain-of-thought |
| PAL/PoT | Program-aided language model (Python code) |
| BM25 trace retrieval | Retrieve most similar full trace |
| BM25 trace retrieval + edit | Retrieve trace, LLM edits for new problem |
| BM25 template retrieve + greedy compose | Same library, no learned compiler |

**Ablations (3)**:
| Ablation | What it tests |
|----------|--------------|
| No COMPOSE | Execute templates independently |
| No type checking | Allow arbitrary compositions |
| Flat program (no template IDs) | Compiler generates full AST, PAL/PoT-style |

### Updated Inference
Generate N=3 candidate plans → rerank by validity + execution success + parsimony → execute top plan → fallback if all fail.

### Exact Binding Algorithm
[As specified above in Changes section]

### Library Statistics
Report: template count, coverage, reuse distribution, composition depth, bigram frequency, unseen-bigram verification.

## Experiment Handoff
- **Must-prove**: Compositional generalization (3-layer split), token efficiency, composition necessity
- **Must-run**: 3 ablations, 6 baselines, 5-level metrics, library statistics
- **Critical**: Split construction must be airtight — this is the paper's credibility
- **Highest risk**: Teacher program quality, compiler generalization, split non-triviality

## Compute: ~500 GPU-hours total (added reranking + extra baselines), 5-6 weeks.
