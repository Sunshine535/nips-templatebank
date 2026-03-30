# Research Proposal: Compositional Template Programs for Mathematical Reasoning

## Problem Anchor
- **Bottom-line problem**: Current LLM reasoning (CoT) regenerates reasoning traces from scratch for every problem, unable to systematically reuse and compose proven reasoning patterns.
- **Must-solve bottleneck**: Existing reasoning reuse methods cannot **compose** fragments from different reasoning traces. When a problem requires combining patterns A and B never seen together in training, all current methods fail.
- **Non-goals**: General program synthesis, pretraining, saturated benchmarks.
- **Constraints**: 8×A100, ~2000 GPU-hours, Qwen3.5-9B, NeurIPS 2026.
- **Success condition**: ≥15% accuracy gain on compositional held-out set over CoT, ≥10% over retrieval, ≥40% fewer tokens.

## Technical Gap

Reasoning reuse methods fall into four categories, all structurally unable to compose:
1. **Retrieval** (ReasoningBank): Returns whole traces; cannot mix fragments.
2. **Compression** (Metacognitive Reuse): Opaque tokens; cannot decompose/recombine.
3. **Code generation** (PAL/PoT): One-off programs; no template reuse or composition.
4. **Static frameworks** (Buffer of Thoughts): Hand-designed; cannot learn or compose.

Missing: **a template library of typed reasoning programs + a compiler that composes templates for new problems**.

## Method Thesis
We build a library of typed reasoning templates distilled from CoT traces, and train a compiler that maps new problems to **composition plans** — sequences of (template_id, slot_bindings) — that are expanded and executed via the template library. This enables compositional generalization: solving problems whose reasoning requires combining templates never paired in training.

## Contribution Focus
- **Dominant**: Compositional template reuse — template library as inference substrate + compiler that selects and composes templates.
- **Supporting**: Rigorous compositional evaluation protocol (3-layer unseen template-bigram holdout with difficulty matching and leakage audit).
- **Non-contributions**: No new architecture, no pretraining, no formal algebraic theory.

## Proposed Method

### Complexity Budget
- **Frozen**: Qwen3.5-9B backbone (LoRA only, r=16, α=32, ~26M params)
- **New**: Single LoRA adapter
- **Excluded**: No retrieval system, no RL, no verifier, no multi-stage training

### System Overview

```
=== Phase 1: Template Library Construction (Offline) ===
  For each (problem, answer) in GSM8K-train + MATH-Algebra-train:
    Teacher (Qwen3.5-32B) generates K=5 candidate JSON-AST programs
    Filter: JSON valid + types valid + executor correct answer
    Rank by parsimony + reuse
  Abstract valid programs → typed templates (values → slots)
  Cluster by normalized AST structure → deduplicate
  → Template Library L: ~200 typed template programs

=== Phase 2: Training Data Construction ===
  For each problem: map gold program → composition plan [(template_id, bindings), ...]
  Multiple valid plans → rank by parsimony + reuse → top-3 as training targets
  → 10K-15K (problem, composition_plan) pairs

=== Phase 3: Compiler Training ===
  Qwen3.5-9B + LoRA: problem → JSON composition plan
  Schema-constrained decoding (outlines): template IDs constrained to library
  Multi-reference + rejection sampling
  → Trained Compiler C

=== Inference ===
  Problem P → generate N=3 candidate plans → rerank (validity + execution + parsimony)
  → Expand from library → typed binding → COMPOSE → execute → answer
  → Fallback: standard CoT if all plans fail
```

### Core Mechanism: Template Composition Plans

The compiler outputs a **composition plan**, NOT a flat program:
```json
{
  "plan": [
    {"template_id": "multi_step_arithmetic_v3", "bindings": {"quantity": 12, "unit_price": 45.0}},
    {"template_id": "percentage_discount_v1", "bindings": {"rate": 0.15}}
  ]
}
```

**Execution**:
1. Load templates by ID from library L
2. Bind explicit slot values from plan
3. Auto-bind cross-template via typed output→input matching (COMPOSE)
4. Execute deterministically
5. Fallback on any failure

**Why compositional**: Compiler operates over template IDs. At training, sees (T_A,T_B) and (T_A,T_C). At test, composes (T_B,T_C) — never seen together. Template-level composition enables recombination.

### Template DSL
**Types**: int | float | string | bool | list[Type]
**Operators** (6): ASSIGN, COMPUTE, COMPARE, AGGREGATE, CONDITION, OUTPUT

### Typed Binding Algorithm
```
function BIND(plan, library):
  env = {}
  for (tid, bindings) in plan:
    template = library[tid]
    for (slot, value) in bindings:
      assert type(value) == template.input_slots[slot].type
      env[slot] = (value, type)
    for slot in template.input_slots not in bindings:
      candidates = env entries matching slot.type
      if |candidates| == 1: bind unambiguously
      elif |candidates| > 1: bind most recent (recency heuristic)
      else: FAIL → fallback
    execute template steps, update env
  return env["__output__"]
```

### Compiler Training
- Single LoRA on Qwen3.5-9B (r=16, α=32, dropout=0.05)
- Input: problem + template signatures → Output: JSON composition plan
- Schema-constrained decoding (outlines)
- Multi-reference (top-3 valid plans), rejection sampling
- lr=2e-4, cosine, 5 epochs, bf16, 8×A100
- ~10K-15K training pairs

### Inference-Time Reranking
- Generate N=3 candidate plans (nucleus p=0.9)
- Rerank: (1) type-check validity, (2) execution success, (3) shortest plan
- Execute top valid plan; fallback if all fail

### Failure Modes
- Teacher fails → filter by correctness, voting across K=5
- Too few templates → relax dedup, add MATH subfamilies
- Low compilation → curriculum training, more data
- No compositional gain → kill at CoT+5%

### Novelty
| | ReasoningBank | MetaCog | PAL/PoT | BoT | **Ours** |
|---|---|---|---|---|---|
| Unit | Full trace | Compressed tokens | One-off code | Static framework | **Typed template** |
| Composition | None | None | None | None | **Explicit COMPOSE** |
| Library | Retrieval | Replay | None | Manual | **Compile + compose** |

First to demonstrate compositional generalization through explicit template reuse in LLM math reasoning.

## Validation

### Compositional Split (3-Layer)
1. **Structural**: Hold out 15-20% of template bigram types
2. **Distributional**: Difficulty-match via KS test
3. **Leakage audit**: No paraphrases (BM25), individual templates seen in train, only bigram unseen
4. **Stress test**: 5% individual templates held out entirely

### Metrics (5-Level)
| Metric | Measures |
|--------|----------|
| Compiler coverage | % valid plans |
| Compiler-only accuracy | No fallback |
| Executor accuracy | Compilation succeeds |
| Fallback rate | % routed to CoT |
| Full system accuracy | End-to-end |

### Baselines (6)
CoT (0-shot), CoT (8-shot), PAL/PoT, BM25 trace retrieval, BM25+edit, BM25 template+greedy compose (same library)

### Ablations (3)
No-COMPOSE, no-types, flat-program (no template IDs)

### Analysis
- Error taxonomy: wrong template / wrong binding / execution error
- Template usage distribution, composition depth histogram
- Library statistics: coverage, reuse, bigram frequency
- Confidence intervals + paired significance tests

## Compute & Timeline
~500 GPU-hours total, 5-6 weeks.
