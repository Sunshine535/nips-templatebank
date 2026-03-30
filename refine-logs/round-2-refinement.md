# Round 2 Refinement

## Problem Anchor
[Same as all rounds — preserved verbatim]
- **Bottom-line problem**: CoT regenerates traces from scratch; cannot reuse/compose proven patterns.
- **Must-solve bottleneck**: No method can compose reasoning fragments from different traces.
- **Non-goals**: General program synthesis, pretraining, saturated benchmarks.
- **Constraints**: 8×A100, ~2000 GPU-hours, Qwen3.5-9B, NeurIPS 2026.
- **Success condition**: ≥15% on compositional held-out over CoT, ≥10% over retrieval, ≥40% fewer tokens.

## Anchor Check
- **Original bottleneck**: Compositional reasoning reuse.
- **Why revised method still addresses it**: The key revision makes composition a **real inference-time mechanism** — the compiler outputs template references + slot bindings (not flat ASTs), and template composition happens explicitly at decode and execution time.
- **Reviewer suggestions rejected as drift**: None. All changes sharpen composition as the central mechanism.

## Simplicity Check
- **Dominant contribution**: Template-level compositional reuse — a template library used as the inference-time reasoning substrate, where the compiler selects and composes templates via typed slot binding.
- **Components removed/merged**: Removed LOOKUP and CONVERT operators (unnecessary for GSM8K/MATH-Algebra). Reduced to 6 core operators.
- **Why minimal**: One template library + one compiler + one executor. Composition is the mechanism, not a label.

## Changes Made

### 1. COMPOSE is now a Real Inference-Time Mechanism
- **Reviewer said**: COMPOSE must operate over reusable templates at inference time, not just be a label on flat ASTs.
- **Action**: Redesigned compiler output format. The compiler now generates a **composition plan**: a sequence of `(template_id, slot_bindings)` pairs. The executor then:
  (a) Looks up each template_id in the library to get the template AST
  (b) Binds slot values from the composition plan
  (c) Connects templates via typed output→input variable binding (COMPOSE)
  (d) Executes the composed program
  This means the template library is the actual inference substrate, and composition is a real operation, not syntactic sugar.
- **Impact**: Contribution Quality — composition is now operationalized and falsifiable.

### 2. Multi-Reference Training with Execution-Guided Selection
- **Reviewer said**: Single-reference AST SFT is too weak for one-to-many compositional targets.
- **Action**: For each training problem, generate K=5 candidate composition plans from the teacher. Filter by executor correctness. Rank valid plans by (a) fewest templates used (parsimony), (b) highest template reuse count in library (prefer common patterns). Train on top-ranked valid plan, with rejection sampling: at training time, if the model generates any valid plan that executes correctly, count it as correct.
- **Impact**: Training signal is richer; model learns multiple valid composition strategies.

### 3. Stronger Baselines Including PAL/PoT
- **Reviewer said**: Need retrieval-edit baseline and PAL/PoT comparison.
- **Action**: Add baselines: (a) PAL (Program-Aided Language Model — LLM generates Python code), (b) PoT (Program of Thought), (c) BM25 trace retrieval + edit (retrieve closest trace, let LLM edit it for the new problem). These cover the code-generation and retrieval-edit landscape.
- **Impact**: Positioning is clear: we differ from PAL/PoT by using **reusable typed templates with explicit composition**, not one-off code generation.

### 4. Detailed Error Analysis and Metrics
- **Reviewer said**: Report compiler-only accuracy, fallback rate, error breakdown.
- **Action**: Report: (a) Compilation coverage (% valid composition plans), (b) Compiler-only accuracy (no fallback), (c) Fallback rate and fallback accuracy, (d) End-to-end accuracy (compiler + fallback), (e) Error taxonomy: wrong template selection / wrong slot binding / correct plan but execution error, (f) Token cost: compiler tokens + executor tokens (zero for deterministic) separately.
- **Impact**: Transparent, reviewer-proof evaluation.

### 5. Narrowed Operator Set
- **Reviewer said**: Remove low-usage operators if GSM8K/MATH-Algebra barely need them.
- **Action**: Reduced from 8 to 6 operators: ASSIGN, COMPUTE, COMPARE, AGGREGATE, CONDITION, OUTPUT. Removed LOOKUP and CONVERT (rare in target benchmarks). Can be re-added for other domains.
- **Impact**: Simpler DSL, cleaner learning signal.

### 6. Tightened Novelty Claim
- **Reviewer said**: Don't claim "first typed DSL" broadly; narrow to "template-level compositional reuse."
- **Action**: Novelty claim is now: "We are the first to demonstrate **compositional generalization through explicit template reuse** in LLM mathematical reasoning — where the compiler selects and composes templates from a learned library, and composition is verified through a rigorous unseen-bigram evaluation protocol."
- **Impact**: Defensible against PAL/PoT comparisons (they don't do template reuse/composition).

## Revised Proposal

# Research Proposal: Compositional Template Programs for Mathematical Reasoning

## Problem Anchor
- **Bottom-line problem**: CoT regenerates reasoning traces from scratch; cannot systematically reuse and compose proven reasoning patterns.
- **Must-solve bottleneck**: No existing method can compose fragments from different reasoning traces into new valid reasoning programs.
- **Non-goals**: General program synthesis, pretraining, saturated benchmarks.
- **Constraints**: 8×A100, ~2000 GPU-hours, Qwen3.5-9B, NeurIPS 2026.
- **Success condition**: ≥15% accuracy on compositional held-out over CoT, ≥10% over retrieval, ≥40% fewer tokens.

## Technical Gap

Three categories of reasoning reuse, all unable to compose:
1. **Retrieval** (ReasoningBank): Whole traces, no fragment mixing.
2. **Compression** (Metacognitive Reuse): Opaque tokens, no decomposition.
3. **Code generation** (PAL/PoT): One-off programs, no template reuse or composition.
4. **Static frameworks** (Buffer of Thoughts): Hand-designed, no learning.

Missing piece: **a template library of typed reasoning programs + a compiler that composes templates for new problems**.

## Method Thesis
We build a library of typed reasoning templates distilled from CoT traces, and train a compiler that maps new problems to **composition plans** — sequences of (template_id, slot_bindings) pairs that are expanded and executed via the template library. This enables compositional generalization: solving problems whose reasoning requires combining templates never paired in training.

## Contribution Focus
- **Dominant**: Compositional template reuse — a learned template library that serves as the inference-time reasoning substrate, with a compiler that selects and composes templates.
- **Supporting**: Rigorous compositional evaluation (unseen template-bigram holdout + difficulty matching + leakage audit).
- **Non-contributions**: No new architecture, no pretraining, no formal algebraic theory.

## Proposed Method

### Complexity Budget
- Frozen: Qwen3.5-9B backbone
- New: Single LoRA adapter (r=16, α=32, ~26M params)
- Excluded: No retrieval system, no RL, no verifier, no multi-stage training

### System Overview

```
=== Phase 1: Template Library Construction (Offline) ===
  For each (problem, answer) in GSM8K-train + MATH-Algebra-train:
    Teacher (Qwen3.5-32B) generates K=5 candidate JSON-AST programs
    Filter: JSON valid + types valid + executor produces correct answer
    Rank by parsimony + template reuse frequency
  Abstract valid programs → typed templates (replace values with slots)
  Cluster by normalized AST structure → deduplicate
  → Template Library L: 100-300 typed template programs
  
=== Phase 2: Training Data Construction ===
  For each (problem, answer):
    Map the gold program to a COMPOSITION PLAN:
      [(template_id_1, {slot: value, ...}), (template_id_2, {slot: value, ...}), ...]
    where each template_id references a template in L
    Multiple valid plans → rank by parsimony + reuse → top plan as gold
  → 10K-15K (problem, composition_plan) pairs

=== Phase 3: Compiler Training ===
  Qwen3.5-9B + LoRA learns: problem → composition_plan (JSON)
  Schema-constrained decoding via outlines
  Execution-guided rejection sampling during training
  → Trained Compiler C

=== Inference ===
  New problem P
    → C(P) = composition_plan = [(tid_1, bindings_1), (tid_2, bindings_2), ...]
    → Expand: for each (tid, bindings), look up template in L, bind slots
    → COMPOSE: connect templates via typed output→input variable binding
    → Execute composed program deterministically → answer
    → If any step fails: fallback to standard CoT
```

### Core Mechanism: Template Composition

**The key insight**: The compiler does NOT generate full programs. It generates **composition plans** that reference templates by ID. The template library L is the inference-time reasoning substrate. Composition happens **explicitly**:

1. **Template Selection**: Compiler selects template_id(s) from L
2. **Slot Binding**: Compiler specifies concrete values for each template's typed slots
3. **Composition**: Templates are connected via typed variable binding:
   - Template T₁ has output variable `result: float`
   - Template T₂ has input slot `{base: float}`
   - COMPOSE binds `result → base` (type-checked: float = float)
4. **Execution**: Walk the composed template program, compute answer deterministically

**Why this is compositional**: At training time, the model sees (T_A, T_B) composed together and (T_A, T_C) composed together. At test time, it must compose (T_B, T_C) — a pair never seen in training. Because the compiler operates over **template IDs** (not raw steps), it can recombine templates in novel ways.

**Composition Plan Format** (compiler output):
```json
{
  "plan": [
    {"template_id": "multi_step_arithmetic_v3", "bindings": {"quantity": 12, "unit_price": 45.0}},
    {"template_id": "percentage_discount_v1", "bindings": {"rate": 0.15}}
  ]
}
```

The executor then:
1. Loads `multi_step_arithmetic_v3` from L → gets template AST with slots
2. Binds `{quantity: 12, unit_price: 45.0}` → instantiated AST₁
3. Loads `percentage_discount_v1` from L → gets template AST with slots
4. Identifies T₁ output `subtotal: float` matches T₂ input `{base: float}` → binds
5. Binds `{rate: 0.15}` → instantiated AST₂
6. Executes AST₁ then AST₂ → final answer

### Template DSL

**Types**: int | float | string | bool | list[Type]

**Operators** (6):
- ASSIGN(var, value, type)
- COMPUTE(op: {add,sub,mul,div,pow,mod}, args, out)
- COMPARE(left, right, op: {eq,lt,gt,le,ge})
- AGGREGATE(op: {sum,mean,max,min,count}, values, out)
- CONDITION(test, if_true, if_false)
- OUTPUT(expr)

**Template**: Operator sequence with typed placeholder slots. Each template has:
- template_id (unique)
- input_slots: list of {name, type}
- output_vars: list of {name, type}
- steps: list of operator applications
- source_count: number of training problems this template covers
- success_rate: fraction of correct executions

### Template Library Construction

1. **Teacher generation**: Qwen3.5-32B generates K=5 JSON-AST programs per problem. Prompt includes JSON schema + 3 examples.
2. **Execution filtering**: Keep only programs where executor produces correct answer.
3. **Template abstraction**: Replace all concrete numeric/string values with typed slots.
4. **Normalization**: Canonicalize variable names (v0, v1, ...), sort ASSIGN steps.
5. **Clustering**: Group by normalized operator sequence + type signature. Merge identical structures.
6. **Library**: Each template gets metadata (source_count, success_rate, avg_slots).
7. **Target**: 80-120 GSM8K templates, 150-250 MATH-Algebra templates. Expect ~200 total after dedup.

### Compiler Training

**Architecture**: Qwen3.5-9B + single LoRA (r=16, α=32, dropout=0.05)
**Input**: `"Generate a composition plan for:\n\n{problem_text}\n\nAvailable templates: {template_id_list_with_signatures}"`
**Output**: JSON composition plan `{"plan": [{"template_id": ..., "bindings": {...}}, ...]}`
**Constrained decoding**: outlines JSON schema enforcement. Template IDs constrained to library.
**Training details**:
- SFT with teacher forcing
- lr=2e-4, cosine schedule, warmup 5%
- bf16, gradient checkpointing
- per_device_batch=4, grad_accum=4, effective batch=128
- 5 epochs, early stop on val compilation coverage
- **Multi-reference**: For each problem, keep top-3 valid composition plans ranked by parsimony
- **Rejection sampling**: During training, if model generates any valid plan → count as positive

### AST Executor
Deterministic, no LLM inference needed:
1. Load templates by ID from library
2. Bind slot values from composition plan
3. Connect templates via typed output→input binding (COMPOSE)
4. Walk steps in order, apply operators, maintain variable environment
5. Type check at each step
6. Return final OUTPUT value
7. On error → fallback to standard CoT

### Modern Primitive Usage
- **Qwen3.5-32B (teacher)**: Offline program generation. Not used at test time.
- **Qwen3.5-9B (compiler)**: Maps problems to composition plans. LoRA keeps it lightweight.
- **Schema-constrained decoding (outlines)**: Ensures 100% syntactic validity. Constrains template IDs to library.
- **JSON as interface**: LLMs generate structured JSON naturally. Better than custom syntax.

### Failure Modes
- Teacher fails (<50% valid programs): Add more few-shot examples, use voting across K=5 attempts.
- Too few templates (<50): Relax dedup threshold, add MATH subfamilies.
- Low compilation coverage (<70%): Curriculum training (1-template plans first), more training data.
- No compositional gain (≤CoT+5%): Kill condition.

### Novelty and Elegance Argument

**Closest work and exact differences**:
| | ReasoningBank | MetaCog Reuse | PAL/PoT | Buffer of Thoughts | **Ours** |
|---|---|---|---|---|---|
| Reasoning unit | Full trace | Compressed tokens | One-off code | Static framework | **Typed template** |
| Composition | None | None | None | None | **Explicit COMPOSE** |
| Library reuse | Retrieval only | Replay only | None | Manual | **Compile + compose** |
| Generalization | Similarity | Pattern | Code syntax | None | **Unseen compositions** |

**Why not PAL/PoT**: PAL/PoT generates one-off Python code per problem. No template library, no reuse, no composition. Each program is unique. Our compiler references **shared templates by ID** and **composes** them — enabling generalization to unseen template combinations.

**Why focused**: One library + one compiler + one executor. The mechanism is composition over typed templates. Nothing else.

## Claim-Driven Validation

### Claim 1: Compositional Generalization (Primary)
- **Split**: Identify all template bigrams in training. Hold out 15-20% of bigram types. Difficulty-match. Leakage audit.
- **Baselines**: CoT (0/8-shot), PAL/PoT, BM25 trace retrieval, BM25 retrieval + LLM edit, our method
- **Metrics**: Exact-match accuracy on compositional held-out (~500 problems)
- **Sub-metrics**: Compiler-only accuracy, fallback rate, fallback accuracy
- **Expected**: ≥15% over CoT, ≥10% over retrieval, ≥5% over PAL/PoT on compositional set

### Claim 2: Token Efficiency
- **Metrics**: Compiler tokens (composition plan), executor tokens (zero), fallback tokens, total tokens
- **Expected**: ≥40% fewer total tokens vs CoT at matched or better accuracy

### Claim 3: Composition Necessity (Ablation)
- **(a) No COMPOSE**: Execute each template independently (no variable binding across templates)
- **(b) No type checking**: Allow arbitrary compositions regardless of type
- **(c) Flat program** (no template IDs): Compiler generates full AST directly (PAL/PoT-style)
- **Expected**: ≥10% drop on compositional set without COMPOSE

### Additional Analysis
- Error taxonomy: wrong template / wrong binding / correct plan but exec error
- Template usage distribution: which templates most/least used
- Compositional depth histogram: how many templates per plan
- Qualitative examples: 5 best + 5 worst compositions

## Experiment Handoff
- **Must-prove**: Compositional generalization, token efficiency, composition necessity
- **Must-run ablations**: No-COMPOSE, no-types, flat-program
- **Critical datasets**: GSM8K test, MATH-Algebra test, compositional held-out
- **Highest-risk**: (1) Teacher program quality, (2) Compiler generalizes to unseen compositions, (3) Split is non-trivial

## Compute & Timeline
- Teacher generation: ~100 GPU-hours
- Library construction: ~20 GPU-hours
- Compiler training: ~80 GPU-hours
- Evaluation + ablations: ~120 GPU-hours
- Buffer: ~80 GPU-hours
- **Total: ~400 GPU-hours, 5-6 weeks**
