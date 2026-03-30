# Round 1 Refinement

## Problem Anchor
- **Bottom-line problem**: Current LLM reasoning (CoT) regenerates reasoning traces from scratch for every problem, unable to systematically reuse and compose proven reasoning patterns. This leads to (a) redundant token generation, (b) inability to solve problems requiring novel combinations of known patterns, and (c) no formal guarantees on reasoning structure validity.
- **Must-solve bottleneck**: Existing reasoning reuse methods (ReasoningBank = whole-trace retrieval, Metacognitive Reuse = opaque compression) cannot **compose** fragments from different reasoning traces into new valid programs.
- **Non-goals**: General program synthesis, pretraining improvements, saturated benchmarks.
- **Constraints**: 8×A100, ~2000 GPU-hours, Qwen3.5-9B, NeurIPS 2026.
- **Success condition**: ≥15% accuracy gain on compositional test over CoT, ≥10% over retrieval, ≥40% fewer tokens.

## Anchor Check
- **Original bottleneck**: Compositional reasoning reuse — combining known patterns in novel ways.
- **Why the revised method still addresses it**: We sharpen the mechanism from a broad "formal algebra" to a concrete "typed template AST with constrained compilation." The bottleneck remains compositional generalization; the implementation becomes executable.
- **Reviewer suggestions rejected as drift**: None. All suggestions sharpen the existing direction.

## Simplicity Check
- **Dominant contribution after revision**: A concrete, executable DSL of typed reasoning template programs, compiled from problems via schema-constrained LLM generation, enabling compositional reuse of reasoning patterns.
- **Components removed or merged**: (1) DAG generality → tree/linear ASTs only, (2) SEQUENCE merged into COMPOSE, (3) BRANCH removed (GSM8K/MATH rarely need conditional strategy), (4) Two LoRAs merged into single structured-output LoRA, (5) "Formal algebra" language de-emphasized → "typed template programs" instead, (6) ABSTRACT/SPECIALIZE inverse claims dropped.
- **Reviewer suggestions rejected as unnecessary complexity**: None.
- **Why the remaining mechanism is still the smallest adequate route**: One DSL + one compiler + one constrained decoder = minimal mechanism for compositional template reuse.

## Changes Made

### 1. Concrete DSL with Closed Operator/Type Set
- **Reviewer said**: Replace arbitrary DAGs with executable AST/DSL, define 6-10 operators and closed type set.
- **Action**: Define a concrete AST with 8 operators (ASSIGN, COMPUTE, COMPARE, LOOKUP, CONVERT, AGGREGATE, CONDITION, OUTPUT) and 5 types (int, float, string, bool, list). Programs are trees, not arbitrary DAGs.
- **Reasoning**: Trees are sufficient for GSM8K/MATH; DAGs add complexity without compositional benefit.
- **Impact**: Method Specificity jumps from placeholder to implementable.

### 2. Direct Structured Program Distillation
- **Reviewer said**: Replace post-hoc trace parsing with structured program distillation from teacher.
- **Action**: Use Qwen3.5-32B (or strongest available open model) as teacher to generate gold JSON-AST programs directly, conditioned on (problem, correct_answer). Filter by type validity + answer correctness. No intermediate "parse NL trace into graph" step.
- **Reasoning**: Direct generation avoids the noisy NL→graph conversion bottleneck entirely. Teacher model generates the target representation directly.
- **Impact**: Removes the weakest link (trace parsing) and makes supervision cleaner.

### 3. Single Compiler with Schema-Constrained Output
- **Reviewer said**: Collapse two LoRAs; use schema-constrained JSON/AST generation.
- **Action**: Train a single LoRA adapter that generates complete template programs as JSON-AST in one pass. Use outlines library for JSON schema-constrained decoding. The schema enforces the DSL grammar at token level.
- **Reasoning**: Staged training (select then fill) is an unnecessary decomposition when schema-constrained generation already ensures structural validity. One-pass generation is simpler and just as effective.
- **Impact**: Halves training complexity, cleaner inference path.

### 4. Strict Compositional Split Construction
- **Reviewer said**: Define split over unseen template bigrams/trigrams, difficulty-match, leakage audit.
- **Action**: (1) Extract template type for each training problem. (2) Identify all template bigrams (ordered pairs of template types used in multi-step solutions). (3) Hold out 15-20% of bigram types entirely — no problem in training uses these specific compositions. (4) Difficulty-match held-out and in-distribution sets by problem difficulty level. (5) Report leakage audit: verify zero train-test template-bigram overlap.
- **Reasoning**: This makes the compositional claim falsifiable and defensible.
- **Impact**: Validation Focus becomes crisp and reviewer-proof.

### 5. Narrowed Scope: GSM8K Primary + MATH-Algebra Subfamily
- **Reviewer said**: Narrow to GSM8K + one MATH subfamily for first version.
- **Action**: Primary experiments on GSM8K (rich in multi-step word problems, good template diversity). Secondary on MATH-Algebra (closest to GSM8K reasoning patterns, manageable diversity). Drop geometry, combinatorics, number theory from initial scope.
- **Reasoning**: Better to nail one benchmark perfectly than spread thin across five.
- **Impact**: Feasibility improves, template quality improves.

### 6. De-emphasized Formal Algebra Language
- **Reviewer said**: Narrow story to one DSL, one compiler, one result. Drop inverse claims and formal language.
- **Action**: Reframe from "Template Algebra" to "Template Programs" — the key novelty is typed, composable template programs compiled by an LLM, not abstract algebraic properties. COMPOSE remains as the key operation (type-checked sequential composition of template programs). ABSTRACT and SPECIALIZE are now described as program construction utilities, not algebraic operators with formal properties.
- **Reasoning**: The paper's value is practical compositional generalization, not theoretical algebra.
- **Impact**: Contribution Quality and Venue Readiness improve by being mechanism-first.

## Revised Proposal

# Research Proposal: Typed Template Programs for Compositional Mathematical Reasoning

## Problem Anchor
- **Bottom-line problem**: Current LLM reasoning (CoT) regenerates reasoning traces from scratch for every problem, unable to systematically reuse and compose proven reasoning patterns.
- **Must-solve bottleneck**: Existing reasoning reuse methods cannot **compose** fragments from different reasoning traces. When a problem requires combining patterns A and B never seen together in training, all current methods fail.
- **Non-goals**: General program synthesis, pretraining, saturated benchmarks.
- **Constraints**: 8×A100, ~2000 GPU-hours, Qwen3.5-9B, NeurIPS 2026.
- **Success condition**: ≥15% accuracy gain on compositional held-out set over CoT, ≥10% over retrieval, ≥40% fewer tokens.

## Technical Gap

Reasoning reuse methods fall into three categories, each structurally unable to compose:
1. **Retrieval** (ReasoningBank): Returns whole traces; cannot mix fragments from different traces.
2. **Compression** (Metacognitive Reuse): Opaque compressed tokens; cannot decompose or recombine.
3. **Static frameworks** (Buffer of Thoughts): Hand-designed; cannot learn or compose from data.

The missing piece: **typed, executable reasoning programs** that can be composed from a library of templates and compiled from new problems via constrained LLM generation.

## Method Thesis
- **One-sentence thesis**: We distill CoT reasoning traces into a typed template program library with a closed AST, and train a single LLM compiler (LoRA) that maps new problems to composed template programs via schema-constrained decoding — enabling compositional generalization where retrieval and compression structurally cannot.
- **Why smallest adequate**: One DSL + one LoRA compiler + schema-constrained decoding. No retrieval system, no RL, no verifier, no multi-stage training.
- **Why timely**: Strong open-weight LLMs enable (1) high-quality program distillation from a teacher model and (2) schema-constrained generation that guarantees syntactic validity at decode time.

## Contribution Focus
- **Dominant contribution**: Typed template programs — a concrete, executable DSL for reasoning, with a learned compiler that enables compositional generalization on mathematical reasoning.
- **Supporting contribution**: A principled method for constructing compositional evaluation splits (unseen template-bigram holdout with difficulty matching and leakage audit).
- **Non-contributions**: No new model architecture, no pretraining, no formal algebraic theory.

## Proposed Method

### Complexity Budget
- **Frozen / reused**: Qwen3.5-9B backbone (frozen, LoRA only)
- **New trainable**: Single LoRA adapter (r=16, α=32, ~26M params, 0.3% of base)
- **Intentionally excluded**: No separate retrieval system, no RL, no verifier, no two-stage training, no BRANCH operation

### System Overview

```
=== Template Library Construction (Offline) ===
GSM8K/MATH problems + answers
  → Teacher model (Qwen3.5-32B) generates JSON-AST programs
  → Filter: type-valid AND answer-correct
  → Cluster by AST structure → deduplicate
  → Template Library (~100-300 typed template programs)

=== Compiler Training ===
(problem, gold_program_AST) pairs from above
  → Schema-constrained SFT on Qwen3.5-9B + LoRA
  → Single-pass: problem → complete JSON-AST program

=== Inference ===
New problem
  → Compiler (Qwen3.5-9B + LoRA) generates JSON-AST program
  → Schema validator checks type correctness
  → AST executor: walk tree, instantiate templates, compute answer
  → If invalid: fallback to standard CoT
```

### Core Mechanism: Typed Template Program DSL

**Type System** (closed set):
```
Type := int | float | string | bool | list[Type]
```

**Operator Set** (8 operators):
```
ASSIGN(var: string, value: Expr, type: Type)        → bind variable
COMPUTE(op: ArithOp, args: list[Expr], out: string) → arithmetic/algebraic computation
COMPARE(left: Expr, right: Expr, op: CmpOp)         → boolean comparison
LOOKUP(table: string, key: Expr, out: string)        → table/context lookup
CONVERT(value: Expr, from_unit: string, to_unit: string, out: string) → unit conversion
AGGREGATE(op: AggOp, values: list[Expr], out: string) → sum/mean/max/min/count
CONDITION(test: Expr, if_true: Program, if_false: Program) → conditional (minimal)
OUTPUT(expr: Expr)                                   → final answer
```
Where `ArithOp ∈ {add, sub, mul, div, pow, mod}`, `CmpOp ∈ {eq, lt, gt, le, ge}`, `AggOp ∈ {sum, mean, max, min, count}`.

**Program Structure**: A program is a list of steps, each using one operator. Variables are typed and scoped. A **template** is a program with some variables left as typed slots (placeholders).

**COMPOSE Operation**: Given template T₁ with output variables V₁ and template T₂ with input slots S₂, COMPOSE(T₁, T₂) concatenates the step lists and binds V₁ to matching-typed slots in S₂. Type checking ensures V₁ types match S₂ types.

**JSON-AST Format** (what the compiler generates):
```json
{
  "template_ids": ["arithmetic_multi_step", "percentage_discount"],
  "composition": "COMPOSE",
  "steps": [
    {"op": "ASSIGN", "var": "price", "value": "45", "type": "float"},
    {"op": "ASSIGN", "var": "quantity", "value": "12", "type": "int"},
    {"op": "COMPUTE", "op_type": "mul", "args": ["price", "quantity"], "out": "subtotal"},
    {"op": "ASSIGN", "var": "discount_rate", "value": "0.15", "type": "float"},
    {"op": "COMPUTE", "op_type": "mul", "args": ["subtotal", "discount_rate"], "out": "discount"},
    {"op": "COMPUTE", "op_type": "sub", "args": ["subtotal", "discount"], "out": "total"},
    {"op": "OUTPUT", "expr": "total"}
  ]
}
```

### Template Library Construction (Teacher Distillation)

1. **Teacher generation**: For each (problem, answer) in GSM8K-train + MATH-Algebra-train, prompt Qwen3.5-32B to generate a JSON-AST program that solves the problem. Use the exact JSON schema above in the prompt.
2. **Filtering**: Keep only programs where (a) JSON parses correctly, (b) all types are valid, (c) AST executor produces the correct answer.
3. **Template abstraction**: For each valid program, replace concrete values with typed slots to create a template. E.g., `"value": "45"` → `"value": "{price: float}"`.
4. **Clustering**: Group templates by normalized AST structure (operator sequence + type signature). Merge structurally identical templates, keeping the one with highest source-problem count.
5. **Library**: Target 80-120 templates from GSM8K, 150-250 from MATH-Algebra. Each template has: template_id, operator sequence, typed slots, source count, success rate.

### Compiler Training

**Single-pass schema-constrained SFT**:
- **Input**: `"Solve this problem as a template program:\n\n{problem_text}"`
- **Output**: Complete JSON-AST program (as above)
- **Model**: Qwen3.5-9B + LoRA (r=16, α=32, dropout=0.05, target: q/k/v/o/gate/up/down)
- **Training**: SFT with teacher forcing, lr=2e-4, cosine schedule, warmup 5%, bf16, gradient checkpointing
- **Batch**: per_device=4, grad_accum=4, effective batch=128 on 8 GPUs
- **Epochs**: 5 (early stop on val compilation coverage)
- **Constrained decoding**: outlines library with JSON schema for the AST format. Ensures every generated program is syntactically valid JSON matching the DSL schema.
- **Data**: ~10K-15K (problem, gold_program) pairs from teacher generation above

### AST Executor

Deterministic execution of generated JSON-AST programs:
1. Parse JSON into AST nodes
2. Walk the step list in order
3. For each step: resolve variable references from environment, apply operator, bind result
4. Type check at each step: ensure operator argument types match
5. On OUTPUT step: return the computed value as final answer
6. On any error (type mismatch, undefined variable, division by zero): return None → trigger CoT fallback

### Inference Pipeline
1. Input problem → Compiler generates JSON-AST (schema-constrained, ~50-150 tokens)
2. AST validator checks structural and type correctness
3. If valid: AST executor computes answer directly (deterministic, ~0 additional LLM tokens)
4. If invalid or executor fails: fallback to standard CoT with base Qwen3.5-9B
5. Token cost: ~50-150 tokens (program) vs ~200-500 tokens (CoT), giving 40-75% reduction

### Modern Primitive Usage
- **Qwen3.5-32B as teacher**: Generates gold programs offline. Its role is purely supervision — not used at test time.
- **Qwen3.5-9B as compiler**: Learns to map problems to programs. LoRA keeps it lightweight.
- **Schema-constrained decoding (outlines)**: Ensures 100% syntactic validity of generated programs. This is the key enabler — without it, program generation would have unacceptable error rates.
- **JSON-AST as interface**: Modern LLMs are natively good at structured JSON generation. This is a better interface than custom DSL syntax.

### Failure Modes and Diagnostics
- **Teacher generates wrong programs**: Filter by executor correctness. Monitor teacher success rate per problem type. If <50% for a category, add few-shot examples to teacher prompt.
- **Too few templates**: If <50 templates from GSM8K, relax dedup threshold or include more MATH subfamilies.
- **Compiler compilation coverage too low**: If <70% valid programs on val set, add curriculum (simple problems first), increase training data, or add rejection sampling.
- **Composed programs fail**: If compositional accuracy <= CoT+5%, this is a kill condition — the mechanism doesn't help.

### Novelty and Elegance Argument
**Closest work**: ReasoningBank (retrieves whole traces, no composition), Metacognitive Reuse (compresses to opaque tokens, no decomposition), Buffer of Thoughts (static frameworks, no learning), PAL/PoT (generates code, no template reuse or composition).

**Exact difference**: We are the first to (1) define a typed DSL for reasoning templates, (2) train a compiler that generates composed template programs, and (3) demonstrate compositional generalization on a rigorously constructed held-out split. The key insight is that **composition should happen at the program level, not the trace level** — and modern constrained decoding makes this practical.

**Why focused**: One DSL + one compiler + one executor. No retrieval, no RL, no multi-agent, no iterative refinement. The mechanism is the composition; the enabler is constrained decoding.

## Claim-Driven Validation Sketch

### Claim 1: Compositional Generalization (Primary)
- **Minimal experiment**: Construct held-out set by identifying template bigrams (ordered pairs of template types used in multi-step solutions). Hold out 15-20% of bigram types entirely. Difficulty-match with in-distribution test. Run leakage audit (zero train-test template-bigram overlap).
- **Baselines**: Standard CoT (0-shot, 8-shot), BM25-retrieved trace (ReasoningBank-style), our method
- **Metric**: Exact-match accuracy on held-out compositional test (~500 problems)
- **Expected evidence**: ≥15% over CoT, ≥10% over retrieval

### Claim 2: Token Efficiency
- **Minimal experiment**: Count end-to-end tokens (including program generation and any fallback CoT) per problem.
- **Baselines**: Same as Claim 1
- **Metric**: Average tokens per problem, accuracy/token ratio
- **Expected evidence**: ≥40% fewer tokens at matched or better accuracy

### Claim 3: Composition Mechanism Necessity (Ablation)
- **Minimal experiment**: (a) Replace COMPOSE with independent template execution (no variable binding across templates), (b) Remove type checking (allow any composition), (c) Replace template programs with direct CoT + template prompt (no AST, just template description in natural language)
- **Metric**: Accuracy degradation on compositional test
- **Expected evidence**: ≥10% drop without COMPOSE, ≥5% drop without type checking

## Experiment Handoff Inputs
- **Must-prove claims**: Compositional generalization, token efficiency, composition necessity
- **Must-run ablations**: COMPOSE ablation, type checking ablation, template-prompt-only ablation
- **Critical datasets**: GSM8K test, MATH-Algebra test, compositional held-out (~500)
- **Highest-risk assumptions**: (1) Teacher generates enough high-quality programs, (2) Compiler learns to compose templates it hasn't seen together, (3) Compositional split is non-trivial

## Compute & Timeline Estimate
- **Teacher generation**: ~100 GPU-hours (Qwen3.5-32B on 8×A100, ~15K problems × 5 attempts)
- **Template extraction & library**: ~20 GPU-hours (clustering, dedup, statistics)
- **Compiler training**: ~80 GPU-hours (Qwen3.5-9B + LoRA, 5 epochs, 8 GPUs)
- **Evaluation**: ~60 GPU-hours (inference on test sets)
- **Ablations**: ~60 GPU-hours (3 ablation variants)
- **Buffer**: ~80 GPU-hours
- **Total**: ~400 GPU-hours (well within 2000 budget)
- **Timeline**: 5-6 weeks
