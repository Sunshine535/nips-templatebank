# Proposal: Template Algebra — Formal Algebraic Operations on Reasoning Templates

## One-Sentence Summary

Define formal algebraic operations (compose, abstract, specialize) on CoT reasoning
templates and train a template compiler that decomposes new problems into executable
template programs, enabling compositional generalization beyond retrieval-based methods.

## Problem Statement

Chain-of-thought reasoning generates reasoning traces from scratch, repeatedly
"reinventing the wheel" for common reasoning patterns. Three recent lines of work
attempt to address this, but each has fundamental limitations:

1. **ReasoningBank (2025)**: Stores successful reasoning traces in a memory bank
   and retrieves the most similar trace for new problems. Limitation: pure retrieval
   cannot compose fragments from different traces — if no single stored trace
   matches the new problem's structure, ReasoningBank fails.

2. **Metacognitive Reuse (2025)**: Compresses recurring reasoning behaviors into
   reusable tokens, achieving 46% token reduction. Limitation: compressed tokens
   are opaque — they cannot be decomposed, recombined, or reasoned about. The
   compression is lossy and domain-specific.

3. **Framework of Thoughts**: Defines structured reasoning frameworks (e.g., "break
   into subproblems → solve each → combine"). Limitation: frameworks are manually
   designed and static — they cannot be learned, composed, or specialized.

## Critical Differentiation

| | ReasoningBank | Metacognitive Reuse | **Template Algebra (Ours)** |
|---|---|---|---|
| **Core mechanism** | Memory retrieval | Behavior compression | **Program synthesis** |
| **Template unit** | Full trace | Compressed token sequence | **Typed step graph** |
| **Novel composition** | Cannot compose | Cannot decompose | **Algebraic COMPOSE** |
| **Generalization** | Similarity matching | Pattern replay | **Type-safe compilation** |
| **Formal properties** | None | 46% compression | **Associativity, type system, inverse** |

ReasoningBank does retrieval → we do program synthesis.
Metacognitive Reuse compresses behaviors → we define formal algebra.

## Thesis (Falsifiable)

> A formal algebra over typed reasoning templates, combined with a trained
> template compiler, enables **compositional generalization** — solving problems
> that require novel combinations of known reasoning patterns — where retrieval-based
> and compression-based methods fail.

## Falsifiable Hypotheses

1. **H1 (Compositional generalization):** On held-out problems requiring novel
   template compositions, the template compiler achieves >= 15% absolute accuracy
   gain over CoT and >= 10% over ReasoningBank retrieval.

2. **H2 (Token efficiency):** Template programs use >= 40% fewer tokens than
   standard CoT at matched or better accuracy, comparable to Metacognitive Reuse's
   46% reduction.

3. **H3 (Compilation coverage):** The compiler successfully decomposes >= 80% of
   test problems into syntactically valid (well-typed) template programs.

4. **H4 (Algebra necessity):** Ablating algebraic operations (replacing COMPOSE
   with concatenation, removing BRANCH) degrades compositional generalization
   accuracy by >= 10%.

5. **H5 (Template transferability):** Templates extracted from GSM8K transfer to
   MATH with <= 5% accuracy gap on structurally similar problem types.

## Quantitative Success Criteria

| Criterion | Metric | Target | Comparison |
|-----------|--------|--------|------------|
| Primary | Compositional accuracy | >= 15% over CoT | Novel composition test set |
| Primary | Token efficiency | >= 40% reduction | vs standard CoT |
| Secondary | Compilation coverage | >= 80% valid programs | Over test problems |
| Secondary | Algebra ablation gap | >= 10% drop without algebra | vs full system |
| Tertiary | Cross-dataset transfer | <= 5% accuracy gap | GSM8K → MATH templates |

## Method

### Phase 1: Template Extraction

Use Qwen3.5-27B to extract abstract templates from GSM8K and MATH reasoning traces:

1. Parse each trace into a **step graph** — nodes are reasoning steps, edges are
   data dependencies (output of step i feeds into step j).
2. **Abstract** the step graph: replace specific numbers, entities, and expressions
   with typed variable slots. E.g., "Add 5 and 3" → "Add {x: number} and {y: number}".
3. **Deduplicate** structurally identical templates (same graph structure, same slot types).
4. Build **template library** with metadata: source problem type, success rate,
   average token count.

Expected: ~50-100 distinct templates from GSM8K, ~200-300 from MATH.

### Phase 2: Define Template Algebra

Formal operations on the template library:

- **COMPOSE(T₁, T₂):** Type-checked sequential composition. T₁'s output slots must
  type-match T₂'s input slots. Result is a merged step graph with internal slots bound.
- **ABSTRACT(T, bindings):** Replace concrete values with fresh variable slots.
  Increases template generality.
- **SPECIALIZE(T, values):** Bind variable slots to concrete values from the problem.
  Inverse of ABSTRACT.
- **SEQUENCE(T₁, ..., Tₙ):** Ordered composition with automatic slot binding between
  adjacent templates.
- **BRANCH(condition, T₁, T₂):** Conditional selection. Condition is a type predicate
  on the problem (e.g., "involves geometry?" → use T_geometry else T_algebra).

### Phase 3: Train Template Compiler

Train Qwen3.5-9B to map problems to template programs:

- **Input:** Problem statement (natural language)
- **Output:** Template program in our algebra DSL
- **Training data:** (problem, gold template program) pairs derived from successful
  trace-template alignments
- **Training objective:** Supervised fine-tuning on program generation
- **Constrained decoding:** Only generate syntactically valid programs (grammar-guided)
- **Execution-based feedback:** Verify compiled programs produce correct answers;
  filter training data accordingly

### Phase 4: Program Execution

Execute the compiled template program:
1. Parse program AST
2. For each SPECIALIZE node: bind variables from problem statement
3. For each COMPOSE/SEQUENCE: execute steps in topological order
4. For each BRANCH: evaluate condition, select appropriate sub-program
5. Generator (Qwen3.5-9B) fills in each step following the template structure
6. Return final answer

## Why Not Just Retrieve Similar Traces?

ReasoningBank retrieves the most similar complete trace. This fails because:
1. **Novel combinations:** If the problem requires pattern A (seen in math) + pattern B
   (seen in logic), but no single trace combines both, retrieval returns a partial match.
2. **No composition:** Retrieved traces are monolithic — you cannot extract step 3 from
   trace X and step 5 from trace Y and combine them.
3. **No abstraction hierarchy:** Retrieval treats all traces as flat — there is no
   notion of "this step is a specialization of that template."
4. **Similarity ≠ applicability:** The most similar-looking trace may have different
   reasoning structure.

## Why Not Just Compress Like Metacognitive Reuse?

Metacognitive Reuse achieves 46% token reduction by compressing recurring patterns:
1. **Opaque compression:** Compressed tokens cannot be inspected, decomposed, or reasoned about.
2. **No composition:** You cannot compose two compressed behaviors into a new one.
3. **Lossy:** Compression discards information that may be needed for novel problems.
4. **Domain-specific:** Compressed patterns don't transfer across domains.

Our algebra preserves **structure** — templates are typed step graphs that can be
composed, decomposed, and transferred with formal guarantees.

## Risk Analysis

| Risk | Mitigation |
|------|------------|
| Template extraction too noisy | Use Qwen3.5-27B (strongest model); filter by execution success |
| Template library too small | Start with GSM8K (rich reasoning diversity); augment with MATH |
| Compiler generates invalid programs | Grammar-constrained decoding; type checker in the loop |
| Compositional test set is trivial | Hand-design held-out combinations; verify with human evaluation |
| Algebra overhead exceeds savings | Template programs are symbolic → near-zero overhead for composition |
| Templates don't transfer across domains | Test on MATH (same domain) first; cross-domain is tertiary goal |

## Compute Budget

| Phase | GPUs | Duration | GPU-Hours |
|-------|------|----------|-----------|
| Trace generation (Qwen3.5-27B) | 8× A100 | 2 days | 384 |
| Template extraction + library | 2× A100 | 2 days | 96 |
| Compiler training (Qwen3.5-9B) | 8× A100 | 3 days | 576 |
| Evaluation suite | 4× A100 | 3 days | 288 |
| Ablations | 4× A100 | 2 days | 192 |
| **Total** | | **12 days** | **1536 GPU-hours** |

## Kill Criteria

Abandon if:
1. Template extraction yields < 30 distinct templates from GSM8K (insufficient diversity)
2. Compiler compilation coverage < 50% (cannot decompose most problems)
3. Compositional test accuracy <= CoT + 5% (algebra doesn't help with novel combinations)
4. Template programs are longer (more tokens) than CoT for > 50% of problems
