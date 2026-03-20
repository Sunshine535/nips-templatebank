# Execution Plan: Template Algebra

8-week stage-gated plan. Each stage has a **go/no-go** checkpoint.

---

## Timeline Overview

```
Week 1-2:  Stage 1 — Reasoning Trace Collection & Template Extraction
Week 3:    Stage 2 — Template Algebra Definition & Implementation
Week 4-5:  Stage 3 — Template Compiler Training
Week 6:    Stage 4 — Full Benchmark Evaluation
Week 7:    Stage 5 — Ablations & Compositional Analysis
Week 8:    Stage 6 — Paper Writing & Figures
```

---

## Stage 1: Reasoning Trace Collection & Template Extraction (Week 1-2)

### Objectives
- Generate high-quality CoT reasoning traces for GSM8K and MATH using Qwen3.5-27B
- Extract abstract templates from traces via the extraction pipeline
- Build initial template library with deduplication and metadata

### Tasks
1. Generate CoT traces for GSM8K train set (7473 problems):
   - Use Qwen3.5-27B with temperature=0.7, generate 5 traces per problem
   - Filter by execution correctness: keep only traces that produce correct answer
   - Expected: ~25K correct traces from 7473 problems
2. Generate CoT traces for MATH train set (7500 problems):
   - Same protocol; expect higher failure rate on harder problems
   - Expected: ~15K correct traces
3. Parse traces into step graphs:
   - Segment each trace into individual reasoning steps
   - Identify data dependencies between steps (output → input flow)
   - Represent as directed acyclic graph (DAG)
4. Abstract step graphs into templates:
   - Use Qwen3.5-27B to replace concrete values with typed variable slots
   - E.g., "5 apples + 3 apples = 8 apples" → "{x: count} {item: entity} + {y: count} {item} = {x+y: count} {item}"
5. Deduplicate templates by structural isomorphism
6. Compute template statistics: frequency, success rate, avg token count, domain coverage

### Deliverables
- [ ] ~40K correct reasoning traces (GSM8K + MATH)
- [ ] Step graph representations for all traces
- [ ] Template library: target 50-100 templates from GSM8K, 200-300 from MATH
- [ ] Template statistics and coverage analysis

### Go/No-Go (end of Week 2)
- **Go:** >= 50 distinct templates extracted from GSM8K with >= 5 instances each
- **Kill:** < 30 distinct templates (insufficient diversity for algebra)
- **Kill:** Step graph parsing fails on > 50% of traces (extraction pipeline broken)

---

## Stage 2: Template Algebra Definition & Implementation (Week 3)

### Objectives
- Implement the 5 algebraic operations with type checking
- Verify algebraic properties (associativity, inverse) on extracted templates
- Build template program DSL and parser

### Tasks
1. Implement Template data structure:
   - Step graph (DAG): nodes=steps, edges=data dependencies
   - Typed slots: number, expression, entity, boolean, list
   - Metadata: source problem type, frequency, avg success rate
2. Implement COMPOSE(T₁, T₂):
   - Type-check: T₁ output slots must match T₂ input slots
   - Merge step graphs: connect T₁ outputs to T₂ inputs
   - Handle slot renaming to avoid conflicts
3. Implement ABSTRACT(T, bindings):
   - Replace specified concrete values with fresh typed variable slots
   - Record binding map for invertibility verification
4. Implement SPECIALIZE(T, values):
   - Bind variable slots to concrete values from problem
   - Type-check: value types must match slot types
5. Implement SEQUENCE(T₁, ..., Tₙ):
   - Auto-bind: match output slots of Tᵢ to input slots of Tᵢ₊₁ by type
   - Return merged step graph with full data flow
6. Implement BRANCH(condition, T₁, T₂):
   - Condition: type predicate on problem features
   - Select and execute appropriate sub-template
7. Implement type checker and program validator
8. Define template program DSL grammar:
   ```
   Program := SPECIALIZE(TemplateRef, Bindings)
            | COMPOSE(Program, Program)
            | SEQUENCE(Program, Program, ...)
            | BRANCH(Condition, Program, Program)
   ```
9. Verify algebraic properties:
   - Test associativity of COMPOSE on 20 template triples
   - Test SPECIALIZE(ABSTRACT(T, b), b) ≈ T on 50 templates
   - Test type safety: ensure invalid compositions are rejected

### Deliverables
- [ ] Template data structure with all 5 operations
- [ ] Type checker and program validator
- [ ] Template program DSL grammar and parser
- [ ] Algebraic property verification report (associativity, inverse, type safety)

### Go/No-Go (end of Week 3)
- **Go:** All algebraic properties hold on test cases; type checker rejects all invalid compositions
- **Kill:** COMPOSE is not associative (fundamental algebra broken)

---

## Stage 3: Template Compiler Training (Week 4-5)

### Objectives
- Generate training data: (problem, gold template program) pairs
- Train Qwen3.5-9B compiler to map problems → template programs
- Implement grammar-constrained decoding

### Tasks
1. Generate training data:
   - For each (problem, trace, template) triplet from Stage 1, construct the
     gold template program that reproduces the trace
   - Use Qwen3.5-27B to generate program annotations:
     "This problem uses SEQUENCE(SPECIALIZE(T_arithmetic, {x=5, y=3}), SPECIALIZE(T_unit_convert, {unit='kg'}))"
   - Filter by execution verification: program must produce correct answer
   - Expected: ~15K valid (problem, program) pairs
2. Design compositional held-out test set:
   - Identify template pairs (Tᵢ, Tⱼ) that appear in COMPOSE in training data
   - Hold out problems that use specific (Tᵢ, Tⱼ) compositions never seen in training
   - Ensure held-out compositions are non-trivial (both templates are non-simple)
   - Target: ~500 held-out compositional test problems
3. Train Qwen3.5-9B compiler:
   - Input: problem text
   - Output: template program in DSL
   - Training: SFT with teacher forcing, lr=2e-5, batch=64, epochs=10
   - Grammar-constrained decoding via FSM (outlines library)
4. Evaluate compilation coverage:
   - % of test problems where compiler produces syntactically valid program
   - % of valid programs that execute to correct answer
5. Iterate on compiler training data quality:
   - Analyze compilation failures
   - Add corrective examples for common failure patterns

### Deliverables
- [ ] ~15K (problem, gold template program) training pairs
- [ ] ~500 compositional held-out test problems
- [ ] Trained compiler checkpoint
- [ ] Compilation coverage metrics (target >= 80% valid programs)

### Go/No-Go (end of Week 5)
- **Go:** Compilation coverage >= 70% AND executed accuracy >= CoT on in-distribution
- **Kill:** Compilation coverage < 50% after iteration (compiler cannot learn the DSL)
- **Kill:** Template program accuracy < CoT − 5% on in-distribution (algebra hurts, not helps)

---

## Stage 4: Full Benchmark Evaluation (Week 6)

### Objectives
- Evaluate on in-distribution, compositional, and OOD test sets
- Compare against all baselines
- Measure token efficiency

### Tasks
1. Prepare evaluation sets:
   - **In-distribution:** GSM8K test (1319 problems), MATH test (5000 problems)
   - **Compositional:** Held-out novel compositions (~500 problems)
   - **OOD:** AIME 2024 (30 problems), AMC 2024 (25 problems)
2. Run all baselines:
   - Standard CoT (zero-shot and 8-shot)
   - Few-shot CoT with BM25-selected exemplars
   - Buffer of Thoughts
   - ReasoningBank (our reimplementation with same trace memory)
   - Metacognitive Reuse (our reimplementation)
3. Run our system:
   - Compiler generates template program → execute → answer
   - Measure: accuracy, token count, compilation success rate
4. Measure token efficiency:
   - Count total tokens generated by each method per problem
   - Compute efficiency: accuracy / tokens_used
5. Statistical significance:
   - Bootstrap confidence intervals (1000 resamples)
   - McNemar's test for pairwise comparisons

### Deliverables
- [ ] Main results table: all methods × all test sets
- [ ] Token efficiency comparison table
- [ ] Compositional generalization results (KEY TABLE)
- [ ] Statistical significance tests

### Go/No-Go (end of Week 6)
- **Go:** >= 10% accuracy gain on compositional test over best baseline
- **Conditional:** If < 10% but > 5%, investigate and augment compositional test set

---

## Stage 5: Ablations & Analysis (Week 7)

### Objectives
- Ablate algebraic operations
- Analyze template library and compiler behavior
- Prepare supplementary material

### Tasks
1. **Ablation: COMPOSE operation** — replace with simple concatenation
   - Expect: compositional accuracy drops significantly
2. **Ablation: BRANCH operation** — remove conditional selection, always use first template
   - Expect: reduced flexibility on diverse problems
3. **Ablation: type system** — remove type checking, allow arbitrary compositions
   - Expect: more compilation errors, lower executed accuracy
4. **Ablation: template abstraction level** — concrete vs abstract vs over-abstract
   - Test: number of variable slots per template (few vs many)
5. **Ablation: compiler model size** — 4B vs 9B for compiler
6. **Analysis: template usage distribution** — which templates are most/least used?
7. **Analysis: compositional depth** — how many COMPOSE operations do compiled programs use?
8. **Analysis: failure cases** — categorize compiler failures
   - Cannot find matching templates
   - Type mismatch in composition
   - Correct program but wrong execution
9. **Analysis: template similarity** — cluster templates by structural similarity
   - Visualize template library structure
10. **Qualitative examples:** Show 5 best and 5 worst compiled programs

### Deliverables
- [ ] Ablation results table
- [ ] Template usage distribution plot
- [ ] Compositional depth histogram
- [ ] Failure case taxonomy
- [ ] Qualitative examples (5 success + 5 failure)

---

## Stage 6: Paper Writing (Week 8)

### Objectives
- Write NeurIPS 2026 submission (9 pages + references + appendix)

### Paper Outline
1. **Introduction** (1 page): CoT reuse problem, retrieval vs compression vs our algebra, key results
2. **Related Work** (1 page): ReasoningBank, Metacognitive Reuse, Buffer of Thoughts, program synthesis
3. **Template Algebra** (2 pages): Template structure, 5 operations, type system, algebraic properties
4. **Template Compiler** (1 page): Training data, architecture, grammar-constrained decoding
5. **Experiments** (2.5 pages): Setup, in-distribution, compositional generalization, OOD, token efficiency
6. **Analysis** (1 page): Ablations, template usage, failure cases
7. **Conclusion** (0.5 pages)
8. **Appendix**: DSL grammar, proof of algebraic properties, full template library, qualitative examples

### Deliverables
- [ ] Complete LaTeX manuscript
- [ ] All figures in publication quality
- [ ] Code release preparation

---

## Resource Allocation

| Stage | GPUs | Duration | GPU-Hours |
|-------|------|----------|-----------|
| Stage 1: Trace+Extraction | 8× A100 | 10 days | 1920 |
| Stage 2: Algebra Impl | 1× A100 | 5 days | 120 |
| Stage 3: Compiler Training | 8× A100 | 10 days | 1920 |
| Stage 4: Evaluation | 4× A100 | 5 days | 480 |
| Stage 5: Ablations | 4× A100 | 5 days | 480 |
| Stage 6: Writing | 0 | 5 days | 0 |
| **Total** | | **40 days** | **4920 GPU-hours** |

---

## Critical Path

```
Trace Generation ──→ Template Extraction ──→ Algebra Implementation
                                               │
                                               ▼
                          Training Data Generation ──→ Compiler Training
                                                          │
                                                          ▼
                                              Compositional Test Design
                                                          │
                                                          ▼
                                                    Full Evaluation
```

The critical path runs through template extraction → algebra → compiler training.
Algebra implementation (Stage 2) can begin in parallel with late Stage 1 tasks
(deduplication, statistics) since the algebra API doesn't depend on specific templates.
