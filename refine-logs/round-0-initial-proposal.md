# Research Proposal: Template Algebra — Compositional Reasoning via Formal Template Programs

## Problem Anchor
- **Bottom-line problem**: Current LLM reasoning (CoT) regenerates reasoning traces from scratch for every problem, unable to systematically reuse and compose proven reasoning patterns. This leads to (a) redundant token generation, (b) inability to solve problems requiring novel combinations of known patterns, and (c) no formal guarantees on reasoning structure validity.
- **Must-solve bottleneck**: Existing reasoning reuse methods (ReasoningBank = whole-trace retrieval, Metacognitive Reuse = opaque compression) cannot **compose** fragments from different reasoning traces into new valid programs. When a problem requires pattern A from domain X combined with pattern B from domain Y, all current methods fail.
- **Non-goals**: We do NOT aim to build a general program synthesis system, improve base model pretraining, or compete on standard benchmarks where vanilla CoT already saturates. We focus specifically on **compositional generalization** — solving problems whose reasoning structure requires combining templates never seen together in training.
- **Constraints**: 8×A100 80GB cluster, ~2000 GPU-hours budget, Qwen3.5-9B as student model (no API-only models), NeurIPS 2026 submission, GSM8K + MATH as primary benchmarks.
- **Success condition**: On a held-out compositional test set (problems requiring template combinations unseen in training), our method achieves ≥15% absolute accuracy gain over standard CoT and ≥10% over the best retrieval-based baseline, while using ≥40% fewer tokens.

## Technical Gap

Current methods fail at **compositional reasoning reuse** for three structural reasons:

1. **ReasoningBank (retrieval)**: Retrieves the single most similar complete trace. Cannot extract step 3 from trace X and step 5 from trace Y to form a new reasoning program. Monolithic retrieval has zero compositional capability.

2. **Metacognitive Reuse (compression)**: Compresses recurring patterns into opaque token sequences. These compressed tokens cannot be inspected, decomposed, or recombined. Compression ≠ composition.

3. **Buffer of Thoughts / Framework approaches**: Use manually designed static frameworks. Cannot learn new compositional patterns from data.

The gap is **a formal algebra that makes reasoning templates composable**. No existing work provides typed, composable reasoning primitives with formal algebraic properties (associativity, type safety, inverse operations) that enable a compiler to synthesize novel reasoning programs from a template library.

**Why naive fixes fail**: Simply retrieving more traces or using longer contexts does not create compositional ability — it just increases the search space without structure. A larger model does not solve the composition problem either; it just memorizes more patterns without making them reusable.

## Method Thesis
- **One-sentence thesis**: We define a typed algebra over reasoning templates extracted from CoT traces, and train a template compiler that decomposes new problems into executable template programs — enabling compositional generalization where retrieval and compression methods structurally cannot.
- **Why this is the smallest adequate intervention**: The algebra adds formal structure (types, composition rules) to existing CoT traces without modifying the base model. The compiler is a lightweight LoRA adapter, not a new architecture.
- **Why this route is timely in the foundation-model era**: LLMs generate high-quality reasoning traces that serve as rich training signal for template extraction. The compiler leverages the base model's language understanding while adding structured program generation via constrained decoding. This is program-synthesis-over-natural-language, enabled by strong base models.

## Contribution Focus
- **Dominant contribution**: Template Algebra — a formal type system and algebraic operations (COMPOSE, ABSTRACT, SPECIALIZE, BRANCH) over reasoning templates, with provable properties (associativity, type safety, inverse), enabling compositional generalization in mathematical reasoning.
- **Optional supporting contribution**: Two-stage template compiler training (template selection → variable instantiation) with grammar-constrained decoding that ensures all generated programs are syntactically valid.
- **Explicit non-contributions**: We do not claim improvements on problems where standard CoT already works well. We do not claim a new architecture. We do not claim cross-domain transfer (that is tertiary).

## Proposed Method

### Complexity Budget
- **Frozen / reused backbone**: Qwen3.5-9B (frozen weights, only LoRA adapters trained)
- **New trainable components**: (1) Template selection LoRA adapter, (2) Variable filling LoRA adapter — both r=16, ~0.1% of base params each
- **Tempting additions intentionally not used**: No reinforcement learning, no self-play, no iterative refinement, no verifier model, no retrieval-augmented generation at inference. The algebra itself provides the structure; we don't need a separate retrieval system.

### System Overview

```
Training Pipeline:
  GSM8K/MATH traces → [Qwen3.5-9B extracts templates] → TemplateBank (50-300 typed templates)
  TemplateBank + traces → [Alignment] → (problem, gold_program) pairs
  Pairs → [Stage 1 SFT: template selection] → Selection LoRA
  Pairs → [Stage 2 SFT: variable filling] → Filling LoRA

Inference Pipeline:
  New problem → [Compiler (Selection LoRA)] → template program in DSL
  Template program → [Type checker validates] → valid program
  Valid program → [Compiler (Filling LoRA)] → instantiated reasoning steps
  Steps → [Execute] → final answer
```

### Core Mechanism: Template Algebra

**Template data structure**:
- Each template T is a typed step graph: DAG where nodes are reasoning steps, edges are data dependencies
- Each step has typed input/output slots: {name: type} where type ∈ {number, expression, entity, boolean, list}
- Templates carry metadata: domain, frequency, success rate

**Algebraic operations**:
1. **COMPOSE(T₁, T₂)**: Type-checked sequential composition. T₁'s output slots must type-match T₂'s input slots. Merges step graphs with internal slots bound. Formally: if T₁: A → B and T₂: B → C, then COMPOSE(T₁,T₂): A → C.
2. **ABSTRACT(T, bindings)**: Replace concrete values with fresh typed variable slots. Increases template generality. ABSTRACT is the left-inverse of SPECIALIZE: SPECIALIZE(ABSTRACT(T,b),b) ≈ T.
3. **SPECIALIZE(T, values)**: Bind variable slots to concrete values from the problem. Type-checked: value types must match slot types.
4. **SEQUENCE(T₁,...,Tₙ)**: Ordered composition with automatic slot binding between adjacent templates via type matching.
5. **BRANCH(condition, T₁, T₂)**: Conditional selection based on type predicates on problem features.

**Formal properties**:
- COMPOSE is associative: COMPOSE(COMPOSE(T₁,T₂),T₃) = COMPOSE(T₁,COMPOSE(T₂,T₃))
- SPECIALIZE(ABSTRACT(T,b),b) ≈ T (inverse property)
- Type checker rejects invalid compositions (soundness)

**DSL Grammar**:
```
Program := SPECIALIZE(TemplateRef, Bindings)
         | COMPOSE(Program, Program)
         | SEQUENCE(Program, ...)
         | BRANCH(Condition, Program, Program)
Bindings := {VarName: Value, ...}
Condition := TypePredicate(ProblemFeature)
```

### Template Extraction Pipeline

1. Generate 5 CoT traces per problem using Qwen3.5-9B (teacher mode, temperature=0.7)
2. Filter by execution correctness (keep only traces producing correct answers)
3. Use Qwen3.5-9B to parse each trace into a step graph with typed slots
4. Abstract step graphs: replace concrete values with typed variables
5. Cluster by structural isomorphism, merge similar templates
6. Target: 50-100 GSM8K templates, 200-300 MATH templates

### Two-Stage Compiler Training

**Stage 1 — Template Selection SFT**:
- Input: problem text
- Output: selected template ID + template program structure in DSL
- Training data: (problem, gold_template_program) pairs from trace-template alignment
- LoRA r=16, α=32, lr=2e-4, 3 epochs
- Grammar-constrained decoding ensures only valid DSL programs are generated

**Stage 2 — Variable Filling SFT**:
- Input: problem text + selected template program
- Output: instantiated reasoning steps (variables bound to concrete values)
- Continues from Stage 1 adapter
- LoRA r=16, α=32, lr=1e-4, 3 epochs

### Modern Primitive Usage
- **Qwen3.5-9B as template extractor (teacher role)**: Leverages the model's language understanding to parse CoT traces into structured step graphs. This is more reliable than rule-based parsing because the model understands mathematical reasoning structure.
- **Qwen3.5-9B as compiler (student role)**: The base model provides strong language understanding; LoRA adapters add program generation capability. This is lighter than training a new model from scratch.
- **Grammar-constrained decoding**: Uses FSM-based constrained decoding (outlines library) to ensure all generated programs are syntactically valid DSL programs. This is the key to achieving high compilation coverage.

### Integration into Base Generator
- The compiler is a LoRA adapter on Qwen3.5-9B — no architecture changes
- At inference: compiler generates program → type checker validates → executor fills variables and produces answer
- Fallback: if compilation fails (invalid program or type error), fall back to standard CoT with the base model
- The template bank is a static JSON file loaded at startup — no retrieval overhead

### Training Plan
1. **Data generation** (~24 GPU-hours): Generate and filter CoT traces for GSM8K + MATH
2. **Template extraction** (~36 GPU-hours): Extract, cluster, deduplicate templates
3. **Compiler Stage 1** (~60 GPU-hours): Template selection SFT with torchrun 8-GPU
4. **Compiler Stage 2** (~60 GPU-hours): Variable filling SFT with torchrun 8-GPU
5. **Evaluation** (~48 GPU-hours): Full benchmark evaluation
6. **Ablations** (~48 GPU-hours): Operation ablation, bank size, model size

### Failure Modes and Diagnostics
- **Template extraction too noisy**: Monitor extraction success rate. If <50% parse successfully, improve extraction prompt or use voting across multiple generations. Diagnostic: template_parse_rate metric.
- **Template library too small**: If <30 templates from GSM8K, augment with MATH traces or relax deduplication threshold. Diagnostic: bank_size metric.
- **Compiler generates invalid programs**: Grammar-constrained decoding should prevent this. If compilation coverage <70%, increase training data or add curriculum from simple to complex programs. Diagnostic: compilation_coverage metric.
- **Composed templates don't improve accuracy**: If compositional test accuracy <= CoT+5%, the algebra doesn't help — this is a kill condition. Diagnostic: compositional_accuracy metric.

### Novelty and Elegance Argument
**Closest work**: ReasoningBank (stores and retrieves complete traces), Metacognitive Reuse (compresses patterns into tokens), Buffer of Thoughts (static frameworks).

**Exact difference**: We introduce a **formal algebra** with typed operations, provable properties, and a trained compiler. This is fundamentally different from retrieval (no composition), compression (no structure), and frameworks (no learning).

**Why focused**: One mechanism (template algebra) + one learned component (compiler). No retrieval system, no reinforcement learning, no verifier, no iterative refinement. The algebra provides structure; the compiler provides learning.

## Claim-Driven Validation Sketch

### Claim 1: Compositional Generalization (Primary)
- **Minimal experiment**: Hold out problems requiring template compositions (T_i, T_j) never seen together in training. Compare our compiler's accuracy on these problems vs. CoT, ReasoningBank (retrieval), and few-shot templates.
- **Baselines / ablations**: Standard CoT, 8-shot CoT, BM25-retrieved traces, ReasoningBank, Metacognitive Reuse (reimplemented)
- **Metric**: Accuracy on held-out compositional test set (~500 problems)
- **Expected evidence**: ≥15% accuracy gain over CoT, ≥10% over ReasoningBank

### Claim 2: Token Efficiency
- **Minimal experiment**: Compare total tokens generated per problem across methods at matched or better accuracy.
- **Baselines / ablations**: Same as Claim 1
- **Metric**: Token count per problem, token efficiency (accuracy / tokens)
- **Expected evidence**: ≥40% token reduction vs. standard CoT

### Claim 3: Algebra Necessity (Ablation)
- **Minimal experiment**: Replace COMPOSE with concatenation, remove BRANCH, remove type checking. Measure degradation on compositional test set.
- **Baselines / ablations**: Full system vs. no-COMPOSE, no-BRANCH, no-type-checking variants
- **Metric**: Accuracy degradation on compositional test set
- **Expected evidence**: ≥10% accuracy drop when algebraic operations are ablated

## Experiment Handoff Inputs
- **Must-prove claims**: Compositional generalization advantage, token efficiency, algebra necessity
- **Must-run ablations**: Operation ablation (COMPOSE, BRANCH, type system), bank size sensitivity, compiler model size
- **Critical datasets / metrics**: GSM8K test (1319), MATH test (5000), compositional held-out (~500), accuracy, token count, compilation coverage
- **Highest-risk assumptions**: (1) Template extraction produces enough diverse, high-quality templates, (2) Compiler learns to generate valid programs with high coverage, (3) Compositional test set is non-trivial

## Compute & Timeline Estimate
- **Estimated GPU-hours**: ~1500 (within 2000 budget)
- **Data / annotation cost**: Zero (all data from existing GSM8K/MATH + LLM-generated traces)
- **Timeline**: 6-8 weeks (Stages 1-6 as in PLAN.md)
