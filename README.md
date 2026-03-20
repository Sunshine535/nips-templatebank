# Template Algebra: Formal Algebraic Operations on Reasoning Templates

## Overview

Chain-of-thought reasoning generates reasoning traces from scratch for every
problem, wasting tokens on recurring reasoning patterns. Existing template reuse
methods — ReasoningBank (memory retrieval), Metacognitive Reuse (behavior
compression) — treat templates as **opaque units** to be retrieved or compressed.

Template Algebra defines **formal operations** (compose, abstract, specialize)
on CoT reasoning templates and trains a **template compiler** that decomposes
new problems into executable template programs.

**This is NOT template retrieval** (that's ReasoningBank). We define a formal
**algebra** on template structures and perform **program synthesis** over it.

**Target venue:** NeurIPS 2026
**Status:** Rewrite of pilot → full algebraic + compiler framework

## Core Contribution

```
Training Phase: Template Extraction
──────────────────────────────────────────────────
GSM8K/MATH Reasoning Traces
    │
    ▼
┌──────────────────────────────────┐
│  Template Extractor (Qwen3.5-27B)│
│  Trace → Abstract Template       │
│  (variable slots, step graph)    │
└──────────┬───────────────────────┘
           │
           ▼
┌──────────────────────────────────┐
│  Template Algebra Definition     │
│                                  │
│  ⊕ COMPOSE(T₁, T₂)            │  Chain T₁ output → T₂ input
│  ↑ ABSTRACT(T, vars)           │  Generalize: bind concrete → variables
│  ↓ SPECIALIZE(T, vals)         │  Instantiate: variables → concrete
│  ∘ SEQUENCE(T₁, ..., Tₙ)      │  Multi-step template program
│  ⊗ BRANCH(cond, T₁, T₂)      │  Conditional template selection
│                                  │
│  Properties:                     │
│  • Compose is associative        │
│  • Abstract/Specialize are inverse│
│  • Type system prevents invalid  │
│    compositions                  │
└──────────┬───────────────────────┘
           │
           ▼
    Template Library with Algebraic Structure

Inference Phase: Template Compilation
──────────────────────────────────────────────────
New Problem
    │
    ▼
┌──────────────────────────────────┐
│  Template Compiler (Qwen3.5-9B)  │
│  Problem → Template Program      │
│                                  │
│  Output:                         │
│    SEQ(                          │
│      SPECIALIZE(T_parse, {x=..})│
│      BRANCH(is_multi_step?,     │
│        COMPOSE(T_arith, T_unit),│
│        SPECIALIZE(T_direct, ..) │
│      ),                          │
│      T_verify                    │
│    )                             │
└──────────┬───────────────────────┘
           │
           ▼
    Execute Template Program → Answer
```

## Key Differentiators

| Aspect | ReasoningBank | Metacognitive Reuse | Framework of Thoughts | **Ours** |
|--------|--------------|--------------------|-----------------------|----------|
| Approach | Memory retrieval | Behavior compression | Structured prompting | **Program synthesis** |
| Template model | Opaque traces | Compressed behaviors | Predefined frameworks | **Formal algebra** |
| Composition | None (retrieve one) | None | Manual chaining | **Algebraic compose** |
| Generalization | Similarity-based | Pattern matching | N/A | **Abstract/Specialize** |
| Theory | None | 46% token reduction | None | **Associativity, type system** |
| Novelty test | Fails on novel structure | Degrades on novel | Requires human design | **Compiles from algebra** |

## Models & Hardware

| Component | Model | Role |
|-----------|-------|------|
| Template Extractor | Qwen3.5-27B | Extract abstract templates from reasoning traces |
| Template Compiler | Qwen3.5-9B | Compile problems into template programs |
| Executor | Qwen3.5-9B | Execute template programs to produce answers |
| **Hardware** | **8× A100-80GB** | Extraction, compiler training, evaluation |

## Template Algebra: Formal Definition

### Template Structure
A template T = (V, E, Slots, Types) where:
- V = set of reasoning step nodes
- E = directed edges (step dependencies)
- Slots = named variable placeholders in each step
- Types = type annotations for each slot (number, expression, entity, etc.)

### Operations

| Operation | Signature | Semantics |
|-----------|-----------|-----------|
| COMPOSE(T₁, T₂) | T × T → T | Connect T₁ output slots to T₂ input slots (type-matched) |
| ABSTRACT(T, bindings) | T × Map → T | Replace concrete values with typed variable slots |
| SPECIALIZE(T, values) | T × Map → T | Bind variable slots to concrete values |
| SEQUENCE(T₁, ..., Tₙ) | T* → T | Chain templates in order, passing intermediate results |
| BRANCH(cond, T₁, T₂) | Cond × T × T → T | Conditional selection based on problem features |

### Algebraic Properties
1. **Associativity:** COMPOSE(COMPOSE(T₁, T₂), T₃) = COMPOSE(T₁, COMPOSE(T₂, T₃))
2. **Inverse:** SPECIALIZE(ABSTRACT(T, b), b) ≈ T (up to variable renaming)
3. **Type safety:** COMPOSE(T₁, T₂) is defined iff output types of T₁ match input types of T₂
4. **Identity:** There exists T_id such that COMPOSE(T, T_id) = T

## Repository Structure

```
nips-templatebank/
├── README.md              # This file
├── PROPOSAL.md            # Falsifiable thesis and success criteria
├── PLAN.md                # Stage-gate execution plan (8 weeks)
├── PAPERS.md              # Core references with URLs and annotations
├── EXPERIMENTS.md         # Evaluation protocol and results
├── environment.yml        # Conda environment spec
├── requirements.txt       # Pip dependencies
├── scripts/
│   ├── extract_templates.py       # Template extraction from traces
│   ├── train_compiler.py          # Template compiler training
│   ├── eval_compositional.py      # Compositional generalization eval
│   └── run_templatebank_pilot.py  # Legacy pilot script
├── src/
│   ├── algebra/                   # Core algebraic operations
│   │   ├── template.py            # Template data structure
│   │   ├── compose.py             # COMPOSE operation
│   │   ├── abstract.py            # ABSTRACT operation
│   │   ├── specialize.py          # SPECIALIZE operation
│   │   ├── sequence.py            # SEQUENCE combinator
│   │   ├── branch.py              # BRANCH combinator
│   │   └── type_system.py         # Type checking for slot compatibility
│   ├── extraction/                # Template extraction pipeline
│   │   ├── trace_parser.py        # Parse reasoning traces into step graphs
│   │   ├── template_extractor.py  # Qwen3.5-27B extraction wrapper
│   │   └── template_library.py    # Template library management
│   ├── compiler/                  # Template compiler
│   │   ├── compiler_model.py      # Qwen3.5-9B compiler wrapper
│   │   ├── program_ast.py         # Template program AST
│   │   └── executor.py            # Program executor
│   └── eval/                      # Evaluation suite
│       ├── compositional_eval.py  # Compositional generalization metrics
│       └── baselines.py           # CoT, retrieval, compression baselines
├── configs/
│   ├── extraction_config.yaml     # Extraction hyperparameters
│   ├── compiler_config.yaml       # Compiler training config
│   └── eval_config.yaml           # Evaluation settings
└── results/
```

## Quick Start

```bash
conda env create -f environment.yml
conda activate nips_template_algebra

# Stage 1: Extract templates from GSM8K/MATH traces
python scripts/extract_templates.py \
    --traces data/gsm8k_cot_traces.jsonl \
    --extractor Qwen/Qwen3.5-27B \
    --output results/template_library/

# Stage 2: Train template compiler
torchrun --nproc_per_node=8 scripts/train_compiler.py \
    --model Qwen/Qwen3.5-9B \
    --template-library results/template_library/ \
    --epochs 10

# Stage 3: Evaluate compositional generalization
python scripts/eval_compositional.py \
    --compiler results/compiler_checkpoint/ \
    --benchmarks gsm8k,math,aime,olympiad
```

## Evaluation Protocol

### In-Distribution (Template Coverage)
- GSM8K test set: problems whose structures are covered by extracted templates
- MATH test set: problems from seen categories
- Metric: accuracy, token efficiency (tokens used / CoT tokens)

### Compositional Generalization (Key Test)
- Held-out problem types that require **novel combinations** of known templates
- e.g., Train on "arithmetic + unit conversion" and "geometry + algebra" separately,
  test on "arithmetic + geometry" combinations never seen in training
- Metric: accuracy on novel compositions vs baselines
- This is the CRITICAL test — if the algebra enables true composition, performance
  on novel combinations should far exceed baselines

### Out-of-Distribution (New Domains)
- AIME / AMC problems (competition math, unseen difficulty)
- Olympiad-level problems requiring multi-step novel reasoning
- Metric: accuracy with and without template-based reasoning

## Success Criteria

- **Primary:** On compositional generalization test (novel template combinations),
  achieve >= 15% absolute accuracy gain over standard CoT and >= 10% over
  ReasoningBank-style retrieval
- **Secondary:** Token efficiency >= 40% reduction vs CoT at matched accuracy
  (comparable to Metacognitive Reuse's 46% but with better accuracy)
- **Tertiary:** Compiler successfully decomposes >= 80% of test problems into
  valid template programs (syntactically well-typed)

## Baselines

1. **Standard CoT** — zero-shot chain-of-thought prompting
2. **Few-shot CoT** — with exemplars selected by similarity
3. **ReasoningBank** — retrieve most similar reasoning trace from memory
4. **Metacognitive Reuse** — compressed behavior patterns (46% token reduction)
5. **Framework of Thoughts** — manually designed reasoning frameworks
6. **Buffer of Thoughts** — thought template retrieval and instantiation

## Key References

- ReasoningBank (2025) — memory-based reasoning trace retrieval
- Metacognitive Reuse (2025) — behavior compression, 46% token reduction
- Framework of Thoughts — structured reasoning with predefined frameworks
- Buffer of Thoughts (NeurIPS 2024) — thought template buffer

See [PAPERS.md](PAPERS.md) for full annotated reference list.

## License

Research code for academic use.
