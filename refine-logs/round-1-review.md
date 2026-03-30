# Round 1 Review (GPT-5.4)

## Scores
| Dimension | Score |
|-----------|-------|
| Problem Fidelity | 8/10 |
| Method Specificity | 5/10 |
| Contribution Quality | 6/10 |
| Frontier Leverage | 8/10 |
| Feasibility | 6/10 |
| Validation Focus | 6/10 |
| Venue Readiness | 6/10 |
| **Overall** | **6.4/10** |

## Verdict: REVISE

## Key Issues

### CRITICAL: Method Specificity (5/10)
- Core interfaces underdefined: "typed step graph," "type checker," "alignment," "structural isomorphism" are placeholders
- Unknown: node vocabulary, type ontology, supervision source for programs, execution path from filled template to answer
- Token-efficiency claim unsupported (model may regenerate substantial free-form reasoning)
- **Fix**: Replace arbitrary DAGs with small executable AST/DSL. Define 6-10 operators and closed type set. Generate gold programs as structured output from teacher LLM, filter by type+answer correctness, train on exact AST targets.

### CRITICAL: Validation Focus (6/10)
- Held-out compositional split underdefined
- Without strict split construction, reviewers will suspect lexical/difficulty mismatch
- **Fix**: Define split over unseen template bigrams/trigrams in normalized ASTs, difficulty-match train/test, report leakage audits.

### IMPORTANT: Contribution Quality (6/10)
- Trying to sell both broad formal algebra AND practical reasoning system
- Formal claims overextended (SPECIALIZE∘ABSTRACT ≈ id is weak in noisy NL pipeline)
- Formalism-first feels rather than mechanism-first
- **Fix**: Narrow to typed reusable template programs compiled under constrained decoding. Drop inverse-style claims. Remove BRANCH unless benchmark requires it.

### IMPORTANT: Feasibility (6/10)
- Hard part is template induction + leakage-free benchmarks, not LoRA training
- MATH traces from 9B model may be thin
- Graph clustering over noisy parsed traces is brittle
- **Fix**: Narrow to GSM8K + one MATH subfamily. Canonicalize as normalized ASTs. Use stronger teacher offline for pseudo-labels.

### IMPORTANT: Venue Readiness (6/10)
- Risks reading as "symbolic scaffold on top of LLM traces"
- **Fix**: Narrow story: one benchmark, one minimal DSL, one compiler, one decisive OOD composition result.

## Simplification Opportunities
1. Delete DAG generality → linear or tree-structured programs
2. Merge SEQUENCE into repeated COMPOSE; keep operator set minimal
3. Collapse two LoRAs into one unless staged training clearly needed

## Modernization Opportunities
1. Replace post-hoc trace parsing with direct structured program distillation from teacher
2. Use schema-constrained JSON/AST generation instead of NL-to-graph parsing
3. Execute filled programs directly for arithmetic, verbalize optionally

## Drift Warning: NONE

<details>
<summary>Full Raw Response</summary>

This is directionally strong on problem choice but not yet paper-ready. The core issue is not contribution sprawl; it is that the main artifact, the template language plus supervision pipeline, is still described at the slogan level rather than the implementation level.

[Full response captured above in scores and analysis]

</details>
