# Round 2 Review (GPT-5.4)

## Scores
| Dimension | Score |
|-----------|-------|
| Problem Fidelity | 8.7/10 |
| Method Specificity | 8.1/10 |
| Contribution Quality | 7.3/10 |
| Frontier Leverage | 7.4/10 |
| Feasibility | 8.4/10 |
| Validation Focus | 8.2/10 |
| Venue Readiness | 7.1/10 |
| **Overall** | **7.9/10** |

## Verdict: REVISE

## Key Remaining Issues

### Composition must be a real inference-time mechanism
- Current: compiler emits flat AST step lists → "COMPOSE" is just a label
- Fix: Compiler output = sequence of `template_id + slot_bindings`, expanded into AST after decoding
- Template library must be the actual inference-time object

### Training signal too weak for unseen composition
- Single-reference AST SFT is one-to-many mismatch
- Fix: Multiple valid teacher programs per problem, execution-guided reranking

### Novelty positioning
- Must distinguish clearly from PAL/PoT/program-synthesis
- Narrow claim to "template-level compositional reuse," not "typed DSL"

### Metrics need refinement
- Report compiler-only accuracy vs fallback rate separately
- Add error analysis: wrong template selection / wrong binding / wrong execution
