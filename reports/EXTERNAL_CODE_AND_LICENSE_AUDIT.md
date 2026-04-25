# External Code and License Audit

## Mandatory Prior Works Audit

| Work | Year/Venue | Mechanism | Code Copied Into Repo? | License | Citation Required? | Baseline Required? |
|---|---|---|---|---|---|---|
| Program of Thoughts | TMLR 2023 | LM→program→interpreter | No | Unknown | Yes | Yes (matched) |
| PAL | ICML 2023 | LM→program→runtime | No | MIT likely | Yes | Yes (matched) |
| Faithful CoT | ACL 2023 | NL→symbolic chain→solver | No | Unknown | Yes | Yes if feasible |
| MARIO | ACL Findings 2024 | Code interpreter output | No | Apache-2.0 | Yes | Compare |
| ToRA | ICLR 2024 | Tool-integrated trajectories | No | MIT | Yes | Yes if feasible |
| Chain of Code | ICML 2024 | Code+LMulator simulation | No | Unknown | Yes | Cite + compare |
| HintMR | arXiv 2026 | Oracle step hints→SLM | No | Unknown | Yes — closest to E | Critical baseline |
| Math-Shepherd | ACL 2024 | Process reward per step | No | Apache-2.0 | Yes | Cite |
| Buffer of Thoughts | NeurIPS 2024 | Reusable thought templates | No | Unknown | Yes | Cite + compare |
| Retrieval-of-Thought | ICLR 2026 | Composable thought graph | No | Unknown | Yes | Cite + compare |
| Composable CoT | arXiv 2025 | Composable CoT format | No | Unknown | Yes | Cite |

## Verdict

**NO_CODE_COPIED** — All main method code (`src/dataflow_plan.py`, training/eval scripts)
is original to this project. External works used only as library dependencies (PyTorch,
Transformers, TRL, PEFT) under permissive licenses. Mandatory citation list above must
be included in paper related work section.

## Novelty Differentiation Required

V-GIFT must be differentiated from:
1. PAL/PoT/ToRA: V-GIFT uses **typed reusable subroutine DAG**, not one-off full programs
2. HintMR: V-GIFT uses **executable typed dataflow + consistency verification**, not generic step hints
3. Chain of Code: V-GIFT uses **deterministic DSL execution**, not LM-simulated pseudocode
4. BoT/RoT: V-GIFT uses **executable numerical dataflow**, not text thought templates
