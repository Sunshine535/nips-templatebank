# Prior Work Implementation Boundary

## Main Method Code Provenance

| File / Directory | Role | External Source? | License Risk? | Safe for Main Method? | Action |
|---|---|---|---|---|---|
| `src/dataflow_plan.py` | Main method | No — written in this project | No | Yes | KEEP |
| `src/template_dsl.py` | Main method foundation | No — original project code | No | Yes | KEEP |
| `src/mcd_split.py` | Evaluation utility | "based on" refers to call graph structure concept, not copied code | No | Yes | KEEP |
| `src/rlvr_evolution.py` | Ablation only | "based on" TRL's GRPOTrainer = uses library API, not copied | No | Yes as ablation | FREEZE |
| `scripts/train_ablation.py` | Training | No external code copied | No | Yes | KEEP |
| `scripts/eval_ablation.py` | Evaluation | No external code copied | No | Yes | KEEP |
| `scripts/build_gift_step_primitives.py` | Data builder | No external code | No | Yes | KEEP |
| `scripts/audit_gift_mechanism.py` | Audit tool | No external code | No | Yes | KEEP |
| `scripts/check_value_leakage.py` | Audit tool | No external code | No | Yes | KEEP |

## External Dependencies (used as libraries, NOT copied into main method)

| Dependency | Role | License | How Used | Risk |
|---|---|---|---|---|
| PyTorch | Training framework | BSD-3 | Library import | None |
| Transformers | Model loading | Apache-2.0 | Library import | None |
| TRL | GRPO/SFT trainer | Apache-2.0 | Library import | None |
| PEFT | LoRA adapters | Apache-2.0 | Library import | None |
| datasets | Data loading | Apache-2.0 | Library import | None |

## Prior Work That Must Be Cited (NOT copied into code)

| Work | Why Cite | Code Copied? | Status |
|---|---|---|---|
| PAL (Gao et al., 2023) | Flat program baseline concept | No | Must cite |
| PoT (Chen et al., 2023) | Program-of-thought concept | No | Must cite |
| Faithful CoT (Lyu et al., 2023) | Faithfulness concept | No | Must cite |
| ToRA (Gou et al., 2024) | Tool-integrated reasoning | No | Must cite |
| MARIO (Liao et al., 2024) | Code interpreter output | No | Must cite |
| Chain of Code (Li et al., 2024) | LM-simulated execution | No | Must cite |
| HintMR (2026) | Oracle step hints for SLM | No | Must cite — closest to E |
| Math-Shepherd (Wang et al., 2024) | Process supervision | No | Must cite |
| Buffer of Thoughts (Yang et al., 2024) | Reusable templates | No | Must cite |
| Retrieval-of-Thought (2026) | Composable thought graph | No | Must cite |
| DreamCoder/LILO | Library learning | No | Must cite |

## Verdict

**NO_PROVENANCE_RISK_DETECTED** — All main method code is original project code.
No external code was copied into `src/` or main `scripts/`. External tools are
used as library imports only. Prior work must be cited but was not copied.
