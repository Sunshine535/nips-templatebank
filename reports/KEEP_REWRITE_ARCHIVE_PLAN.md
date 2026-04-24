# Keep / Rewrite / Archive Plan

| Item | Path | Current Role | Action | Reason | Risk |
|------|------|-------------|--------|--------|------|
| DSL primitives | `src/template_dsl.py` | Executor, Program, Step | KEEP | Foundation works | None |
| CompositionExecutor | `src/template_dsl.py` | Old implicit executor | KEEP AS ABLATION | Implicit binding is the diagnosed problem | Comparison baseline |
| GIFT dataflow | `src/dataflow_plan.py` | New explicit executor | KEEP (new main) | Implements missing mechanism | Core |
| MCD split | `src/mcd_split.py` | Compound extraction | REWRITE | Adjacency compounds not real flow | Need true dataflow |
| RLVR evolution | `src/rlvr_evolution.py` | Library evolution | FREEZE AS ABLATION | Built on weak semantics | Resume after GIFT |
| Test-time tools | `src/test_time_tools.py` | Failure repair | FREEZE AS ABLATION | Heuristic, leakage risk | Later |
| Template algebra | `src/template_algebra.py` | Dead code | ARCHIVE | Not imported anywhere | None |
| Extraction | `scripts/extract_templates.py` | Program extraction | KEEP | Works (697 verified) | None |
| train_seval.py | `scripts/train_seval.py` | SFT+GRPO flat | KEEP AS BASELINE | Produces 29.5% result | Baseline |
| train_template_compiler.py | `scripts/train_template_compiler.py` | Old SFT | REWRITE | Needs seed/resume fixes | Reproducibility |
| eval script | `scripts/eval_template_reasoning.py` | Evaluation | REWRITE | Fallback masking, no adapter check | Metric reliability |
| Verified programs | `results/templates_verified/all_programs.json` | Source data | KEEP | 697 valid programs | None |
| Old compose_train | `results/templates_verified/compose_train.json` | Old labels | KEEP AS NEGATIVE EVIDENCE | 98% empty bindings | Do not train on |
| Old flat_train | `results/templates_verified/flat_train.json` | Old baseline | KEEP AS NEGATIVE EVIDENCE | Contaminated | Do not train on |
| GIFT data | `results/gift/` | New faithful data | KEEP | 141 faithful plans | Coverage low |
| SFT/GRPO results | `results/seval/` | Recent results | KEEP | Valid experiments | Only seed 42 |
| Old eval | `results/eval_v2/` | Historical | KEEP AS DIAGNOSTIC | Confirms binding failure | None |
| MCD split 27B | `results/mcd_split_27b.json` | Old split | ARCHIVE | Synthetic, bad atom_tvd | None |
| Pilot results | `results/templatebank_pilot_*.json` | Old pilot | ARCHIVE | Weak, irreproducible | None |
| README claims | `README.md` | Public claims | REWRITE | Unsupported claims | Must weaken |
| PAPERS.md | `PAPERS.md` | Related work | MARK UNVERIFIED | Placeholder IDs | Academic integrity |
| configs/gift_minimal.yaml | `configs/gift_minimal.yaml` | GIFT config | KEEP | New method config | None |
