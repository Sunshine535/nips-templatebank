# Local Repository Scan

## Top-Level Directory Map
```
.
├── configs/           — experiment configs
├── idea-stage/        — idea discovery outputs
├── refine-logs/       — method refinement history
├── reports/           — execution reports (this scan)
├── research-wiki/     — literature notes
├── results/           — experiment results
├── review-stage/      — auto-review loop state
├── scripts/           — training/eval/extraction scripts
├── src/               — core method implementation
├── tests/             — unit tests
└── [root docs]        — README, proposals, reviews
```

## Component Table

| Component | Path | Purpose | Importance | Notes |
|-----------|------|---------|------------|-------|
| DSL core | `src/template_dsl.py` | Program, Step, Subroutine, CompositionExecutor | Critical | Implicit binding is the core problem |
| MCD split | `src/mcd_split.py` | Atom/compound split for compositional eval | Critical | Compounds from adjacency, not real flow |
| MCTS search | `src/mcts_search.py` | Tree search over plans | Low | Not in main pipeline |
| RLVR evolution | `src/rlvr_evolution.py` | Library evolution via GRPO | Medium | Freeze; ablation only |
| Test-time tools | `src/test_time_tools.py` | Failure repair heuristics | Medium | Freeze; ablation only |
| Template algebra | `src/template_algebra.py` | Aspirational typed algebra | Low | Dead code; archive |
| Teacher extraction | `scripts/extract_templates.py` | Extract verified programs | Critical | Recently fixed (binding permutation) |
| Batched extraction | `scripts/extract_vllm.py` | Fast HF batch extraction | Medium | Created this session |
| Student training | `scripts/train_template_compiler.py` | SFT compose/flat/CoT | High | Needs reproducibility fixes |
| SEVAL training | `scripts/train_seval.py` | SFT+GRPO flat programs | High | Rewritten this session; working |
| Evaluation | `scripts/eval_template_reasoning.py` | compose/flat/CoT eval | Critical | Fallback masking, adapter issues |
| MCD split builder | `scripts/build_mcd_split.py` | Construct MCD splits | High | Needs true-flow compounds |
| Config | `configs/template_config.yaml` | All experiment params | High | Needs GIFT config |
| Verified programs | `results/templates_verified/all_programs.json` | 697 verified GSM8K programs | High | Source data; KEEP |
| Old compose labels | `results/templates_verified/compose_train.json` | 418 training plans | Critical | BROKEN: empty bindings, mismatched |
| Old flat labels | `results/templates_verified/flat_train.json` | 418 flat baselines | Critical | BROKEN: contaminated |
| Old eval | `results/eval_v2/eval_results_seed42.json` | seed42 compose 2%, flat 5% | High | Historical negative; KEEP |
| New SFT eval | `results/seval/sft_seed42/eval_results.json` | SFT 84% train | Medium | Overfitting evidence |
| New GSM8K test | `results/seval/gsm8k_test/results.json` | base 0%, SFT 29.5%, GRPO 29.5% | High | Valid recent result |
| Subroutine library | `results/templates_verified/subroutine_library.json` | 16 templates | High | Weak interfaces |
| MCD split v2 | `results/mcd_split_v2.json` | Current split | Medium | Compounds semantically weak |
| MCD split 27B | `results/mcd_split_27b.json` | Old synthetic split | Low | Archive |
| Pilot results | `results/templatebank_pilot_*.json` | Memory pilot | Low | Archive |
| Unit tests | `tests/test_templatebank.py` | DSL tests | High | Encode wrong composition behavior |
| README | `README.md` | Public claims | High | Unsupported claims |
| Proposals/Reviews | `PROPOSAL.md`, `REVIEW_SEVAL_V2.md`, etc. | Historical docs | Medium | Keep as reference |
