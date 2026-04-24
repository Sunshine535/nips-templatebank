# Unreliable Results Manifest

All raw files are preserved in their original locations. This manifest labels their reliability status.

| Result | Path | Status | Reason |
|--------|------|--------|--------|
| Old compose eval seed42 | `results/eval_v2/eval_results_seed42.json` | **diagnostic negative** | compose 2% GSM8K, 0% MCD — confirms binding failure |
| Verified programs | `results/templates_verified/all_programs.json` | **valid source data** | 697/697 exec+answer correct |
| Old compose_train | `results/templates_verified/compose_train.json` | **contaminated** | 98% empty bindings, mismatched templates |
| Old flat_train | `results/templates_verified/flat_train.json` | **contaminated** | Inlines mismatched library representative |
| Subroutine library | `results/templates_verified/subroutine_library.json` | **usable with caution** | Weak interfaces, only 3/16 used in training |
| MCD split v2 | `results/mcd_split_v2.json` | **partially valid** | atom_tvd good, compounds from adjacency not real flow |
| MCD split 27B | `results/mcd_split_27b.json` | **unreliable** | synthetic_used=true, atom_tvd 0.11 above threshold |
| Extract 27B 500 | `results/extract_27b_500/` | **historical** | Early extraction, superseded |
| Pilot static/dynamic | `results/templatebank_pilot_*.json` | **weak mixed** | reuse up but accuracy flat, low reproducibility |
| Pipeline 4% report | `PIPELINE_REPORT.md` | **historical** | N=50 synthetic, unreliable |
| SFT train eval | `results/seval/sft_seed42/eval_results.json` | **valid but overfit** | 84% on training data, 29.5% on test |
| GRPO train eval | `results/seval/grpo_seed42/eval_results.json` | **valid** | Same as SFT, confirms GRPO no improvement |
| GSM8K test 3-way | `results/seval/gsm8k_test/results.json` | **valid** | base 0%, SFT 29.5%, GRPO 29.5% |
| GIFT plan audit | `results/gift/plan_audit.json` | **valid** | 141/697 faithful (20.2%) |
| GIFT eval | `results/gift/gsm8k_test/results.json` | **valid** | 7% accuracy, 99.5% parse |
| PAPERS.md citations | `PAPERS.md` | **unreliable** | Contains placeholder arXiv IDs |
| Template algebra | `src/template_algebra.py` | **dead code** | Not imported by main pipeline |
