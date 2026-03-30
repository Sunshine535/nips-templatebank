# AUTO REVIEW

**Date**: 2026-03-31
**Reviewer**: Codex
**Score**: 3/10

Overall assessment: the repository has a reasonable scaffold and some core components are readable, but it is not yet ready to support NeurIPS-grade experimental claims. The main blockers are methodological mismatches between the README/config and the actual code, silent synthetic fallbacks, incomplete evaluation wiring, and only partial support for multi-GPU execution and resume.

## Major Findings

### 1. Experimental protocol is incomplete and partially disconnected

- `scripts/run_all_experiments.sh:68-75` builds an MCD split, but `scripts/eval_template_reasoning.py:308-339` and `scripts/eval_template_reasoning.py:345-355` never use `--split_path`; evaluation runs on the raw benchmark `test_split` instead of the generated MCD split.
- `configs/template_config.yaml:85-102` advertises `cot_budget`, `retrieval_compose`, and `num_seeds: 3`, but `scripts/eval_template_reasoning.py:363-390` only executes `compose`, `flat_inline`, and `direct_cot` for a single seed.
- `scripts/run_all_experiments.sh:117-125` marks the library-size ablation stage complete after creating empty directories. There is no ablation logic.

### 2. Data integrity and result validity are not reliable enough

- `scripts/extract_templates.py:103-105` silently falls back to synthetic benchmark data if dataset loading fails.
- `scripts/train_template_compiler.py:50-53` silently trains on synthetic data if training files are missing.
- `scripts/extract_templates.py:150-161` accepts any executable program; it never checks `exec_result` against the gold `answer`, even though the README claims correct executable filtering.
- `scripts/extract_templates.py:206-233` and `scripts/extract_templates.py:236-263` implement a much weaker method than claimed: library mining is by operator sequence only, and every "composition plan" is reduced to a single library call.
- `src/mcd_split.py:98-150` exposes `min_unseen_compounds` but never enforces it. Combined with single-call plans and compound extraction that largely reduces to binding keys and positions (`src/mcd_split.py:39-58`), the split is not yet a convincing compositional benchmark.

### 3. Multi-GPU support is partial, not end-to-end

- Positive: Stage 3 training is set up to use `torchrun` (`scripts/run_all_experiments.sh:78-99`), and the trainer places each rank on `LOCAL_RANK` (`scripts/train_template_compiler.py:122-128`).
- Negative: extraction and evaluation are not implemented as distributed multi-GPU workloads. They load a single model with `device_map="auto"` and iterate serially over the dataset (`scripts/extract_templates.py:385-390`, `scripts/eval_template_reasoning.py:365-388`), which is not the same as data-parallel experiment execution.
- `scripts/gpu_utils.sh:8-31` hard-fails when `nvidia-smi` is unavailable, so even the smoke path is GPU-only.

### 4. Checkpoint resume support is only partial

- Positive: the trainer auto-resumes from the latest `checkpoint-*` directory (`scripts/train_template_compiler.py:173-179`), and the top-level pipeline skips completed phases via marker files (`scripts/run_all_experiments.sh:17-25`).
- Negative: resume is coarse-grained. Extraction and evaluation restart whole stages after interruption, and phase markers do not verify that outputs are complete or consistent before skipping.

### 5. Code quality is mixed and reproducibility is weak

- I found no test suite in the repository. Static validation is limited.
- The repo still contains stale code paths. For example, `scripts/eval_template_algebra.py:149-175` expects config keys (`config["extraction"]`, `config["evaluation"]["test_datasets"]`) that do not exist in the current config.
- The checked-in pilot path is not runnable as-is: `README_RUN.md:3-8`, `EXPERIMENTS.md:32-42`, and `scripts/run_templatebank_pilot.py:11` / `scripts/run_templatebank_pilot.py:170-176` point to missing `methods/...` files. Running `python3 scripts/run_templatebank_pilot.py` currently fails with `FileNotFoundError`.
- The current `results/` directory only contains pilot JSON snapshots, not outputs from the advertised full pipeline.

## Strengths

- The repository structure is clean and the intent of each stage is understandable.
- Core local components are at least syntactically valid: `python3 -m compileall src scripts` passed.
- A minimal local smoke check of the DSL executor and MCD splitter succeeded.

## Readiness Verdict

Not ready to produce paper-quality experimental results. In its current state, I would view this repository as a prototype scaffold rather than a reproducible experiment package.

## Actionable Feedback

1. Remove silent synthetic fallbacks from the default pipeline. Keep synthetic data only behind explicit opt-in flags, and record that mode in output metadata.
2. Make Stage 1 correctness-preserving: generate multiple candidates per example, execute them, and retain only programs whose output matches the gold answer.
3. Rework subroutine mining and planner targets so plans can contain multiple calls. Then use those plans to build a real MCD split and enforce the configured unseen-compound threshold.
4. Wire the generated split into training and evaluation. If the paper claims MCD generalization, the evaluation script must actually consume `results/mcd_split.json`.
5. Finish the evaluation matrix: implement `cot_budget`, `retrieval_compose`, multi-seed runs, and actual library-size ablations with saved per-seed metrics.
6. Decide what "multi-GPU support" means operationally. If only training is distributed, say so; otherwise add true distributed extraction/evaluation or batched sharded inference.
7. Add a small reproducibility suite: unit tests for the DSL/executor and split logic, plus a tiny checked-in fixture dataset that exercises the full smoke path without external private files.
8. Delete or repair stale scripts and docs so the repository has one authoritative path to reproduce results.
