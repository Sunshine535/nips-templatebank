# Experiments: TemplateBank++ (Revised v2)

## Benchmarks
- GSM8K, MATH, BBH, StrategyQA, Game of 24.
- Cross-family OOD transfer split.

## Baselines
- Greedy CoT.
- Self-Refine.
- Static prompt template library.
- Buffer-of-Thoughts style memory baseline.

## Metrics
- Accuracy.
- Avg output tokens and latency.
- Template reuse rate.
- Template validity rate.
- OOD transfer delta.

## Statistical Protocol
- 3 replications minimum.
- Paired bootstrap on accuracy and cost-normalized score.

## NeurIPS Minimum Publishable Standard
- `>= +2` absolute accuracy at matched cost on at least two datasets.
- Demonstrated benefit of dynamic memory over static bank.
- Full template-id and instantiation log release.

## Current Status
- Pilot implementation and first result are now available.

## Implemented Pilot (2026-02-27)
- Script:
  - `methods/06_templatebank_pp/scripts/run_templatebank_pilot.py`
- Command:
  ```bash
  python methods/06_templatebank_pp/scripts/run_templatebank_pilot.py
  ```
- Input:
  - `methods/01_adathink/results/per_sample_Qwen3_8B_20260227_140410.csv`
- Output:
  - `methods/06_templatebank_pp/results/templatebank_pilot_20260227_150036.json`

## Pilot Snapshot
- Static memory:
  - acc `0.50`
  - avg tokens `39.35`
  - utility `0.4769`
  - reuse rate `0.65`
- Dynamic memory:
  - acc `0.50`
  - avg tokens `39.35`
  - utility `0.4769`
  - reuse rate `0.70`
- Baselines:
  - fixed64 acc `0.35`
  - fixed128 acc `0.40`
  - fixed256 acc `0.525`

## Limitation
- Current template extraction is coarse lexical signatures, not full trace-to-template abstraction and constrained instantiation.
