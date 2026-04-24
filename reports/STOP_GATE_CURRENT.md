# STOP GATE — Do NOT Run Full Benchmark

## Status: BLOCKED

Per GPT-5.5 Pro second review (Decision B), full benchmark is blocked until:

1. `true_dataflow > 0` in GIFT training data (currently 0)
2. `active_binding_rate > 70%` measured (currently 0 tested)
3. B ablation implemented (currently missing)
4. Eval reliability fixed (`--no_fallback`, `--require_adapters`)
5. Training reproducibility fixed (`--seed`, `--no_resume`, hashes)
6. 3-seed minimal A/B/C passes on same data
7. README/PAPERS integrity cleaned

## Current Blocking Evidence

| Gate | Required | Actual | Status |
|------|----------|--------|--------|
| GIFT coverage | >30% | 20.2% | FAIL |
| true_dataflow plans | >0 | 0 | FAIL |
| active_binding_tested | >0 | 0 | FAIL |
| B ablation | exists | missing | FAIL |
| eval no_fallback | implemented | not implemented | FAIL |
| training reproducibility | implemented | not implemented | FAIL |
| C > A | required | C=7.0% < A=29.5% | FAIL |

## Forbidden Actions Until Gates Pass

- Running full GSM8K/MATH benchmark
- Writing paper claims about GIFT superiority
- Running 3-seed production experiments
- Claiming "mechanism works" based on 99.5% parse rate
- Making any SOTA claims

## Allowed Actions

- Implementing the 10 tasks from NEXT_GPT55_REVIEW_PACKAGE.md
- Running small debugging experiments (N≤100) for mechanism verification
- Cleaning up README, PAPERS.md
- Committing missing artifacts
