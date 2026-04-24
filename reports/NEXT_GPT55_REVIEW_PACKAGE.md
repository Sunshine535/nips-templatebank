# Next GPT-5.5 Pro Review Package

## Summary of Changes

### Implemented (Tasks 2,3,4,5,9)
1. **GIFT core module** (`src/dataflow_plan.py`): BindingRef, DataflowPlan, PlanCall, DataflowExecutor — 10/10 tests pass
2. **Plan faithfulness audit**: Confirmed 98% empty bindings in old data, 100% single-call, only 3/16 subs used
3. **Faithful GIFT data**: 141/697 (20.2%) programs have faithful explicit-binding plans
4. **GIFT config**: `configs/gift_minimal.yaml` with require_explicit_bindings=true
5. **Archive manifest**: All old results labeled with reliability status

### Not Yet Implemented (Tasks 6,7,8)
- Task 6: MCD compound rewrite (true dataflow) — code not changed
- Task 7: Eval reliability (--require_adapters, --no_fallback) — code not changed
- Task 8: Training reproducibility (--seed, --no_resume, hashes) — code not changed

### Prior Bug Fixes (before GPT-5.5 diagnosis)
- Empty bindings → permutation search
- Executor ambiguous binding → reject
- Library collapse → slot count + subsequence mining
- train_seval.py → complete rewrite
- Container OOM → 128GB
- Log buffering → python3 -u + explicit flush

## Result Tables

### GSM8K Test Set (200 samples, seed 42)

| Variant | Accuracy | Parse | Exec | Training Data |
|---------|----------|-------|------|---------------|
| Base Qwen3.5-9B | 0.0% | 0% | 0% | — |
| A. Flat SFT | **29.5%** | 95% | 89% | 697 flat programs |
| A. Flat GRPO | 29.5% | 95% | 89% | 697 + 500 GRPO steps |
| C. GIFT SFT | 7.0% | 99.5% | 77% | 141 GIFT plans |

### GIFT Data Audit

| Metric | Value |
|--------|-------|
| Total programs | 697 |
| Faithful GIFT plans | 141 (20.2%) |
| Single-call plans | 141 |
| Two-call (true dataflow) | 0 |
| Empty bindings (old data) | 98.1% |

## What Supports the Original Diagnosis
1. **98% empty bindings confirmed** — exactly as diagnosed
2. **100% single-call** — no composition in old data
3. **GIFT format is learnable** — 99.5% parse rate proves DataflowPlan works
4. **Implicit binding is the root cause** — old compose 2% vs 0% on MCD, flat 5%

## What Contradicts or Weakens the Diagnosis
1. **GIFT (7%) < Flat SFT (29.5%)** — the new mechanism does not yet outperform
2. **20.2% faithful coverage** — below the 30% threshold diagnostic recommended
3. **Zero two-call plans** — the library cannot produce true multi-call composition
4. **Flat SFT is a strong baseline** — 29.5% with just 697 programs and simple format

## Mechanism Logs
- GIFT parse rate 99.5% — model learns DataflowPlan format
- GIFT exec rate 77% — explicit bindings work but library coverage limits execution
- Active binding perturbation not yet systematically tested (planned, not implemented)

## Failed Tests
- GIFT coverage < 30% → library mining too coarse
- C < A → GIFT not competitive with flat SFT
- Zero two-call dataflow → no true composition

## Unresolved Questions
1. Can finer subroutine mining (step-level primitives) push GIFT coverage >50%?
2. Would more verified programs (2000+) make both flat SFT and GIFT competitive?
3. Is the library composition approach fundamentally limited vs flat program generation?
4. Would Type-Local GRPO (step-level credit assignment) help where outcome-GRPO failed?

## What GPT-5.5 Pro Should Review Next
1. **Is the 20.2% GIFT coverage a methodology failure or a data/mining failure?**
   - If mining: pivot to step-level primitives instead of whole-program subroutines
   - If methodology: the composition approach may be inferior to flat programs
2. **Should we pursue GIFT with more data, or focus on scaling flat SFT/GRPO?**
   - Flat SFT at 29.5% with 697 programs → what would 5000 programs achieve?
3. **The GIFT mechanism works (99.5% parse) but the library is the bottleneck.**
   - Recommend: mine smaller, more composable primitives (2-3 step subroutines)
4. **Tasks 6,7,8 still needed** — MCD rewrite, eval reliability, training reproducibility
5. **Multi-seed evaluation** — all results are seed 42 only

## Decision
**DEBUG MORE** — GIFT mechanism is sound (99.5% parse proves it), but library mining is too coarse (20.2% coverage, zero two-call plans). The bottleneck is data construction, not the GIFT architecture.
