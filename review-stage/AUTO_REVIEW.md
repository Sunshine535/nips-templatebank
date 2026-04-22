# SEVAL Auto Review Loop

## Round 1 (2026-04-23T02:30:00Z)

### Assessment (Summary)
- Score: 2/10
- Verdict: not ready
- Reviewer: GPT via Codex MCP (nightmare mode, read-only repo access)
- Thread: 019db66b-f612-7c01-86d8-56f4e043e209

### Key criticisms:
- All 4 claims (C1-C4) unverified with zero positive results
- train_seval.py has wrong data schema and broken reward (empty bindings, None answer)
- All eval data is GSM8K, not MATH as proposal claims
- Library mining produces 0.72% faithful plans
- C3 eval script would crash on actual data
- No trained baselines exist (eval was on untrained base model)

<details>
<summary>Click to expand full reviewer response</summary>

Score: 2/10. Verdict: not ready.

Verified claims: none.

Unverified/false claims:
- C1: No evolved-vs-frozen result. Only eval shows compose at 2% GSM8K, 0% MCD.
- C2: RLVR-vs-SFT comparison not implemented. train_seval.py reward is broken.
- C3: eval_test_time_tools.py would crash on mcd_split_v2.json data format.
- C4: No transfer artifacts exist.

Weaknesses (ranked):
1. Training entrypoint broken: expects question/bindings/gold_answer, data has instruction/output/source. Reward uses empty bindings.
2. Composition plans 0.72% faithful (410/418 empty bindings).
3. Eval on wrong benchmark (GSM8K not MATH).
4. Metrics include fallback CoT, only 5/10 methods implemented.
5. C3 non-operational.
6. Library miner collapses semantics.
7. C2 eval script doesn't exist.

Path: Fix data schema, achieve >90% faithful plans, train real SFT baselines, then run SEVAL seed.

</details>

### Actions Taken

1. **Fixed 3 critical bugs** in extraction pipeline (binding mapping, executor fallback, library mining)
   - Commit: 8dc645a
   - Local validation: 2% → 6.5% faithful (still insufficient for composition approach)

2. **Strategic pivot**: Composition/library layer has fundamental limitation (fixed subroutine expressions only match exact same math structure). Pivoting to flat program GRPO as primary approach.

3. **Rewrote train_seval.py** (commit: 7e3fa60)
   - Fixed data schema: `normalize_training_data()` handles all formats
   - Fixed reward: properly parses program, extracts bindings, executes, checks answer
   - Added SFT pre-training stage
   - Added proper evaluation with parse/exec/accuracy metrics
   - Supports 3 modes: sft, grpo, full (sft then grpo)
   - Dropped composition/library dependency for core training path

4. **Phase 0 extraction running** on pod (n88, 4x H200) — will produce verified programs for SFT/GRPO training

### Strategy Change

**Original SEVAL (4 claims, composition-based)** → **Focused SEVAL (RLVR + verified execution)**

Core research question: *Does GRPO with execution-verification reward improve math program generation beyond SFT?*

New claim structure:
- C1: GRPO with verified execution > SFT (on program generation accuracy)
- C2: GRPO expands compositional reasoning (pass@K analysis, RLVR vs SFT)
- C3: Cross-model transfer (9B → 4B)

Experiment plan:
1. Phase 0: Teacher extraction (RUNNING on pod) → ~500+ verified programs
2. SFT on verified programs → baseline accuracy
3. GRPO with execution reward → improved accuracy
4. Compare GRPO vs SFT at pass@1, pass@4, pass@16, pass@64
5. Transfer to 4B model

### Status
- Continuing to round 2 after Phase 0 extraction completes
- Difficulty: nightmare
- Pending: Phase 0 extraction (~2-6h remaining)
