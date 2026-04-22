# Reviewer Memory

## Round 1 — Score: 2/10
- **Suspicion**: Claimed bug fixes may exist in code but not in active entrypoints or saved artifacts
- **Suspicion**: The repo has dead-code "correct" implementations that are not actually used (e.g., reward path in rlvr_evolution.py vs broken path in train_seval.py)
- **Verified**: All current data is GSM8K-only, not MATH as proposal claims
- **Verified**: compose metrics contaminated by fallback CoT successes
- **Verified**: No adapter/checkpoint dirs exist — eval was on untrained base model
- **Verified**: Training data has wrong schema (instruction/output vs question/bindings/gold_answer)
- **Verified**: reward_fn called with empty bindings and None answer
- **Unresolved**: Whether library mining can ever produce >90% faithful plans (currently 0.72%)
- **Unresolved**: Whether the fundamental subroutine approach has value vs flat programs
- **Patterns**: Author tendency to implement elaborate architectures without validating basic pipeline correctness first
