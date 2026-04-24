# Bug Fix Log

## Bug Fix: Empty bindings in composition training data
Files changed: `scripts/extract_templates.py`
Reason: 98% of compose_train had `bindings: {}` because slot names didn't match across programs
Evidence: `results/templates_verified/compose_train.json` — 410/418 empty
Change: Added `_find_matching_bindings()` with permutation search over values
Verification: Local test showed 2% → 6.5% faithful (still insufficient due to deeper issue)
Before: 98% empty bindings
After: Permutation search finds correct bindings when math structure matches
Remaining risk: Only 20.2% coverage — library mining too coarse for most programs

## Bug Fix: CompositionExecutor silent wrong binding
Files changed: `src/template_dsl.py`
Reason: When multiple type-matching candidates existed, executor silently used the LAST one for ALL slots
Evidence: All slots got same value → near-zero accuracy despite high exec rate
Change: Reject ambiguous multi-candidate bindings instead of silent wrong assignment
Verification: `pytest tests/test_templatebank.py` — 31/31 pass
Before: Silent wrong answers
After: Clear failure on ambiguous bindings

## Bug Fix: Library mining collapse
Files changed: `scripts/extract_templates.py`, `configs/template_config.yaml`
Reason: 697 programs collapsed to 3 usable templates because signature ignored slot count
Evidence: Only L00, L08, L15 used in 418 training plans
Change: Include slot count in mining signature + subsequence mining + target_size 16→32
Verification: New mining produces 32 subroutines with better diversity
Before: 3/16 templates used
After: 32 templates with slot-count differentiation

## Bug Fix: train_seval.py data schema mismatch
Files changed: `scripts/train_seval.py`
Reason: Expected `question/bindings/gold_answer` but data had `instruction/output/source`. Reward used `reward_fn(completion, {}, None)`
Evidence: External reviewer (GPT, 2/10 score) identified this
Change: Complete rewrite with `normalize_training_data()`, proper reward function
Verification: SFT training completed successfully, 29.5% on GSM8K test
Before: Training script crashes or trains with wrong reward
After: Working SFT+GRPO pipeline

## Bug Fix: Container OOM (16GB limit)
Files changed: k8s deployment (user action)
Reason: Container memory limit 16GB, 27B model needs ~54GB for loading
Evidence: `kubectl describe pod` showed `Reason: OOMKilled, Exit Code: 137`
Change: Memory limit increased to 128GB
Verification: No more OOM crashes
Before: Container killed after ~50 minutes of inference
After: Stable operation

## Bug Fix: Python log buffering
Files changed: Pod startup scripts, `scripts/extract_templates.py`
Reason: Python logging output never reached log file (block buffered)
Evidence: FD position = file size for hours despite active GPU inference
Change: Added `python3 -u`, explicit `_flush_logs()` every 50 problems, checkpointing
Verification: Logs now update in real-time
Before: Zero progress visibility for hours
After: Progress visible every 50 problems
