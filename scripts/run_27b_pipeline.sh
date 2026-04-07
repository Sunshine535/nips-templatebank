#!/usr/bin/env bash
# Full 27B pipeline: verify → split → rebuild → QLoRA train → eval
# Run after extraction: bash scripts/run_27b_pipeline.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source /home/claude/nips-env/bin/activate

export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false

EXTRACT_DIR="${1:-results/extract_27b_500}"
VERIFIED_DIR="results/verified_27b"
PLANNER_DIR="results/planner_27b"
EVAL_DIR="results/eval_27b"
CONFIG="configs/template_config.yaml"

mkdir -p "$VERIFIED_DIR" "$PLANNER_DIR" "$EVAL_DIR" logs

echo "================================================================"
echo "  27B Full Pipeline — $(date)"
echo "  Extraction: $EXTRACT_DIR"
echo "================================================================"

# Step 1: Verify programs
echo "========== Step 1: Verify Programs =========="
python3 -u scripts/verify_programs.py \
    --input "$EXTRACT_DIR/all_programs.json" \
    --output "$VERIFIED_DIR/all_programs.json" \
    --re_execute \
    2>&1 | tee logs/27b_step1_verify.log

VERIFIED=$(python3 -c "import json; print(len(json.load(open('$VERIFIED_DIR/all_programs.json'))))")
echo "Verified: $VERIFIED programs"

# Step 2: Build MCD split
echo "========== Step 2: MCD Split =========="
python3 -u scripts/build_mcd_split.py \
    --config "$CONFIG" \
    --programs "$VERIFIED_DIR/all_programs.json" \
    --output results/mcd_split_27b.json \
    --num_trials 500 \
    2>&1 | tee logs/27b_step2_mcd.log

# Step 3: Post-split rebuild
echo "========== Step 3: Post-split Rebuild =========="
python3 -u scripts/extract_templates.py \
    --config "$CONFIG" \
    --output_dir "$VERIFIED_DIR" \
    --post_split \
    --split_path results/mcd_split_27b.json \
    --programs_path "$VERIFIED_DIR/all_programs.json" \
    2>&1 | tee logs/27b_step3_rebuild.log

TRAIN_SIZE=$(python3 -c "import json; print(len(json.load(open('$VERIFIED_DIR/compose_train.json'))))")
echo "Training data: $TRAIN_SIZE examples"

# Step 4a: QLoRA train compose planner (27B)
echo "========== Step 4a: QLoRA Train Compose (27B) =========="
python3 -u scripts/train_template_compiler.py \
    --config "$CONFIG" --mode compose \
    --training_data "$VERIFIED_DIR/compose_train.json" \
    --output_dir "$PLANNER_DIR/compose" \
    --resume auto \
    2>&1 | tee logs/27b_step4a_train_compose.log

# Step 4b: QLoRA train flat baseline (27B)
echo "========== Step 4b: QLoRA Train Flat (27B) =========="
python3 -u scripts/train_template_compiler.py \
    --config "$CONFIG" --mode flat \
    --training_data "$VERIFIED_DIR/flat_train.json" \
    --output_dir "$PLANNER_DIR/flat" \
    --resume auto \
    2>&1 | tee logs/27b_step4b_train_flat.log

# Step 5: Evaluation
echo "========== Step 5: Evaluation =========="
python3 -u scripts/eval_template_reasoning.py \
    --config "$CONFIG" \
    --compose_dir "$PLANNER_DIR/compose" \
    --flat_dir "$PLANNER_DIR/flat" \
    --library_path "$VERIFIED_DIR/subroutine_library.json" \
    --split_path results/mcd_split_27b.json \
    --programs_path "$VERIFIED_DIR/all_programs.json" \
    --output_dir "$EVAL_DIR" \
    --seed 42 \
    --max_samples 200 \
    --skip_retrieval \
    2>&1 | tee logs/27b_step5_eval.log

echo "================================================================"
echo "  Pipeline Complete — $(date)"
echo "  Results: $EVAL_DIR"
echo "================================================================"

# ===== Post-pipeline: Git push + ZIP =====
echo "========== Post-Pipeline: Push + Package =========="

cd /home/claude/nips-templatebank

# Git push
git add -A
git commit -m "$(cat <<'EOF'
feat: 27B planner results (QLoRA, GSM8K+MATH)

- Teacher: Qwen3.5-27B (88% GSM8K CoT baseline)
- Planner: Qwen3.5-27B + QLoRA (4-bit, r=32, α=64)
- Extraction: 500 GSM8K + 500 MATH with enable_thinking=False
- Training: 3 epochs QLoRA on H100 80GB
- Eval: compose, flat_inline, direct_cot, cot_budget on GSM8K + MCD

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)" || echo "Nothing to commit"
git push origin main || echo "Push failed - will retry"

# ZIP everything
echo "Creating ZIP archive..."
cd /home/claude
zip -r /workspace/nips-templatebank-full.zip nips-templatebank/ \
    -x "nips-templatebank/.git/*" \
    -x "nips-templatebank/__pycache__/*" \
    -x "nips-templatebank/*/__pycache__/*" \
    -x "nips-templatebank/results/planner_27b/*/checkpoint-*/*" \
    2>&1 | tail -5

echo "================================================================"
echo "  ZIP saved to: /workspace/nips-templatebank-full.zip"
echo "  Size: $(du -sh /workspace/nips-templatebank-full.zip 2>/dev/null | cut -f1)"
echo "  Pipeline fully complete — $(date)"
echo "================================================================"
