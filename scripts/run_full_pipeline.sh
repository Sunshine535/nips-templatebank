#!/usr/bin/env bash
# Full experiment pipeline for single H100 GPU
# Run after extraction completes: bash scripts/run_full_pipeline.sh
set -euo pipefail

cd "$(dirname "$0")/.."
source "$(dirname "$0")/../.venv/bin/activate"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled  # Disable wandb for now

TEMPLATE_DIR="results/templates_full"
VERIFIED_DIR="results/templates_verified"
PLANNER_DIR="results/planner_v2"
EVAL_DIR="results/eval_v2"
LOG_DIR="logs"
CONFIG="configs/template_config.yaml"

mkdir -p "$VERIFIED_DIR" "$PLANNER_DIR" "$EVAL_DIR" "$LOG_DIR"

echo "================================================================"
echo "  Full Pipeline — Single H100 80GB"
echo "  $(date)"
echo "================================================================"

# Step 1: Verify programs (answer-correct filtering)
echo "========== Step 1: Verify Programs =========="
if [ ! -f "$VERIFIED_DIR/all_programs.json" ]; then
    python scripts/verify_programs.py \
        --input "$TEMPLATE_DIR/all_programs.json" \
        --output "$VERIFIED_DIR/all_programs.json" \
        --re_execute \
        2>&1 | tee "$LOG_DIR/step1_verify.log"
else
    echo "  [SKIP] Verified programs already exist"
fi

# Step 2: Build MCD split from verified programs
echo "========== Step 2: Build MCD Split =========="
if [ ! -f "results/mcd_split_v2.json" ]; then
    python scripts/build_mcd_split.py \
        --config "$CONFIG" \
        --programs "$VERIFIED_DIR/all_programs.json" \
        --output results/mcd_split_v2.json \
        --num_trials 500 \
        2>&1 | tee "$LOG_DIR/step2_mcd_split.log"
else
    echo "  [SKIP] MCD split already exists"
fi

# Step 3: Post-split rebuild (library + plans from train only)
echo "========== Step 3: Post-split Rebuild =========="
if [ ! -f "$VERIFIED_DIR/subroutine_library.json" ]; then
    python scripts/extract_templates.py \
        --config "$CONFIG" \
        --output_dir "$VERIFIED_DIR" \
        --post_split \
        --split_path results/mcd_split_v2.json \
        --programs_path "$VERIFIED_DIR/all_programs.json" \
        2>&1 | tee "$LOG_DIR/step3_post_split.log"
else
    echo "  [SKIP] Post-split rebuild already exists"
fi

# Step 4a: Train compose planner
echo "========== Step 4a: Train Compose Planner =========="
if [ ! -f "$PLANNER_DIR/compose/adapter_config.json" ]; then
    python scripts/train_template_compiler.py \
        --config "$CONFIG" --mode compose \
        --training_data "$VERIFIED_DIR/compose_train.json" \
        --output_dir "$PLANNER_DIR/compose" \
        --resume auto \
        2>&1 | tee "$LOG_DIR/step4a_train_compose.log"
else
    echo "  [SKIP] Compose planner already trained"
fi

# Step 4b: Train flat baseline
echo "========== Step 4b: Train Flat Baseline =========="
if [ ! -f "$PLANNER_DIR/flat/adapter_config.json" ]; then
    python scripts/train_template_compiler.py \
        --config "$CONFIG" --mode flat \
        --training_data "$VERIFIED_DIR/flat_train.json" \
        --output_dir "$PLANNER_DIR/flat" \
        --resume auto \
        2>&1 | tee "$LOG_DIR/step4b_train_flat.log"
else
    echo "  [SKIP] Flat baseline already trained"
fi

# Step 5: Canary evaluation (compose vs flat, MCD only, 1 seed)
echo "========== Step 5: Canary Evaluation =========="
python scripts/eval_template_reasoning.py \
    --config "$CONFIG" \
    --compose_dir "$PLANNER_DIR/compose" \
    --flat_dir "$PLANNER_DIR/flat" \
    --library_path "$VERIFIED_DIR/subroutine_library.json" \
    --split_path results/mcd_split_v2.json \
    --programs_path "$VERIFIED_DIR/all_programs.json" \
    --output_dir "$EVAL_DIR" \
    --seed 42 \
    --skip_cot --skip_retrieval \
    2>&1 | tee "$LOG_DIR/step5_canary_eval.log"

echo "================================================================"
echo "  Canary evaluation complete — check results in $EVAL_DIR"
echo "  If compose > flat_inline on fallback_free_accuracy, continue!"
echo "  $(date)"
echo "================================================================"
