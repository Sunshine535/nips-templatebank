#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
source "${SCRIPT_DIR}/gpu_utils.sh"
auto_setup

PROJ_DIR_ROOT="$PROJECT_DIR"
if [ -f "$PROJ_DIR_ROOT/.venv/bin/activate" ]; then
    source "$PROJ_DIR_ROOT/.venv/bin/activate"
fi
export PATH="$HOME/.local/bin:$PATH"

# --- Phase resume ---
PHASE_MARKER_DIR="$PROJECT_DIR/results/.phase_markers"
mkdir -p "$PHASE_MARKER_DIR"
FORCE_RERUN="${FORCE_RERUN:-0}"
phase_done() { touch "$PHASE_MARKER_DIR/phase_${1}.done"; echo "[PHASE $1] Completed at $(date)"; }
is_phase_done() {
    [[ "$FORCE_RERUN" == "1" ]] && return 1
    [[ -f "$PHASE_MARKER_DIR/phase_${1}.done" ]] && echo "[PHASE $1] Already completed. Skipping." && return 0
    return 1
}

CONFIG="${PROJECT_DIR}/configs/template_config.yaml"
TEMPLATE_DIR="${PROJECT_DIR}/results/templates"
OPS_DIR="${PROJECT_DIR}/results/operations"
COMPILER_DIR="${PROJECT_DIR}/results/compiler"
EVAL_DIR="${PROJECT_DIR}/results/eval"
LOG_DIR="${PROJECT_DIR}/logs"

cd "$PROJECT_DIR"
mkdir -p results "$LOG_DIR"

echo "================================================================"
echo "  Template Algebra — Full Experiment Pipeline"
echo "  Model: Qwen/Qwen3.5-9B | GPUs: ${NUM_GPUS}"
echo "================================================================"

# Stage 1: Extract Templates
if ! is_phase_done 1; then
    echo "========== STAGE 1: Extract Templates =========="
    python scripts/extract_templates.py \
        --config "$CONFIG" --output_dir "$TEMPLATE_DIR" \
        2>&1 | tee "$LOG_DIR/stage1_extract.log"
    phase_done 1
fi

# Stage 2: Template Algebra Operations
if ! is_phase_done 2; then
    echo "========== STAGE 2: Template Operations =========="
    python scripts/run_template_operations.py \
        --config "$CONFIG" --template_bank "${TEMPLATE_DIR}/template_bank.json" \
        --output_dir "$OPS_DIR" \
        2>&1 | tee "$LOG_DIR/stage2_operations.log"
    phase_done 2
fi

# Stage 3: Train Template Compiler
if ! is_phase_done 3; then
    echo "========== STAGE 3: Train Compiler =========="
    echo "  --- Stage 3a: Template Selection SFT ---"
    $(get_torchrun_cmd "$NUM_GPUS") scripts/train_template_compiler.py \
        --config "$CONFIG" --training_data "${TEMPLATE_DIR}/compiler_training_data.json" \
        --output_dir "$COMPILER_DIR" --skip_stage2 \
        2>&1 | tee "$LOG_DIR/stage3a_selection.log"

    echo "  --- Stage 3b: Variable Filling SFT ---"
    $(get_torchrun_cmd "$NUM_GPUS") scripts/train_template_compiler.py \
        --config "$CONFIG" --training_data "${TEMPLATE_DIR}/compiler_training_data.json" \
        --output_dir "$COMPILER_DIR" --skip_stage1 \
        2>&1 | tee "$LOG_DIR/stage3b_filling.log"
    phase_done 3
fi

# Stage 4: Full Evaluation
if ! is_phase_done 4; then
    echo "========== STAGE 4: Evaluation =========="
    python scripts/eval_template_reasoning.py \
        --config "$CONFIG" --compiler_dir "${COMPILER_DIR}/stage2_filling" \
        --template_bank "${TEMPLATE_DIR}/template_bank.json" \
        --output_dir "$EVAL_DIR" \
        2>&1 | tee "$LOG_DIR/stage4_eval.log"
    phase_done 4
fi

# Stage 5: Ablations
if ! is_phase_done 5; then
    echo "========== STAGE 5: Ablations =========="
    for BANK_SIZE in 10 25 50 100 200 300; do
        ABL_DIR="${PROJECT_DIR}/results/ablation_bank_${BANK_SIZE}"
        mkdir -p "$ABL_DIR"
        echo "  Testing bank size: ${BANK_SIZE}"
        python scripts/run_template_operations.py \
            --config "$CONFIG" --template_bank "${TEMPLATE_DIR}/template_bank.json" \
            --output_dir "$ABL_DIR" \
            2>&1 | tee "${ABL_DIR}/ablation.log"
    done
    phase_done 5
fi

echo "================================================================"
echo "  Pipeline Complete — $(date)"
echo "================================================================"

DONE_FILE="$PROJECT_DIR/results/.pipeline_done"
cat > "$DONE_FILE" << DONEEOF
{
  "project": "nips-templatebank",
  "completed_at": "$(date -u '+%Y-%m-%dT%H:%M:%SZ')",
  "hostname": "$(hostname)",
  "gpus": "${NUM_GPUS:-unknown}",
  "status": "PIPELINE_COMPLETE"
}
DONEEOF
echo "[PIPELINE_COMPLETE] Run 'bash collect_results.sh' to package results."
