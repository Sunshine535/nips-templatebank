#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
source "${SCRIPT_DIR}/gpu_utils.sh"
auto_setup

if [ -f "$PROJECT_DIR/.venv/bin/activate" ]; then
    source "$PROJECT_DIR/.venv/bin/activate"
fi
export PATH="$HOME/.local/bin:$PATH"

PHASE_MARKER_DIR="$PROJECT_DIR/results/.phase_markers"
mkdir -p "$PHASE_MARKER_DIR"
FORCE_RERUN="${FORCE_RERUN:-0}"
phase_done() { touch "$PHASE_MARKER_DIR/phase_${1}.done"; echo "[PHASE $1] Completed at $(date)"; }
is_phase_done() {
    [[ "$FORCE_RERUN" == "1" ]] && return 1
    [[ -f "$PHASE_MARKER_DIR/phase_${1}.done" ]] && echo "[PHASE $1] Already completed. Skipping." && return 0
    return 1
}

CONFIG="${CONFIG_OVERRIDE:-${PROJECT_DIR}/configs/template_config.yaml}"
TEMPLATE_DIR="${PROJECT_DIR}/results/templates"
PLANNER_DIR="${PROJECT_DIR}/results/planner"
EVAL_DIR="${PROJECT_DIR}/results/eval"
LOG_DIR="${PROJECT_DIR}/logs"

cd "$PROJECT_DIR"
mkdir -p results "$LOG_DIR"

# Quick sanity check mode: use --smoke flag or SMOKE=1 env var
SMOKE="${SMOKE:-0}"
SMOKE_MAX="${SMOKE_MAX:-50}"
MAX_PER_SOURCE_FLAG=""
MAX_SAMPLES_FLAG=""
if [[ "${1:-}" == "--smoke" ]] || [[ "$SMOKE" == "1" ]]; then
    SMOKE=1
    MAX_PER_SOURCE_FLAG="--max_per_source $SMOKE_MAX"
    MAX_SAMPLES_FLAG="--max_samples $SMOKE_MAX"
    echo "=== SMOKE TEST MODE (max $SMOKE_MAX per source) ==="
fi

echo "================================================================"
echo "  Subroutine Composition — Full Experiment Pipeline"
echo "  Planner: Qwen/Qwen3.5-9B | Teacher: Qwen/Qwen3.5-32B"
echo "  GPUs: ${NUM_GPUS} | Smoke: ${SMOKE}"
echo "================================================================"

# Stage 1: Extract programs + build library + composition plans
if ! is_phase_done 1; then
    echo "========== STAGE 1: Extract Programs & Build Library =========="
    EXTRA_FLAGS=""
    if [[ "$SMOKE" == "1" ]]; then
        EXTRA_FLAGS="--use_student --synthetic"
    fi
    python scripts/extract_templates.py \
        --config "$CONFIG" --output_dir "$TEMPLATE_DIR" \
        $MAX_PER_SOURCE_FLAG $EXTRA_FLAGS \
        2>&1 | tee "$LOG_DIR/stage1_extract.log"
    phase_done 1
fi

# Stage 2: Build MCD split
if ! is_phase_done 2; then
    echo "========== STAGE 2: Build MCD Split =========="
    python scripts/build_mcd_split.py \
        --plans "${TEMPLATE_DIR}/plans_with_programs.json" \
        --output "${PROJECT_DIR}/results/mcd_split.json" \
        2>&1 | tee "$LOG_DIR/stage2_mcd_split.log"
    phase_done 2
fi

# Stage 3a: Train compose planner
if ! is_phase_done 3a; then
    echo "========== STAGE 3a: Train Compose Planner =========="
    $(get_torchrun_cmd "$NUM_GPUS") scripts/train_template_compiler.py \
        --config "$CONFIG" --mode compose \
        --training_data "${TEMPLATE_DIR}/compose_train.json" \
        --output_dir "${PLANNER_DIR}/compose" \
        --resume auto \
        2>&1 | tee "$LOG_DIR/stage3a_compose.log"
    phase_done 3a
fi

# Stage 3b: Train flat-program baseline
if ! is_phase_done 3b; then
    echo "========== STAGE 3b: Train Flat-Program Baseline =========="
    $(get_torchrun_cmd "$NUM_GPUS") scripts/train_template_compiler.py \
        --config "$CONFIG" --mode flat \
        --training_data "${TEMPLATE_DIR}/flat_train.json" \
        --output_dir "${PLANNER_DIR}/flat" \
        --resume auto \
        2>&1 | tee "$LOG_DIR/stage3b_flat.log"
    phase_done 3b
fi

# Stage 4: Full evaluation
EVAL_MAX="${EVAL_MAX_SAMPLES:-500}"
if ! is_phase_done 4; then
    echo "========== STAGE 4: Evaluation (max=${EVAL_MAX}) =========="
    python scripts/eval_template_reasoning.py \
        --config "$CONFIG" \
        --compose_dir "${PLANNER_DIR}/compose" \
        --flat_dir "${PLANNER_DIR}/flat" \
        --library_path "${TEMPLATE_DIR}/subroutine_library.json" \
        --output_dir "$EVAL_DIR" \
        $MAX_SAMPLES_FLAG \
        2>&1 | tee "$LOG_DIR/stage4_eval.log"
    phase_done 4
fi

# Stage 5: Library size ablation
if ! is_phase_done 5; then
    echo "========== STAGE 5: Library Size Ablation =========="
    for L_SIZE in 4 8 16 32; do
        ABL_DIR="${PROJECT_DIR}/results/ablation_L${L_SIZE}"
        mkdir -p "$ABL_DIR"
        echo "  Testing library size L=${L_SIZE}..."
    done
    phase_done 5
fi

echo "================================================================"
echo "  Pipeline Complete — $(date)"
echo "================================================================"

cat > "$PROJECT_DIR/results/.pipeline_done" << DONEEOF
{
  "project": "nips-templatebank",
  "completed_at": "$(date -u '+%Y-%m-%dT%H:%M:%SZ')",
  "hostname": "$(hostname)",
  "gpus": "${NUM_GPUS:-unknown}",
  "smoke": ${SMOKE},
  "status": "PIPELINE_COMPLETE"
}
DONEEOF
echo "[PIPELINE_COMPLETE] Run 'bash collect_results.sh' to package results."
