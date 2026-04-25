#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="${HF_HOME:-/openbayes/input/input0}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"

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
ALLOW_SYNTH_FLAG=""
if [[ "${1:-}" == "--smoke" ]] || [[ "$SMOKE" == "1" ]]; then
    SMOKE=1
    MAX_PER_SOURCE_FLAG="--max_per_source $SMOKE_MAX"
    MAX_SAMPLES_FLAG="--max_samples $SMOKE_MAX"
    ALLOW_SYNTH_FLAG="--allow_synthetic"
    echo "=== SMOKE TEST MODE (max $SMOKE_MAX per source) ==="
fi

echo "================================================================"
echo "  Subroutine Composition — Full Experiment Pipeline"
echo "  Planner: Qwen/Qwen3.5-9B | Teacher: Qwen/Qwen3.5-27B"
echo "  GPUs: ${NUM_GPUS} | Smoke: ${SMOKE}"
echo "================================================================"

# Stage 1: Extract programs + build library + composition plans
if ! is_phase_done 1; then
    echo "========== STAGE 1: Extract Programs & Build Library =========="
    EXTRA_FLAGS=""
    if [[ "$SMOKE" == "1" ]]; then
        EXTRA_FLAGS="--use_student --synthetic --allow_synthetic"
    fi
    python scripts/extract_templates.py \
        --config "$CONFIG" --output_dir "$TEMPLATE_DIR" \
        $MAX_PER_SOURCE_FLAG $EXTRA_FLAGS \
        2>&1 | tee "$LOG_DIR/stage1_extract.log"
    [[ ${PIPESTATUS[0]} -eq 0 ]] || { echo "[PHASE 1] FAILED"; exit 1; }
    phase_done 1
fi

# Stage 2: Build MCD split
if ! is_phase_done 2; then
    echo "========== STAGE 2: Build MCD Split =========="
    python scripts/build_mcd_split.py \
        --plans "${TEMPLATE_DIR}/plans_with_programs.json" \
        --output "${PROJECT_DIR}/results/mcd_split.json" \
        2>&1 | tee "$LOG_DIR/stage2_mcd_split.log"
    [[ ${PIPESTATUS[0]} -eq 0 ]] || { echo "[PHASE 2] FAILED"; exit 1; }
    phase_done 2
fi

# Stage 3a: Train compose planner
if ! is_phase_done 3a; then
    echo "========== STAGE 3a: Train Compose Planner =========="
    $(get_torchrun_cmd "$NUM_GPUS") scripts/train_template_compiler.py \
        --config "$CONFIG" --mode compose \
        --training_data "${TEMPLATE_DIR}/compose_train.json" \
        --output_dir "${PLANNER_DIR}/compose" \
        --resume auto $ALLOW_SYNTH_FLAG \
        2>&1 | tee "$LOG_DIR/stage3a_compose.log"
    [[ ${PIPESTATUS[0]} -eq 0 ]] || { echo "[PHASE 3a] FAILED"; exit 1; }
    phase_done 3a
fi

# Stage 3b: Train flat-program baseline
if ! is_phase_done 3b; then
    echo "========== STAGE 3b: Train Flat-Program Baseline =========="
    $(get_torchrun_cmd "$NUM_GPUS") scripts/train_template_compiler.py \
        --config "$CONFIG" --mode flat \
        --training_data "${TEMPLATE_DIR}/flat_train.json" \
        --output_dir "${PLANNER_DIR}/flat" \
        --resume auto $ALLOW_SYNTH_FLAG \
        2>&1 | tee "$LOG_DIR/stage3b_flat.log"
    [[ ${PIPESTATUS[0]} -eq 0 ]] || { echo "[PHASE 3b] FAILED"; exit 1; }
    phase_done 3b
fi

# Stage 4: Full evaluation (multi-seed from config: evaluation.num_seeds)
EVAL_MAX="${EVAL_MAX_SAMPLES:-500}"
CONFIG_NUM_SEEDS=$(python3 -c "import yaml; c=yaml.safe_load(open('${CONFIG}')); print(c.get('evaluation',{}).get('num_seeds',3))" 2>/dev/null || echo 3)
NUM_SEEDS="${NUM_SEEDS:-$CONFIG_NUM_SEEDS}"
if ! is_phase_done 4; then
    echo "========== STAGE 4: Evaluation (max=${EVAL_MAX}, seeds=${NUM_SEEDS}) =========="
    for SEED in $(seq 1 "$NUM_SEEDS"); do
        SEED_VAL=$((41 + SEED))
        echo "  --- Seed ${SEED}/${NUM_SEEDS} (seed=${SEED_VAL}) ---"
        python scripts/eval_template_reasoning.py \
            --config "$CONFIG" \
            --compose_dir "${PLANNER_DIR}/compose" \
            --flat_dir "${PLANNER_DIR}/flat" \
            --library_path "${TEMPLATE_DIR}/subroutine_library.json" \
            --split_path "${PROJECT_DIR}/results/mcd_split.json" \
            --output_dir "$EVAL_DIR" \
            --seed "$SEED_VAL" \
            $MAX_SAMPLES_FLAG \
            2>&1 | tee "$LOG_DIR/stage4_eval_seed${SEED_VAL}.log"
        [[ ${PIPESTATUS[0]} -eq 0 ]] || { echo "[PHASE 4] FAILED at seed=${SEED_VAL}"; exit 1; }
    done
    phase_done 4
fi

# Stage 5: Library size ablation L={4,8,16,32} (matches config ablation.library_sizes)
if ! is_phase_done 5; then
    echo "========== STAGE 5: Library Size Ablation L={4,8,16,32} =========="
    ABLATION_DIR="${PROJECT_DIR}/results/ablation"
    FULL_LIB="${TEMPLATE_DIR}/subroutine_library.json"

    for L_SIZE in 4 8 16 32; do
        L_DIR="${ABLATION_DIR}/L${L_SIZE}"
        mkdir -p "$L_DIR"
        SUBLIB="${L_DIR}/subroutine_library.json"
        python -c "
import json, random, sys
random.seed(42)
with open('${FULL_LIB}') as f:
    lib = json.load(f)
subs = lib.get('subroutines', lib.get('library', []))
L = ${L_SIZE}
if L >= len(subs):
    sampled = subs
else:
    sampled = random.sample(subs, L)
out = dict(lib)
key = 'subroutines' if 'subroutines' in lib else 'library'
out[key] = sampled
with open('${SUBLIB}', 'w') as f:
    json.dump(out, f, indent=2)
print(f'Ablation L={L}: {len(sampled)}/{len(subs)} subroutines')
"
        python scripts/eval_template_reasoning.py \
            --config "$CONFIG" \
            --compose_dir "${PLANNER_DIR}/compose" \
            --flat_dir "${PLANNER_DIR}/flat" \
            --library_path "$SUBLIB" \
            --split_path "${PROJECT_DIR}/results/mcd_split.json" \
            --output_dir "$L_DIR" \
            --skip_cot --skip_flat \
            $MAX_SAMPLES_FLAG \
            2>&1 | tee "$LOG_DIR/stage5_ablation_L${L_SIZE}.log"
        [[ ${PIPESTATUS[0]} -eq 0 ]] || { echo "[PHASE 5] FAILED at L=${L_SIZE}"; exit 1; }
    done

    # Control: random templates (same size as L=16, the main_size)
    RAND_DIR="${ABLATION_DIR}/random_control"
    mkdir -p "$RAND_DIR"
    RANDLIB="${RAND_DIR}/subroutine_library.json"
    python -c "
import json, random
random.seed(99)
with open('${FULL_LIB}') as f:
    lib = json.load(f)
subs = lib.get('subroutines', lib.get('library', []))
L = min(16, len(subs))
shuffled = list(subs)
for s in shuffled:
    if 'steps' in s:
        random.shuffle(s['steps'])
    if 'signature' in s:
        s['signature'] = 'random_' + s.get('name', 'sub')
sampled = random.sample(shuffled, L) if L < len(shuffled) else shuffled
out = dict(lib)
key = 'subroutines' if 'subroutines' in lib else 'library'
out[key] = sampled
with open('${RANDLIB}', 'w') as f:
    json.dump(out, f, indent=2)
print(f'Random control: {len(sampled)} shuffled subroutines')
"
    python scripts/eval_template_reasoning.py \
        --config "$CONFIG" \
        --compose_dir "${PLANNER_DIR}/compose" \
        --flat_dir "${PLANNER_DIR}/flat" \
        --library_path "$RANDLIB" \
        --split_path "${PROJECT_DIR}/results/mcd_split.json" \
        --output_dir "$RAND_DIR" \
        --skip_cot --skip_flat \
        $MAX_SAMPLES_FLAG \
        2>&1 | tee "$LOG_DIR/stage5_ablation_random.log"
    [[ ${PIPESTATUS[0]} -eq 0 ]] || { echo "[PHASE 5] FAILED at random control"; exit 1; }

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
