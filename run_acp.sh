#!/bin/bash
# SenseCore Startup Script (4× H100)
# Startup command: bash /data/szs/250010072/nwh/nips-templatebank/run_acp.sh
set -euo pipefail

PROJECT_DIR=/data/szs/250010072/nwh/nips-templatebank
DATA_DIR=/data/szs/share/templatebank
SHARE_DIR=/data/szs/share

mkdir -p "${DATA_DIR}"/{results,logs,hf_cache}
mkdir -p "${DATA_DIR}/results/.phase_markers"

# ── Environment ──────────────────────────────────────────────
export HF_HOME="${DATA_DIR}/hf_cache"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE="${WANDB_MODE:-offline}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=8

# ── GPU detection (SenseCore exposes SENSECORE_ACCELERATE_DEVICE_COUNT) ──
if [ -n "${SENSECORE_ACCELERATE_DEVICE_COUNT:-}" ]; then
    NUM_GPUS="$SENSECORE_ACCELERATE_DEVICE_COUNT"
elif [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -ra _gpu_arr <<< "$CUDA_VISIBLE_DEVICES"
    NUM_GPUS=${#_gpu_arr[@]}
else
    NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
fi
export NUM_GPUS
echo "[run_acp] GPUs detected: ${NUM_GPUS}"

# ── Symlink results/ and logs/ to shared storage ─────────────
cd "$PROJECT_DIR"
for dir in results logs; do
    if [ -L "$dir" ]; then
        echo "[symlink] $dir -> $(readlink "$dir") (already linked)"
    elif [ -d "$dir" ] && [ "$(readlink -f "$dir")" != "$(readlink -f "${DATA_DIR}/${dir}")" ]; then
        if [ "$(ls -A "$dir" 2>/dev/null)" ]; then
            echo "[symlink] Migrating existing $dir/ contents to ${DATA_DIR}/${dir}/"
            cp -a "$dir"/. "${DATA_DIR}/${dir}/"
        fi
        rm -rf "$dir"
        ln -sfn "${DATA_DIR}/${dir}" "$dir"
        echo "[symlink] $dir -> ${DATA_DIR}/${dir}"
    elif [ ! -e "$dir" ]; then
        ln -sfn "${DATA_DIR}/${dir}" "$dir"
        echo "[symlink] $dir -> ${DATA_DIR}/${dir}"
    fi
done

# ── Resolve model paths ─────────────────────────────────────
STUDENT_MODEL="${SHARE_DIR}/Qwen3.5-9B"
if [ -d "${SHARE_DIR}/Qwen3.5-32B" ]; then
    TEACHER_MODEL="${SHARE_DIR}/Qwen3.5-32B"
elif [ -d "${SHARE_DIR}/Qwen3.5-27B" ]; then
    TEACHER_MODEL="${SHARE_DIR}/Qwen3.5-27B"
else
    echo "[WARN] No 32B/27B teacher model found under ${SHARE_DIR}, falling back to student"
    TEACHER_MODEL="$STUDENT_MODEL"
fi
echo "[run_acp] Student model : ${STUDENT_MODEL}"
echo "[run_acp] Teacher model : ${TEACHER_MODEL}"

# ── Generate runtime config with local model paths ──────────
RUNTIME_CONFIG="${PROJECT_DIR}/configs/template_config_acp.yaml"
python3 - <<PYEOF
import yaml, shutil, os
src = "${PROJECT_DIR}/configs/template_config.yaml"
dst = "${RUNTIME_CONFIG}"
with open(src) as f:
    cfg = yaml.safe_load(f)
cfg["teacher"]["model"] = "${TEACHER_MODEL}"
cfg["planner"]["model"] = "${STUDENT_MODEL}"
cfg["training"]["report_to"] = os.environ.get("WANDB_MODE", "offline") != "offline" and "wandb" or "none"
with open(dst, "w") as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
print(f"[config] Written runtime config: {dst}")
PYEOF

# ── Install / verify deps ───────────────────────────────────
echo "[deps] Checking core dependencies..."
python3 -c "import torch, transformers, trl, peft, datasets, accelerate" 2>/dev/null || {
    echo "[deps] Installing missing dependencies..."
    pip install --quiet -r "${PROJECT_DIR}/requirements.txt" 2>&1 | tail -5
}

python3 -c "
import torch
print(f'  PyTorch  : {torch.__version__}')
print(f'  CUDA     : {torch.version.cuda}')
n = torch.cuda.device_count()
print(f'  GPUs     : {n}')
for i in range(n):
    props = torch.cuda.get_device_properties(i)
    print(f'    GPU {i}: {props.name}  {props.total_memory // (1024**3)} GB')
"

# ── Run pipeline ─────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  nips-templatebank — SenseCore Pipeline"
echo "  GPUs: ${NUM_GPUS}  |  Config: ${RUNTIME_CONFIG}"
echo "  Results: ${DATA_DIR}/results/"
echo "================================================================"

export CONFIG_OVERRIDE="${RUNTIME_CONFIG}"

exec bash scripts/run_all_experiments.sh "$@" 2>&1 | tee "${DATA_DIR}/logs/run_acp_$(date +%Y%m%d_%H%M%S).log"
