#!/usr/bin/env bash
# =============================================================================
# Download all datasets + models for SEVAL project
# -----------------------------------------------------------------------------
# Usage (on download container with internet, e.g. /openbayes/input/input0):
#   git clone https://github.com/Sunshine535/nips-templatebank.git
#   cd nips-templatebank
#   bash scripts/download_assets.sh
#
# After download completes:
#   rsync -avP assets/ user@tju-hpc:/path/to/project/assets/
#
# Output layout (relative to project root):
#   assets/datasets/gsm8k/
#   assets/datasets/math/
#   assets/models/Qwen3.5-27B/
#   assets/models/Qwen3.5-9B/
#   assets/models/Qwen3.5-4B/
# =============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ASSETS_DIR="${ASSETS_DIR:-$PROJECT_DIR/assets}"
DATASETS_DIR="$ASSETS_DIR/datasets"
MODELS_DIR="$ASSETS_DIR/models"

mkdir -p "$DATASETS_DIR" "$MODELS_DIR"

# Use HF_ENDPOINT mirror if set (e.g. https://hf-mirror.com for users in CN)
# export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"

# Optional: set HF_TOKEN for higher rate limits
# export HF_TOKEN="..."

echo "============================================================"
echo " SEVAL Asset Download"
echo " Project: $PROJECT_DIR"
echo " Assets:  $ASSETS_DIR"
echo " HF endpoint: ${HF_ENDPOINT:-https://huggingface.co (default)}"
echo " Time:    $(date)"
echo "============================================================"

# --- Ensure huggingface_hub is installed -----------------------------------
if ! python3 -c "import huggingface_hub" 2>/dev/null; then
    echo "[setup] Installing huggingface_hub + hf_transfer..."
    pip install -q huggingface_hub hf_transfer datasets
fi

# Enable fast downloads via hf_transfer (uses Rust client for parallelism)
export HF_HUB_ENABLE_HF_TRANSFER=1

# --- Helper: skip if already downloaded ------------------------------------
already_downloaded() {
    local marker="$1/.download_done"
    [ -f "$marker" ]
}
mark_done() {
    touch "$1/.download_done"
}

# --- Download datasets ------------------------------------------------------
download_dataset() {
    local repo="$1"
    local local_dir="$2"
    local short_name="$3"

    if already_downloaded "$local_dir"; then
        echo "[dataset] $short_name already downloaded at $local_dir (skip)"
        return 0
    fi
    echo ""
    echo "[dataset] Downloading $short_name from $repo -> $local_dir"
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='$repo',
    repo_type='dataset',
    local_dir='$local_dir',
    max_workers=8,
)
print('OK: $short_name')
"
    mark_done "$local_dir"
}

download_dataset "openai/gsm8k" "$DATASETS_DIR/gsm8k" "GSM8K"
download_dataset "EleutherAI/hendrycks_math" "$DATASETS_DIR/math" "MATH (Hendrycks)"

# --- Download models --------------------------------------------------------
download_model() {
    local repo="$1"
    local local_dir="$2"
    local short_name="$3"

    if already_downloaded "$local_dir"; then
        echo "[model] $short_name already downloaded at $local_dir (skip)"
        return 0
    fi
    echo ""
    echo "[model] Downloading $short_name from $repo -> $local_dir"
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='$repo',
    local_dir='$local_dir',
    max_workers=8,
    allow_patterns=[
        '*.json', '*.txt', '*.md',
        '*.safetensors', '*.safetensors.index.json',
        '*.jinja', 'tokenizer*', 'vocab*', 'merges*',
    ],
    ignore_patterns=['*.bin', '*.msgpack', '*.h5', '*.onnx', '*.gguf'],
)
print('OK: $short_name')
"
    mark_done "$local_dir"
}

# Teacher
download_model "Qwen/Qwen3.5-27B" "$MODELS_DIR/Qwen3.5-27B" "Qwen3.5-27B (Teacher)"

# Student A (planner)
download_model "Qwen/Qwen3.5-9B" "$MODELS_DIR/Qwen3.5-9B" "Qwen3.5-9B (Student A)"

# Student B (transfer)
download_model "Qwen/Qwen3.5-4B" "$MODELS_DIR/Qwen3.5-4B" "Qwen3.5-4B (Student B)"

# --- Summary ----------------------------------------------------------------
echo ""
echo "============================================================"
echo " Download Summary"
echo "============================================================"
du -sh "$DATASETS_DIR"/* "$MODELS_DIR"/* 2>/dev/null || true
echo ""
TOTAL=$(du -sh "$ASSETS_DIR" 2>/dev/null | awk '{print $1}')
echo " Total size: ${TOTAL}"
echo "============================================================"
echo ""
echo " To transfer to tju-hpc:"
echo "   rsync -avP --progress $ASSETS_DIR/ user@tju-hpc:/path/to/project/assets/"
echo ""
echo " On tju-hpc, set env vars before running experiments:"
echo "   export HF_HOME=\$PWD/assets/hf_cache"
echo "   export TRANSFORMERS_OFFLINE=1"
echo "   export HF_DATASETS_OFFLINE=1"
echo "   # Use local paths in configs/template_config.yaml:"
echo "   #   teacher.model: ./assets/models/Qwen3.5-27B"
echo "   #   planner.model: ./assets/models/Qwen3.5-9B"
echo ""
echo " Done at $(date)"
