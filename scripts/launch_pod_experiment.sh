#!/usr/bin/env bash
# Waits for download_assets.sh to finish, then launches Phase 0 extraction
# using local model paths. Designed to run inside the k8s pod.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/root/nips-templatebank}"
ASSETS_DIR="${ASSETS_DIR:-/root/assets}"
DOWNLOAD_LOG="${DOWNLOAD_LOG:-/root/download-logs/download.log}"
EXP_LOG_DIR="${EXP_LOG_DIR:-/root/experiment-logs}"

mkdir -p "$EXP_LOG_DIR"

# --- Wait for all assets to be ready ---------------------------------------
wait_for_download() {
    echo "[launcher] Waiting for download_assets.sh to finish..."
    local required_markers=(
        "$ASSETS_DIR/datasets/gsm8k/.download_done"
        "$ASSETS_DIR/datasets/math/.download_done"
        "$ASSETS_DIR/models/Qwen3.5-27B/.download_done"
        "$ASSETS_DIR/models/Qwen3.5-9B/.download_done"
        "$ASSETS_DIR/models/Qwen3.5-4B/.download_done"
    )
    until {
        all_ready=true
        for m in "${required_markers[@]}"; do
            [[ -f "$m" ]] || { all_ready=false; break; }
        done
        $all_ready
    }; do
        echo "[$(date '+%H:%M:%S')] still downloading..."
        du -sh "$ASSETS_DIR"/models/* 2>/dev/null | tail -5
        sleep 120
    done
    echo "[launcher] All assets ready."
    du -sh "$ASSETS_DIR"/*
}

# --- Patch config to use local paths ---------------------------------------
patch_config() {
    local cfg="$PROJECT_DIR/configs/template_config.yaml"
    local backup="$cfg.bak"
    if [ ! -f "$backup" ]; then
        cp "$cfg" "$backup"
    fi
    sed -i "s|Qwen/Qwen3.5-27B|$ASSETS_DIR/models/Qwen3.5-27B|g" "$cfg"
    sed -i "s|Qwen/Qwen3.5-9B|$ASSETS_DIR/models/Qwen3.5-9B|g" "$cfg"
    sed -i "s|Qwen/Qwen3.5-4B|$ASSETS_DIR/models/Qwen3.5-4B|g" "$cfg"
    echo "[launcher] Config patched to local paths:"
    grep -A1 "model:" "$cfg" | head -4
}

# --- Launch Phase 0 extraction ---------------------------------------------
launch_phase0() {
    cd "$PROJECT_DIR"
    export HF_HOME="$ASSETS_DIR/hf_cache"
    export TRANSFORMERS_OFFLINE=1
    export HF_DATASETS_OFFLINE=0   # datasets loaded from HF cache via snapshot_download
    export TOKENIZERS_PARALLELISM=false
    mkdir -p "$EXP_LOG_DIR"

    echo "[launcher] Starting Phase 0 extraction at $(date)"
    echo "[launcher] Log: $EXP_LOG_DIR/phase0.log"

    # Stage 1: teacher extraction
    nohup python3 scripts/extract_templates.py \
        --config configs/template_config.yaml \
        --output_dir results/templates_pod \
        >> "$EXP_LOG_DIR/phase0.log" 2>&1 &
    local pid=$!
    echo "$pid" > "$EXP_LOG_DIR/phase0.pid"
    echo "[launcher] Phase 0 started in background, pid=$pid"
    echo "[launcher] Follow with: tail -f $EXP_LOG_DIR/phase0.log"
}

main() {
    wait_for_download
    patch_config
    launch_phase0
}

main "$@"
