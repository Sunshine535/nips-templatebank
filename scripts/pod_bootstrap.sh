#!/usr/bin/env bash
# =============================================================================
# One-shot bootstrap for k8s pod. Installs deps, starts GPU keep-alive, sets up
# auto-recovery so EVERYTHING restarts automatically after pod restarts.
#
# Run ONCE inside the pod:
#   bash /root/nips-templatebank/scripts/pod_bootstrap.sh
#
# After this, pod restarts are handled by watchdog — no manual intervention.
# =============================================================================
set -e

PROJ_DIR="/root/nips-templatebank"
SCRIPTS="$PROJ_DIR/scripts"

export http_proxy=http://192.168.3.226:7890
export https_proxy=http://192.168.3.226:7890
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$HOME/.local/lib/python3.10/site-packages:${PYTHONPATH:-}"

# ---- Step 1: Install persistent Python deps (idempotent) -------------------
echo "[bootstrap] === Install persistent Python deps ==="
pip install --user --upgrade \
    "transformers>=5.5" "trl>=0.12" "peft>=0.13" "accelerate>=1.0" \
    "datasets>=4.0" "huggingface_hub>=1.0" hf_transfer \
    wandb pyyaml evaluate sentence-transformers rank-bm25 outlines 2>&1 | tail -5

python3 -c "
import sys; sys.path.insert(0, '$HOME/.local/lib/python3.10/site-packages')
import transformers; print(f'transformers={transformers.__version__}')
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained('/root/assets/models/Qwen3.5-27B')
print(f'model_type={cfg.model_type} OK')
"

# ---- Step 2: Copy scripts to /root/ for easy access ------------------------
echo "[bootstrap] === Copy scripts to /root/ ==="
cp "$SCRIPTS/nccl_test.sh" /root/nccl_test.sh
cp "$SCRIPTS/gpu_keepalive.py" /root/gpu_keepalive.py
chmod +x /root/nccl_test.sh

# ---- Step 3: Create auto-recovery script -----------------------------------
echo "[bootstrap] === Create auto-recovery script ==="
cat > /root/autostart.sh << 'AUTOSTART'
#!/usr/bin/env bash
# Called by watchdog every 60s. Ensures GPU keep-alive + experiment are alive.
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$HOME/.local/lib/python3.10/site-packages:${PYTHONPATH:-}"
export http_proxy=http://192.168.3.226:7890
export https_proxy=http://192.168.3.226:7890

LOGFILE=/root/autostart.log
KEEPALIVE_PID_FILE=/root/gpu_keepalive.pid

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOGFILE"; }

# --- GPU keep-alive (Python matmul — no build required) ---------------------
if [ -f "$KEEPALIVE_PID_FILE" ] && kill -0 "$(cat "$KEEPALIVE_PID_FILE")" 2>/dev/null; then
    : # already running
else
    log "GPU keep-alive not running, starting..."
    # Try nccl-tests first, fall back to Python matmul
    if [ -x /root/nccl-tests/build/all_reduce_perf ]; then
        log "Using nccl-tests keep-alive"
        setsid /root/nccl_test.sh start > /dev/null 2>&1 < /dev/null
        sleep 2
        if /root/nccl_test.sh status > /dev/null 2>&1; then
            log "nccl-tests started OK"
            # Store nccl supervisor PID for tracking
            cat /root/nccl-keepalive-logs/nccl-tests.pid > "$KEEPALIVE_PID_FILE" 2>/dev/null || true
        else
            log "nccl-tests failed, falling back to Python keepalive"
            setsid python3 /root/gpu_keepalive.py \
                --interval 0.1 --size 2048 --bursts 5 \
                >> /root/gpu_keepalive.log 2>&1 < /dev/null &
            echo $! > "$KEEPALIVE_PID_FILE"
            disown $! 2>/dev/null || true
            log "Python keepalive started, pid=$(cat $KEEPALIVE_PID_FILE)"
        fi
    else
        log "nccl binary not found, using Python keepalive"
        setsid python3 /root/gpu_keepalive.py \
            --interval 0.1 --size 2048 --bursts 5 \
            >> /root/gpu_keepalive.log 2>&1 < /dev/null &
        echo $! > "$KEEPALIVE_PID_FILE"
        disown $! 2>/dev/null || true
        log "Python keepalive started, pid=$(cat $KEEPALIVE_PID_FILE)"
    fi
fi

# --- Phase 0 extraction (only start if not already done) --------------------
PHASE0_PID_FILE=/root/experiment-logs/phase0.pid
PHASE0_DONE=/root/nips-templatebank/results/templates_pod/.extraction_done

if [ -f "$PHASE0_DONE" ]; then
    : # Extraction already completed
elif [ -f "$PHASE0_PID_FILE" ] && kill -0 "$(cat "$PHASE0_PID_FILE")" 2>/dev/null; then
    : # Already running
else
    log "Phase 0 not running, starting..."
    mkdir -p /root/experiment-logs
    cd /root/nips-templatebank
    setsid env \
        PATH="$HOME/.local/bin:$PATH" \
        PYTHONPATH="$HOME/.local/lib/python3.10/site-packages:${PYTHONPATH:-}" \
        http_proxy=http://192.168.3.226:7890 \
        https_proxy=http://192.168.3.226:7890 \
        HF_ENDPOINT=https://huggingface.co \
        TOKENIZERS_PARALLELISM=false \
        python3 scripts/extract_templates.py \
            --config configs/template_config.yaml \
            --output_dir results/templates_pod \
        > /root/experiment-logs/phase0.log 2>&1 < /dev/null &
    echo $! > "$PHASE0_PID_FILE"
    disown $! 2>/dev/null || true
    log "Phase 0 started, pid=$(cat $PHASE0_PID_FILE)"
fi
AUTOSTART
chmod +x /root/autostart.sh

# ---- Step 4: Create watchdog daemon ----------------------------------------
echo "[bootstrap] === Create watchdog daemon ==="
cat > /root/watchdog.sh << 'WATCHDOG'
#!/usr/bin/env bash
# Long-running watchdog: checks GPU keepalive + experiment every 60s.
LOGFILE=/root/autostart.log
PIDFILE=/root/watchdog.pid
echo $$ > "$PIDFILE"
echo "[$(date)] Watchdog started, pid=$$" >> "$LOGFILE"
while true; do
    bash /root/autostart.sh
    sleep 60
done
WATCHDOG
chmod +x /root/watchdog.sh

# Kill old watchdog if any
OLD_WD=$(cat /root/watchdog.pid 2>/dev/null || echo '')
if [ -n "$OLD_WD" ] && kill -0 "$OLD_WD" 2>/dev/null; then
    kill "$OLD_WD" 2>/dev/null || true
    sleep 1
fi

# Start watchdog fully detached
setsid /root/watchdog.sh > /dev/null 2>&1 < /dev/null &
disown $! 2>/dev/null || true
sleep 2
echo "Watchdog pid=$(cat /root/watchdog.pid)"

# ---- Step 5: Add to .bashrc for auto-start on pod restart ------------------
echo "[bootstrap] === Add auto-start to .bashrc ==="
if ! grep -q "watchdog.sh" /root/.bashrc 2>/dev/null; then
    cat >> /root/.bashrc << 'BASHRC'

# Auto-start watchdog on login (pod restart recovery)
if [ ! -f /root/watchdog.pid ] || ! kill -0 "$(cat /root/watchdog.pid 2>/dev/null)" 2>/dev/null; then
    setsid /root/watchdog.sh > /dev/null 2>&1 < /dev/null &
    disown $! 2>/dev/null || true
    echo "[autostart] Watchdog started, pid=$(cat /root/watchdog.pid)"
fi
BASHRC
    echo "Added to .bashrc"
else
    echo "Already in .bashrc"
fi

# ---- Step 6: Wait for first cycle -----------------------------------------
echo "[bootstrap] === Waiting for first autostart cycle ==="
sleep 15

echo ""
echo "[bootstrap] === Final status ==="
# GPU keep-alive
KA_PID=$(cat /root/gpu_keepalive.pid 2>/dev/null || echo '')
if [ -n "$KA_PID" ] && kill -0 "$KA_PID" 2>/dev/null; then
    echo "GPU keep-alive: RUNNING pid=$KA_PID"
else
    echo "GPU keep-alive: NOT RUNNING - check logs"
fi
# Phase 0
PID=$(cat /root/experiment-logs/phase0.pid 2>/dev/null || echo '')
if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
    echo "Phase 0: RUNNING pid=$PID"
else
    echo "Phase 0: NOT RUNNING - check /root/experiment-logs/phase0.log"
fi
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv
echo ""
echo "[bootstrap] Done. Watchdog will auto-recover both processes every 60s."
