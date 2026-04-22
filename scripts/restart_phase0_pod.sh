#!/usr/bin/env bash
# Restart Phase 0 extraction inside k8s pod with proxy-enabled networking.
# Models load from local dir (TRANSFORMERS_OFFLINE=1), datasets via proxy.
set -e

cd /root/nips-templatebank

# Kill any existing extract_templates processes (by specific pattern, not pkill)
OLD_PID=$(pgrep -f 'scripts/extract_templates.py' | head -1 || true)
if [ -n "$OLD_PID" ]; then
    echo "Killing old extract_templates PID=$OLD_PID"
    kill -9 "$OLD_PID" 2>/dev/null || true
    sleep 2
fi

# Environment for Phase 0
# Model loads from local dir (absolute path in config) regardless of online flag.
# Datasets lib uses /root/.cache/huggingface/datasets (populated from earlier run).
export http_proxy=http://192.168.3.226:7890
export https_proxy=http://192.168.3.226:7890
export HF_ENDPOINT=https://huggingface.co
unset TRANSFORMERS_OFFLINE
unset HF_DATASETS_OFFLINE
unset HF_HUB_OFFLINE
export TOKENIZERS_PARALLELISM=false
mkdir -p /root/experiment-logs

# Launch fully detached via setsid (survives kubectl exec session termination)
setsid python3 scripts/extract_templates.py \
    --config configs/template_config.yaml \
    --output_dir results/templates_pod \
    > /root/experiment-logs/phase0.log 2>&1 < /dev/null &

echo $! > /root/experiment-logs/phase0.pid
disown $! 2>/dev/null || true
echo "Phase 0 started (detached), pid=$(cat /root/experiment-logs/phase0.pid)"
echo "Follow log: tail -f /root/experiment-logs/phase0.log"
