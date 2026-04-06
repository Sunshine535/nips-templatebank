#!/usr/bin/env bash
# Wait for extraction PID to finish, then run gate + pipeline
# Usage: nohup bash scripts/wait_and_run.sh > logs/wait_and_run.log 2>&1 &

set -euo pipefail
cd "$(dirname "$0")/.."

EXTRACTION_PID=${1:-48222}

echo "[$(date)] Waiting for extraction PID $EXTRACTION_PID to complete..."

# Wait for extraction to finish
while kill -0 "$EXTRACTION_PID" 2>/dev/null; do
    sleep 60
    # Show progress every minute
    grep "\[INFO\]" logs/extraction_full.log 2>/dev/null | tail -1
done

echo "[$(date)] Extraction complete! Running gate diagnostics..."

# Run gate
bash scripts/post_extraction_gate.sh

# Check gate result
if grep -q "GATE: GO\|GATE: MARGINAL" logs/gate_results.log 2>/dev/null; then
    echo "[$(date)] Gate passed! Starting full pipeline..."
    bash scripts/run_full_pipeline.sh
else
    echo "[$(date)] Gate FAILED. Check logs/gate_results.log for details."
    echo "Manual intervention needed."
fi
