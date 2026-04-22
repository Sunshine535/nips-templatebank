#!/usr/bin/env bash
# Quick status check for the k8s pod experiment
kubectl exec deploy/renf-templatebank-templatebank -- bash -c '
echo "=== GPU ==="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv
echo ""
echo "=== GPU Keep-alive ==="
KA_PID=$(cat /root/gpu_keepalive.pid 2>/dev/null || echo "")
if [ -n "$KA_PID" ] && kill -0 "$KA_PID" 2>/dev/null; then
    echo "RUNNING pid=$KA_PID"
else
    echo "NOT RUNNING"
fi
echo ""
echo "=== Watchdog ==="
WD_PID=$(cat /root/watchdog.pid 2>/dev/null || echo "")
if [ -n "$WD_PID" ] && kill -0 "$WD_PID" 2>/dev/null; then
    echo "RUNNING pid=$WD_PID"
else
    echo "NOT RUNNING"
fi
echo ""
echo "=== Phase 0 ==="
PH0_PID=$(cat /root/experiment-logs/phase0.pid 2>/dev/null || echo "")
if [ -n "$PH0_PID" ] && kill -0 "$PH0_PID" 2>/dev/null; then
    echo "RUNNING pid=$PH0_PID"
    tail -5 /root/experiment-logs/phase0.log 2>/dev/null
else
    echo "NOT RUNNING"
    tail -3 /root/experiment-logs/phase0.log 2>/dev/null || echo "no log"
fi
echo ""
echo "=== Assets ==="
du -sh /root/assets/datasets/* /root/assets/models/* 2>/dev/null || echo "no assets"
echo ""
echo "=== Autostart Log (last 5) ==="
tail -5 /root/autostart.log 2>/dev/null || echo "no log"
'
