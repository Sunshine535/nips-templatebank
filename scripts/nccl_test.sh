#!/usr/bin/env bash
# nccl-tests keep-alive supervisor — prevents k8s auto-release (2h <5% GPU rule).
# Usage: ./nccl_test.sh {start|stop|status}
# Ref: /home/tarkoy/nips/k8s操作.pdf

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NCCL_TESTS_DIR="${NCCL_TESTS_DIR:-/root/nccl-tests}"
BIN_PATH="${BIN_PATH:-$NCCL_TESTS_DIR/build/all_reduce_perf}"
LOG_DIR="${LOG_DIR:-/root/nccl-keepalive-logs}"
PID_FILE="${PID_FILE:-$LOG_DIR/nccl-tests.pid}"
RUN_LOG="${RUN_LOG:-$LOG_DIR/nccl-tests.log}"
DEFAULT_ARGS=(-b 8M -e 256M -f 2 -n 500000 -w 5 -c 0 -z 0)
RESTART_DELAY="${RESTART_DELAY:-1}"

mkdir -p "$LOG_DIR"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(timestamp)] $*" | tee -a "$RUN_LOG"; }

detect_gpus() {
    if [[ -n "${NGPUS:-}" ]]; then echo "$NGPUS"; return 0; fi
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi -L | wc -l | awk '{print $1}'
        return 0
    fi
    echo 1
}

check_binary() {
    if [[ -x "$BIN_PATH" ]]; then return 0; fi
    echo "Binary not found: $BIN_PATH" >&2
    echo "Build nccl-tests first: cd $NCCL_TESTS_DIR && make -j" >&2
    exit 1
}

is_running() {
    [[ -f "$PID_FILE" ]] || return 1
    local pid
    pid="$(cat "$PID_FILE")"
    [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

run_loop() {
    check_binary
    local gpus extra_args=()
    local child_pid=""
    gpus="$(detect_gpus)"

    if [[ $# -gt 0 ]]; then
        extra_args=("$@")
    elif [[ -n "${NCCL_ARGS:-}" ]]; then
        extra_args=(${NCCL_ARGS})
    else
        extra_args=("${DEFAULT_ARGS[@]}")
    fi

    log "Starting nccl-tests supervisor, binary=$BIN_PATH, gpus=$gpus, args=${extra_args[*]}"

    cleanup() {
        if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then
            log "Stopping child process pid=$child_pid"
            kill "$child_pid" 2>/dev/null || true
            wait "$child_pid" 2>/dev/null || true
        fi
        rm -f "$PID_FILE"
        exit 0
    }
    trap cleanup INT TERM

    while true; do
        log "Launching: $BIN_PATH -g $gpus ${extra_args[*]}"
        set +e
        "$BIN_PATH" -g "$gpus" "${extra_args[@]}" >>"$RUN_LOG" 2>&1 &
        child_pid=$!
        wait "$child_pid"
        local exit_code=$?
        child_pid=""
        set -e
        log "nccl-tests exited with code $exit_code, restarting in ${RESTART_DELAY}s"
        sleep "$RESTART_DELAY"
    done
}

start() {
    if is_running; then
        echo "nccl-tests is already running, pid=$(cat "$PID_FILE")"
        exit 0
    fi
    nohup "$0" run "$@" >>"$RUN_LOG" 2>&1 &
    local pid=$!
    echo "$pid" >"$PID_FILE"
    sleep 2
    if kill -0 "$pid" 2>/dev/null; then
        echo "nccl-tests started in background, pid=$pid"
        echo "log: $RUN_LOG"
    else
        echo "failed to start nccl-tests, check log: $RUN_LOG" >&2
        exit 1
    fi
}

stop() {
    if ! is_running; then
        echo "nccl-tests is not running"
        rm -f "$PID_FILE"
        exit 0
    fi
    local pid
    pid="$(cat "$PID_FILE")"
    kill "$pid"
    rm -f "$PID_FILE"
    echo "stopped nccl-tests, pid=$pid"
}

status() {
    if is_running; then
        echo "nccl-tests is running, pid=$(cat "$PID_FILE")"
        echo "log: $RUN_LOG"
    else
        echo "nccl-tests is not running"
        exit 1
    fi
}

usage() {
    cat <<EOF
Usage: $0 {start|stop|status|run} [nccl-test args...]

Keep-alive for k8s pods: runs nccl-tests all_reduce_perf in a restart loop
to keep GPU utilization above 5%, preventing auto-release (2h rule).

Environment variables:
  NGPUS         override GPU count (default: auto-detect)
  NCCL_ARGS     override default nccl args
  LOG_DIR       log directory (default: /root/nccl-keepalive-logs)
  RESTART_DELAY seconds between restarts (default: 5)
EOF
}

cmd="${1:-start}"
[[ $# -gt 0 ]] && shift

case "$cmd" in
    start)  start "$@" ;;
    stop)   stop ;;
    status) status ;;
    run)    run_loop "$@" ;;
    *)      usage; exit 1 ;;
esac
