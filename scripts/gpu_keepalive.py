#!/usr/bin/env python3
"""GPU keep-alive: lightweight matmul loop to maintain GPU utilization >5%.

Falls back to this when nccl-tests binary is not available.
Uses ~200MB GPU memory per GPU, ~10-15% utilization.

Usage:
    python3 gpu_keepalive.py                  # all GPUs
    CUDA_VISIBLE_DEVICES=3 python3 gpu_keepalive.py  # single GPU
    python3 gpu_keepalive.py --interval 0.5   # less aggressive
"""
import argparse
import os
import signal
import sys
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=float, default=0.1,
                        help="seconds between matmul bursts (lower=higher util)")
    parser.add_argument("--size", type=int, default=2048,
                        help="matrix dimension (NxN)")
    parser.add_argument("--bursts", type=int, default=5,
                        help="matmuls per cycle")
    args = parser.parse_args()

    try:
        import torch
    except ImportError:
        print("torch not available, exiting", file=sys.stderr)
        sys.exit(1)

    if not torch.cuda.is_available():
        print("CUDA not available, exiting", file=sys.stderr)
        sys.exit(1)

    n_gpus = torch.cuda.device_count()
    print(f"[keepalive] Starting GPU keep-alive on {n_gpus} GPU(s), "
          f"size={args.size}, interval={args.interval}s, bursts={args.bursts}")

    tensors = []
    for i in range(n_gpus):
        a = torch.randn(args.size, args.size, device=f"cuda:{i}", dtype=torch.float16)
        b = torch.randn(args.size, args.size, device=f"cuda:{i}", dtype=torch.float16)
        tensors.append((a, b))
        mem_mb = torch.cuda.memory_allocated(i) / 1024**2
        print(f"[keepalive] GPU {i}: allocated {mem_mb:.0f} MB")

    running = True
    def handle_signal(sig, frame):
        nonlocal running
        running = False
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    cycle = 0
    while running:
        for i, (a, b) in enumerate(tensors):
            for _ in range(args.bursts):
                torch.mm(a, b)
            torch.cuda.synchronize(i)
        cycle += 1
        if cycle % 1000 == 0:
            print(f"[keepalive] cycle={cycle}, alive")
        time.sleep(args.interval)

    print("[keepalive] Stopped.")

if __name__ == "__main__":
    main()
