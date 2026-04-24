#!/usr/bin/env bash
# A/B/C minimal ablation runner (GPT-5.5 Task 10).
# Runs the 4 variants defined in configs/ablations/ across multiple seeds.
#
# Usage:
#   bash scripts/run_gift_minimal_ablation.sh \
#     --seeds 42 123 456 \
#     --max_train 500 \
#     --max_eval 200
#
# Requires: pod with 4x H200 GPU, TRL, PEFT, verified programs + GIFT data.

set -euo pipefail

SEEDS=(42 123 456)
MAX_TRAIN=500
MAX_EVAL=200
VARIANTS=(old_fragment_only gift_no_call_output gift_no_active_gate full_gift_step)
OUT_BASE="results/gift_ablation"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seeds) shift; SEEDS=($1); shift;;
    --max_train) shift; MAX_TRAIN=$1; shift;;
    --max_eval) shift; MAX_EVAL=$1; shift;;
    --variants) shift; VARIANTS=($1); shift;;
    --output_base) shift; OUT_BASE=$1; shift;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

mkdir -p "${OUT_BASE}"

echo "=============================================="
echo "  GIFT A/B/C Minimal Ablation"
echo "  Seeds: ${SEEDS[*]}"
echo "  Variants: ${VARIANTS[*]}"
echo "  Train/Eval: ${MAX_TRAIN}/${MAX_EVAL}"
echo "=============================================="

for variant in "${VARIANTS[@]}"; do
  cfg="configs/ablations/${variant}.yaml"
  if [[ ! -f "$cfg" ]]; then
    echo "[WARN] Missing config: $cfg — skipping"
    continue
  fi
  for seed in "${SEEDS[@]}"; do
    out="${OUT_BASE}/${variant}/seed${seed}"
    mkdir -p "$out"
    echo ""
    echo "--- VARIANT: ${variant} SEED: ${seed} ---"
    echo "  Config: $cfg"
    echo "  Output: $out"
    echo "  Command (simulated): "
    echo "    python3 scripts/train_template_compiler.py \\"
    echo "      --config $cfg \\"
    echo "      --mode compose \\"
    echo "      --output_dir $out \\"
    echo "      --seed $seed \\"
    echo "      --no_resume \\"
    echo "      --training_data <variant-specific>"
    echo "    python3 scripts/eval_template_reasoning.py \\"
    echo "      --config $cfg \\"
    echo "      --compose_dir $out \\"
    echo "      --max_samples ${MAX_EVAL} \\"
    echo "      --require_adapters --no_fallback \\"
    echo "      --output_dir ${out}/eval"
  done
done

echo ""
echo "=============================================="
echo "  Aggregation: run after all experiments"
echo "=============================================="
echo "python3 scripts/aggregate_ablation_results.py --base ${OUT_BASE} --seeds ${SEEDS[*]}"
