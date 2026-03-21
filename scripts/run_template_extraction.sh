#!/usr/bin/env bash
set -euo pipefail

# Activate venv if available
_PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "$_PROJ_ROOT/.venv/bin/activate" ]; then source "$_PROJ_ROOT/.venv/bin/activate"; fi
export PATH="$HOME/.local/bin:$PATH"

export HF_ENDPOINT="https://hf-mirror.com"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG="${PROJECT_DIR}/configs/template_config.yaml"

cd "$PROJECT_DIR"
mkdir -p outputs/templates outputs/compiler outputs/eval

echo "========================================"
echo "  TemplateBank: Template Algebra"
echo "  Teacher: Qwen/Qwen3.5-27B"
echo "  Student: Qwen/Qwen3.5-9B"
echo "  GPUs: 8x A100-80GB"
echo "========================================"

echo "=== Step 1: Extract templates from CoT traces ==="
python scripts/extract_templates.py \
    --config "$CONFIG" \
    --output_dir outputs/templates \
    2>&1 | tee outputs/templates/extract.log

echo ""
echo "=== Step 2: Train template compiler ==="
torchrun \
    --nproc_per_node=8 \
    --master_port=29800 \
    scripts/train_template_compiler.py \
        --config "$CONFIG" \
        --training_data outputs/templates/compiler_training_data.json \
        --output_dir outputs/compiler \
    2>&1 | tee outputs/compiler/train.log

echo ""
echo "=== Step 3: Evaluate ==="
python scripts/eval_template_algebra.py \
    --config "$CONFIG" \
    --compiler_dir outputs/compiler \
    --template_bank outputs/templates/template_bank.json \
    --output_dir outputs/eval \
    2>&1 | tee outputs/eval/eval.log

echo ""
echo "=== Pipeline complete ==="
