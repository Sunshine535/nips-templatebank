#!/usr/bin/env bash
set -euo pipefail

#####################################################################
#  Template Algebra: Full Experiment Pipeline
#
#  Pipeline: extract_templates → template_operations → train_compiler
#            → eval → ablations → paper tables
#
#  Requirements: 8x GPU (A100-80GB recommended)
#  Estimated time: ~18-24 hours total
#####################################################################

export HF_ENDPOINT="https://hf-mirror.com"
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG="${PROJECT_DIR}/configs/template_config.yaml"
NUM_GPUS=${NUM_GPUS:-8}

TEMPLATE_DIR="${PROJECT_DIR}/results/templates"
OPS_DIR="${PROJECT_DIR}/results/operations"
COMPILER_DIR="${PROJECT_DIR}/results/compiler"
EVAL_DIR="${PROJECT_DIR}/results/eval"

cd "$PROJECT_DIR"
mkdir -p results

echo "================================================================"
echo "  Template Algebra — Full Experiment Pipeline"
echo "  Model    : Qwen/Qwen3.5-9B"
echo "  Data     : GSM8K (7473 train) + MATH (7500 train)"
echo "  LoRA     : r=16, alpha=32"
echo "  GPUs     : ${NUM_GPUS}"
echo "  Config   : ${CONFIG}"
echo "  Started  : $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "================================================================"

# ==========================================
#  STAGE 1: Extract Templates from CoT Traces
# ==========================================
echo ""
echo "========== STAGE 1: Extract Templates =========="
python scripts/extract_templates.py \
    --config "$CONFIG" \
    --output_dir "$TEMPLATE_DIR" \
    2>&1 | tee results/stage1_extract.log

echo "  [DONE] Stage 1 — Template extraction complete"
echo "  Bank: ${TEMPLATE_DIR}/template_bank.json"

# ==========================================
#  STAGE 2: Test Template Algebra Operations
# ==========================================
echo ""
echo "========== STAGE 2: Template Operations =========="
python scripts/run_template_operations.py \
    --config "$CONFIG" \
    --template_bank "${TEMPLATE_DIR}/template_bank.json" \
    --output_dir "$OPS_DIR" \
    2>&1 | tee results/stage2_operations.log

echo "  [DONE] Stage 2 — Template operations complete"

# ==========================================
#  STAGE 3: Train Template Compiler (Two-Stage SFT)
# ==========================================
echo ""
echo "========== STAGE 3: Train Compiler =========="

echo "  --- Stage 3a: Template Selection SFT ---"
torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=29800 \
    scripts/train_template_compiler.py \
        --config "$CONFIG" \
        --training_data "${TEMPLATE_DIR}/compiler_training_data.json" \
        --output_dir "$COMPILER_DIR" \
        --skip_stage2 \
    2>&1 | tee results/stage3a_selection.log

echo "  --- Stage 3b: Variable Filling SFT ---"
torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=29810 \
    scripts/train_template_compiler.py \
        --config "$CONFIG" \
        --training_data "${TEMPLATE_DIR}/compiler_training_data.json" \
        --output_dir "$COMPILER_DIR" \
        --skip_stage1 \
    2>&1 | tee results/stage3b_filling.log

echo "  [DONE] Stage 3 — Compiler training complete"

# ==========================================
#  STAGE 4: Full Evaluation
# ==========================================
echo ""
echo "========== STAGE 4: Evaluation =========="
python scripts/eval_template_reasoning.py \
    --config "$CONFIG" \
    --compiler_dir "${COMPILER_DIR}/stage2_filling" \
    --template_bank "${TEMPLATE_DIR}/template_bank.json" \
    --output_dir "$EVAL_DIR" \
    2>&1 | tee results/stage4_eval.log

echo "  [DONE] Stage 4 — Evaluation complete"

# ==========================================
#  STAGE 5: Ablations (bank size, operation types)
# ==========================================
echo ""
echo "========== STAGE 5: Ablations =========="

for BANK_SIZE in 10 25 50 100 200 300; do
    ABLATION_DIR="${PROJECT_DIR}/results/ablation_bank_${BANK_SIZE}"
    mkdir -p "$ABLATION_DIR"
    echo "  Testing bank size: ${BANK_SIZE}"
    python scripts/run_template_operations.py \
        --config "$CONFIG" \
        --template_bank "${TEMPLATE_DIR}/template_bank.json" \
        --output_dir "$ABLATION_DIR" \
        2>&1 | tee "${ABLATION_DIR}/ablation.log"
done

echo "  [DONE] Stage 5 — Ablations complete"

# ==========================================
#  Summary
# ==========================================
echo ""
echo "================================================================"
echo "  Pipeline Complete — $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "================================================================"
echo ""
echo "  Results:"
echo "    Templates   : ${TEMPLATE_DIR}/template_bank.json"
echo "    Operations  : ${OPS_DIR}/template_operations_results.json"
echo "    Compiler    : ${COMPILER_DIR}/"
echo "    Evaluation  : ${EVAL_DIR}/eval_results.json"
echo "    Ablations   : results/ablation_bank_*/"
echo ""
echo "================================================================"
