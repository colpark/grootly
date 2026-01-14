#!/bin/bash
# Finetuning script for plug_stacking dataset (Trossen AI Mobile robot)
#
# Prerequisites:
#   1. Convert dataset: python finetuning/convert_plug_stacking.py
#   2. Ensure training data exists at: ./data/plug_stacking_train
#
# Usage:
#   ./finetuning/finetune_plug_stacking.sh
#
# For multi-GPU training (recommended):
#   ./finetuning/finetune_plug_stacking.sh 4
#
# With additional options:
#   ./finetuning/finetune_plug_stacking.sh 4 --max-steps 5000

set -e

# Default settings
NUM_GPUS=${1:-4}
shift 2>/dev/null || true  # Remove first arg from $@ if present
DATASET_PATH="${DATASET_PATH:-./data/plug_stacking_train}"
BASE_MODEL="${BASE_MODEL:-nvidia/GR00T-N1.6-3B}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/plug_stacking}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-20}"

echo "=================================================="
echo "Plug Stacking GR00T Finetuning"
echo "=================================================="
echo "Dataset: ${DATASET_PATH}"
echo "Base Model: ${BASE_MODEL}"
echo "Output: ${OUTPUT_DIR}"
echo "GPUs: ${NUM_GPUS}"
echo "Save Total Limit: ${SAVE_TOTAL_LIMIT}"
echo "=================================================="

# Check if dataset exists
if [ ! -d "${DATASET_PATH}" ]; then
    echo "ERROR: Dataset not found at ${DATASET_PATH}"
    echo "Please run: python finetuning/convert_plug_stacking.py"
    exit 1
fi

# Check if meta/info.json exists
if [ ! -f "${DATASET_PATH}/meta/info.json" ]; then
    echo "ERROR: Invalid dataset format (missing meta/info.json)"
    exit 1
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Run finetuning
# Note: Uses same modality config as ball2_groot (same robot: Trossen AI Mobile)
echo ""
echo "Starting finetuning..."
echo ""

torchrun --nproc_per_node=${NUM_GPUS} --standalone \
    -m gr00t.experiment.launch_finetune \
    --modality-config-path ./finetuning/trossen_modality_config.py \
    --embodiment-tag NEW_EMBODIMENT \
    --dataset-path "${DATASET_PATH}" \
    --base-model-path "${BASE_MODEL}" \
    --output-dir "${OUTPUT_DIR}" \
    --save-total-limit "${SAVE_TOTAL_LIMIT}" \
    "$@"

echo ""
echo "=================================================="
echo "Finetuning complete!"
echo "Checkpoints saved to: ${OUTPUT_DIR}"
echo ""
echo "To evaluate, run:"
echo "  python finetuning/evaluate_plug_stacking.py --checkpoint_path ${OUTPUT_DIR}/checkpoint-XXXX"
echo ""
echo "Or batch evaluate:"
echo "  CHECKPOINT_DIR=${OUTPUT_DIR} DATASET_PATH=./data/plug_stacking_test ./finetuning/evaluate_checkpoints.sh --all"
echo "=================================================="
