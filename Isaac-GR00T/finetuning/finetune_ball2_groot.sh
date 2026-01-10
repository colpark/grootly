#!/bin/bash
# Finetuning script for ball2_groot dataset (Trossen AI Mobile robot)
#
# Prerequisites:
#   1. Convert dataset: python finetuning/convert_ball2_groot.py
#   2. Ensure training data exists at: ./data/ball2_groot_train
#
# Usage:
#   ./finetuning/finetune_ball2_groot.sh
#
# For multi-GPU training (recommended):
#   ./finetuning/finetune_ball2_groot.sh --gpus 4

set -e

# Default settings
NUM_GPUS=${1:-4}
DATASET_PATH="${DATASET_PATH:-./data/ball2_groot_train}"
BASE_MODEL="${BASE_MODEL:-nvidia/GR00T-N1.6-3B}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/ball2_groot}"

echo "=================================================="
echo "Ball2 GR00T Finetuning"
echo "=================================================="
echo "Dataset: ${DATASET_PATH}"
echo "Base Model: ${BASE_MODEL}"
echo "Output: ${OUTPUT_DIR}"
echo "GPUs: ${NUM_GPUS}"
echo "=================================================="

# Check if dataset exists
if [ ! -d "${DATASET_PATH}" ]; then
    echo "ERROR: Dataset not found at ${DATASET_PATH}"
    echo "Please run: python finetuning/convert_ball2_groot.py"
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
    "$@"

echo ""
echo "=================================================="
echo "Finetuning complete!"
echo "Checkpoints saved to: ${OUTPUT_DIR}"
echo ""
echo "To evaluate, run:"
echo "  python finetuning/evaluate_ball2_groot.py --checkpoint_path ${OUTPUT_DIR}/checkpoint-XXXX"
echo "=================================================="
