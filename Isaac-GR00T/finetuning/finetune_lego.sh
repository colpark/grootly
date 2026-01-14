#!/bin/bash
# Finetuning script for trossen_lego dataset (Trossen AI Mobile robot)
#
# Prerequisites:
#   1. Download dataset: python scripts/download_hf_dataset.py TrossenRoboticsCommunity/trossen_ai_mobile_lego --name trossen_lego
#   2. Convert dataset: python finetuning/convert_lego.py
#   3. Ensure training data exists at: ./data/lego_train
#
# Usage:
#   ./finetuning/finetune_lego.sh
#
# For multi-GPU training (recommended):
#   ./finetuning/finetune_lego.sh 4

set -e

# Default settings
NUM_GPUS=${1:-4}
DATASET_PATH="${DATASET_PATH:-./data/lego_train}"
BASE_MODEL="${BASE_MODEL:-nvidia/GR00T-N1.6-3B}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/lego}"

echo "=================================================="
echo "LEGO Manipulation GR00T Finetuning"
echo "=================================================="
echo "Dataset: ${DATASET_PATH}"
echo "Base Model: ${BASE_MODEL}"
echo "Output: ${OUTPUT_DIR}"
echo "GPUs: ${NUM_GPUS}"
echo "=================================================="

# Check if dataset exists
if [ ! -d "${DATASET_PATH}" ]; then
    echo "ERROR: Dataset not found at ${DATASET_PATH}"
    echo "Please run:"
    echo "  1. python scripts/download_hf_dataset.py TrossenRoboticsCommunity/trossen_ai_mobile_lego --name trossen_lego"
    echo "  2. python finetuning/convert_lego.py"
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
# Note: Uses same modality config as ball2_groot and plug_stacking (same robot: Trossen AI Mobile)
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
echo "  python finetuning/evaluate_lego.py --checkpoint_path ${OUTPUT_DIR}/checkpoint-XXXX"
echo ""
echo "Or batch evaluate:"
echo "  ./finetuning/evaluate_lego_checkpoints.sh --all"
echo "=================================================="
