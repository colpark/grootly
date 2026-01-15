#!/bin/bash
# Evaluate plug_stacking model at multiple checkpoint steps
#
# Usage:
#   ./finetuning/evaluate_plug_stacking_checkpoints.sh 1000 2000 3000 4000 5000
#   ./finetuning/evaluate_plug_stacking_checkpoints.sh --all  # Evaluate all checkpoints
#
# Or with custom output dir:
#   OUTPUT_DIR=./my_results ./finetuning/evaluate_plug_stacking_checkpoints.sh 1000 2000

set -e

CHECKPOINT_DIR="${CHECKPOINT_DIR:-./outputs/plug_stacking}"
DATASET_PATH="${DATASET_PATH:-./data/plug_stacking_test}"
OUTPUT_BASE="${OUTPUT_DIR:-./eval_results/plug_stacking}"

# Parse arguments
if [[ "$1" == "--all" ]]; then
    # Find all checkpoints
    STEPS=$(ls -d ${CHECKPOINT_DIR}/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n)
    if [[ -z "$STEPS" ]]; then
        echo "ERROR: No checkpoints found in ${CHECKPOINT_DIR}"
        exit 1
    fi
else
    STEPS="$@"
fi

if [[ -z "$STEPS" ]]; then
    echo "Usage: $0 <step1> <step2> ... OR $0 --all"
    echo ""
    echo "Example:"
    echo "  $0 1000 2000 3000 5000 10000"
    echo "  $0 --all"
    echo ""
    echo "Environment variables:"
    echo "  CHECKPOINT_DIR  Directory containing checkpoints (default: ./outputs/plug_stacking)"
    echo "  DATASET_PATH    Path to test dataset (default: ./data/plug_stacking_test)"
    echo "  OUTPUT_DIR      Output directory for results (default: ./eval_results/plug_stacking)"
    exit 1
fi

echo "=================================================="
echo "Plug Stacking GR00T Checkpoint Evaluation"
echo "=================================================="
echo "Checkpoint dir: ${CHECKPOINT_DIR}"
echo "Dataset: ${DATASET_PATH}"
echo "Output: ${OUTPUT_BASE}"
echo "Steps to evaluate: ${STEPS}"
echo "=================================================="

# Check dataset exists
if [[ ! -d "${DATASET_PATH}" ]]; then
    echo "ERROR: Dataset not found at ${DATASET_PATH}"
    exit 1
fi

# Create summary file
SUMMARY_FILE="${OUTPUT_BASE}/checkpoint_comparison.csv"
mkdir -p "${OUTPUT_BASE}"
echo "step,mse,mae,success_005,success_01,success_02,base_mae,left_arm_mae,right_arm_mae" > "${SUMMARY_FILE}"

# Evaluate each checkpoint
for STEP in ${STEPS}; do
    CHECKPOINT="${CHECKPOINT_DIR}/checkpoint-${STEP}"

    if [[ ! -d "${CHECKPOINT}" ]]; then
        echo "WARNING: Checkpoint not found: ${CHECKPOINT}, skipping..."
        continue
    fi

    echo ""
    echo "=================================================="
    echo "Evaluating checkpoint-${STEP}"
    echo "=================================================="

    OUTPUT_STEP="${OUTPUT_BASE}/step_${STEP}"
    mkdir -p "${OUTPUT_STEP}"

    python finetuning/evaluate_plug_stacking.py \
        --checkpoint_path "${CHECKPOINT}" \
        --dataset_path "${DATASET_PATH}" \
        --output_dir "${OUTPUT_STEP}" \
        --steps 200 \
        2>&1 | tee "${OUTPUT_STEP}/eval_log.txt"

    # Extract metrics from summary file and append to CSV
    if [[ -f "${OUTPUT_STEP}/evaluation_summary.txt" ]]; then
        MSE=$(grep "Average MSE:" "${OUTPUT_STEP}/evaluation_summary.txt" | awk '{print $3}')
        MAE=$(grep "Average MAE:" "${OUTPUT_STEP}/evaluation_summary.txt" | awk '{print $3}')
        SUCCESS_005=$(grep "@ 0.05 threshold:" "${OUTPUT_STEP}/evaluation_summary.txt" | head -1 | awk '{print $4}' | tr -d '%')
        SUCCESS_01=$(grep "@ 0.10 threshold:" "${OUTPUT_STEP}/evaluation_summary.txt" | head -1 | awk '{print $4}' | tr -d '%')
        SUCCESS_02=$(grep "@ 0.20 threshold:" "${OUTPUT_STEP}/evaluation_summary.txt" | head -1 | awk '{print $4}' | tr -d '%')
        BASE_MAE=$(grep "Base Velocity:" "${OUTPUT_STEP}/evaluation_summary.txt" | awk '{print $3}')
        LEFT_MAE=$(grep "Left Arm:" "${OUTPUT_STEP}/evaluation_summary.txt" | head -1 | awk '{print $3}')
        RIGHT_MAE=$(grep "Right Arm:" "${OUTPUT_STEP}/evaluation_summary.txt" | head -1 | awk '{print $3}')

        echo "${STEP},${MSE},${MAE},${SUCCESS_005},${SUCCESS_01},${SUCCESS_02},${BASE_MAE},${LEFT_MAE},${RIGHT_MAE}" >> "${SUMMARY_FILE}"
    fi
done

echo ""
echo "=================================================="
echo "EVALUATION COMPLETE"
echo "=================================================="
echo ""
echo "Results saved to: ${OUTPUT_BASE}"
echo "Comparison CSV: ${SUMMARY_FILE}"
echo ""
echo "Summary:"
cat "${SUMMARY_FILE}"
