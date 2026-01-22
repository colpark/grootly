#!/bin/bash
# GR00T Inference Server for Ball2 (Ball Transfer) Task
#
# Task: Bimanual ball transfer between arms
# Robot: Trossen AI Mobile
# Dataset: ball2_groot (inhouse)
#
# Usage:
#   ./deployment/run_server_ball2.sh
#
# Server Configuration:
#   - Host: 0.0.0.0 (all interfaces)
#   - Port: 5559
#   - Checkpoint: ./outputs/ball2_groot/checkpoint-10000

set -e

# Task-specific configuration
TASK_NAME="ball2"
CHECKPOINT="${CHECKPOINT:-./outputs/ball2_groot/checkpoint-10000}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-5559}"
DEVICE="${DEVICE:-cuda:0}"
MODALITY_CONFIG="${MODALITY_CONFIG:-./finetuning/trossen_modality_config.py}"

echo "=============================================="
echo "GR00T Server: Ball Transfer Task (Ball2)"
echo "=============================================="
echo "Task: ${TASK_NAME}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Host: ${HOST}"
echo "Port: ${PORT}"
echo "Device: ${DEVICE}"
echo "=============================================="

# Validate checkpoint exists
if [[ ! -d "${CHECKPOINT}" ]]; then
    echo "ERROR: Checkpoint not found at ${CHECKPOINT}"
    echo ""
    echo "Available Ball2 checkpoints:"
    ls -d ./outputs/ball2_groot/checkpoint-* 2>/dev/null || echo "  No checkpoints found in ./outputs/ball2_groot/"
    echo ""
    echo "To train the Ball2 model:"
    echo "  1. python finetuning/convert_ball2_groot.py"
    echo "  2. ./finetuning/finetune_ball2_groot.sh"
    exit 1
fi

# Validate modality config exists
if [[ ! -f "${MODALITY_CONFIG}" ]]; then
    echo "ERROR: Modality config not found at ${MODALITY_CONFIG}"
    exit 1
fi

echo ""
echo "Starting server..."
echo "Robot client should connect to: tcp://${HOST}:${PORT}"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the server with modality config registration
python -c "
import sys
sys.path.insert(0, '.')

# Register modality config
exec(open('${MODALITY_CONFIG}').read())

# Start the server
from gr00t.eval.run_gr00t_server import main, ServerConfig

config = ServerConfig(
    model_path='${CHECKPOINT}',
    embodiment_tag='NEW_EMBODIMENT',
    device='${DEVICE}',
    host='${HOST}',
    port=${PORT},
    strict=False,
)

main(config)
"
