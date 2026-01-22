#!/bin/bash
# GR00T Inference Server for Plug Stacking Task
#
# Task: Stacking plug/peg objects
# Robot: Trossen AI Mobile
# Dataset: plug_stacking (inhouse)
#
# Usage:
#   ./deployment/run_server_plug_stacking.sh
#
# Server Configuration:
#   - Host: 0.0.0.0 (all interfaces)
#   - Port: 5559
#   - Checkpoint: ./outputs/plug_stacking/checkpoint-10000

set -e

# Task-specific configuration
TASK_NAME="plug_stacking"
CHECKPOINT="${CHECKPOINT:-./outputs/plug_stacking/checkpoint-10000}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-5559}"
DEVICE="${DEVICE:-cuda:0}"
MODALITY_CONFIG="${MODALITY_CONFIG:-./finetuning/trossen_modality_config.py}"

echo "=============================================="
echo "GR00T Server: Plug Stacking Task"
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
    echo "Available Plug Stacking checkpoints:"
    ls -d ./outputs/plug_stacking/checkpoint-* 2>/dev/null || echo "  No checkpoints found in ./outputs/plug_stacking/"
    echo ""
    echo "To train the Plug Stacking model:"
    echo "  1. python finetuning/convert_plug_stacking.py"
    echo "  2. ./finetuning/finetune_plug_stacking.sh"
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
from gr00t.data.embodiment_tags import EmbodimentTag

config = ServerConfig(
    model_path='${CHECKPOINT}',
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
    device='${DEVICE}',
    host='${HOST}',
    port=${PORT},
    strict=False,
)

main(config)
"
