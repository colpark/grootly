#!/bin/bash
# GR00T Inference Server for Trossen AI Mobile Robot
#
# Run this on the GPU server to serve model inference requests
# from the robot's client computer.
#
# Usage:
#   ./deployment/run_trossen_server.sh
#
# With custom settings:
#   CHECKPOINT=./outputs/lego/checkpoint-5000 PORT=5560 ./deployment/run_trossen_server.sh
#
# Server Configuration:
#   - Default host: 0.0.0.0 (all interfaces)
#   - Default port: 5559
#   - Default checkpoint: ./outputs/lego/checkpoint-10000
#   - Device: cuda:0

set -e

# Configuration (can be overridden via environment variables)
CHECKPOINT="${CHECKPOINT:-./outputs/lego/checkpoint-10000}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-5559}"
DEVICE="${DEVICE:-cuda:0}"
MODALITY_CONFIG="${MODALITY_CONFIG:-./finetuning/trossen_modality_config.py}"

echo "=============================================="
echo "GR00T Inference Server for Trossen AI Mobile"
echo "=============================================="
echo "Checkpoint: ${CHECKPOINT}"
echo "Host: ${HOST}"
echo "Port: ${PORT}"
echo "Device: ${DEVICE}"
echo "Modality Config: ${MODALITY_CONFIG}"
echo "=============================================="

# Validate checkpoint exists
if [[ ! -d "${CHECKPOINT}" ]]; then
    echo "ERROR: Checkpoint not found at ${CHECKPOINT}"
    echo ""
    echo "Available checkpoints:"
    ls -d ./outputs/lego/checkpoint-* 2>/dev/null || echo "  No checkpoints found in ./outputs/lego/"
    exit 1
fi

# Validate modality config exists
if [[ ! -f "${MODALITY_CONFIG}" ]]; then
    echo "ERROR: Modality config not found at ${MODALITY_CONFIG}"
    exit 1
fi

# Check for required model files
if [[ ! -f "${CHECKPOINT}/config.json" ]] && [[ ! -f "${CHECKPOINT}/model_config.json" ]]; then
    echo "WARNING: No config.json found in checkpoint directory"
    echo "         This may indicate an incomplete checkpoint"
fi

echo ""
echo "Starting server..."
echo "Robot client should connect to: tcp://${HOST}:${PORT}"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Import the modality config before starting the server
# This ensures the NEW_EMBODIMENT tag is registered
python -c "
import sys
sys.path.insert(0, '.')

# Register modality config
exec(open('${MODALITY_CONFIG}').read())

# Now start the server
from gr00t.eval.run_gr00t_server import main, ServerConfig

config = ServerConfig(
    model_path='${CHECKPOINT}',
    embodiment_tag='NEW_EMBODIMENT',
    device='${DEVICE}',
    host='${HOST}',
    port=${PORT},
    strict=False,  # Disable strict validation for flexibility
)

main(config)
"
