#!/bin/bash
# GR00T Robot Client for LEGO Manipulation Task
#
# Task: LEGO block stacking/manipulation
# Robot: Trossen AI Mobile
#
# Usage:
#   ./deployment/run_client_lego.sh
#   ./deployment/run_client_lego.sh --dry-run  # Test without robot
#
# Configuration:
#   - Server IP: 130.199.95.27
#   - Server Port: 5559
#   - Task: "Stack the lego blocks"

set -e

# Task-specific configuration
TASK_NAME="lego"
TASK_INSTRUCTION="${TASK_INSTRUCTION:-Stack the lego blocks}"
SERVER_IP="${SERVER_IP:-130.199.95.27}"
SERVER_PORT="${SERVER_PORT:-5559}"

echo "=============================================="
echo "GR00T Client: LEGO Manipulation Task"
echo "=============================================="
echo "Task: ${TASK_NAME}"
echo "Instruction: ${TASK_INSTRUCTION}"
echo "Server: tcp://${SERVER_IP}:${SERVER_PORT}"
echo "=============================================="

# Parse arguments
EXTRA_ARGS=""
if [[ "$1" == "--dry-run" ]]; then
    EXTRA_ARGS="--dry-run --verbose"
    echo "Mode: DRY RUN (no robot commands)"
elif [[ "$1" == "--test" ]]; then
    echo "Testing connection only..."
    python deployment/test_connection.py \
        --server-ip "${SERVER_IP}" \
        --server-port "${SERVER_PORT}"
    exit $?
fi

echo ""
echo "Starting robot client..."
echo "Press Ctrl+C to stop"
echo ""

python deployment/trossen_client.py \
    --server-ip "${SERVER_IP}" \
    --server-port "${SERVER_PORT}" \
    --task "${TASK_INSTRUCTION}" \
    ${EXTRA_ARGS} \
    "$@"
