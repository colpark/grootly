#!/bin/bash
# GR00T Robot Client for Ball2 (Ball Transfer) Task
#
# Task: Bimanual ball transfer between arms
# Robot: Trossen AI Mobile
#
# Usage:
#   ./deployment/run_client_ball2.sh
#   ./deployment/run_client_ball2.sh --dry-run  # Test without robot
#
# Configuration:
#   - Server IP: 130.199.95.27
#   - Server Port: 5559
#   - Task: "Transfer the ball from one arm to the other"

set -e

# Task-specific configuration
TASK_NAME="ball2"
TASK_INSTRUCTION="${TASK_INSTRUCTION:-Transfer the ball from one arm to the other}"
SERVER_IP="${SERVER_IP:-130.199.95.27}"
SERVER_PORT="${SERVER_PORT:-5559}"

echo "=============================================="
echo "GR00T Client: Ball Transfer Task (Ball2)"
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
