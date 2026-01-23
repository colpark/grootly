#!/bin/bash
# GR00T Robot Client for Ball2 (Ball Transfer) Task
#
# Task: Bimanual ball transfer between arms
# Robot: Trossen AI Mobile
#
# Usage:
#   ./deployment/run_client_ball2.sh
#   ./deployment/run_client_ball2.sh --dry-run    # Test without robot commands
#   ./deployment/run_client_ball2.sh --mock       # Test without robot hardware
#   ./deployment/run_client_ball2.sh --test-only  # Test server connection only
#
# Environment Variables:
#   SERVER_IP              - GPU server IP (default: 130.199.95.27)
#   SERVER_PORT            - GPU server port (default: 5560)
#   TASK_INSTRUCTION       - Task description
#   LEFT_ARM_IP            - Left arm IP (default: 192.168.1.5)
#   RIGHT_ARM_IP           - Right arm IP (default: 192.168.1.4)
#   CAM_HIGH_SERIAL        - High camera serial number
#   CAM_LEFT_WRIST_SERIAL  - Left wrist camera serial number
#   CAM_RIGHT_WRIST_SERIAL - Right wrist camera serial number

set -e

# Task-specific configuration
TASK_NAME="ball2"
TASK_INSTRUCTION="${TASK_INSTRUCTION:-Transfer the ball}"
SERVER_IP="${SERVER_IP:-130.199.95.27}"
SERVER_PORT="${SERVER_PORT:-5560}"

# Robot hardware configuration
LEFT_ARM_IP="${LEFT_ARM_IP:-192.168.1.5}"
RIGHT_ARM_IP="${RIGHT_ARM_IP:-192.168.1.4}"
CAM_HIGH_SERIAL="${CAM_HIGH_SERIAL:-130322271562}"
CAM_LEFT_WRIST_SERIAL="${CAM_LEFT_WRIST_SERIAL:-130322272107}"
CAM_RIGHT_WRIST_SERIAL="${CAM_RIGHT_WRIST_SERIAL:-130322273493}"

echo "=============================================="
echo "GR00T Client: Ball Transfer Task (Ball2)"
echo "=============================================="
echo "Task: ${TASK_NAME}"
echo "Instruction: ${TASK_INSTRUCTION}"
echo "Server: tcp://${SERVER_IP}:${SERVER_PORT}"
echo "=============================================="

# Parse arguments
EXTRA_ARGS=""
for arg in "$@"; do
    case $arg in
        --dry-run)
            EXTRA_ARGS="${EXTRA_ARGS} --dry-run --verbose"
            echo "Mode: DRY RUN (no robot commands)"
            ;;
        --mock)
            EXTRA_ARGS="${EXTRA_ARGS} --mock"
            echo "Mode: MOCK (no robot hardware)"
            ;;
        --test-only)
            EXTRA_ARGS="${EXTRA_ARGS} --test-only"
            echo "Mode: TEST ONLY (server connection)"
            ;;
        --verbose|-v)
            EXTRA_ARGS="${EXTRA_ARGS} --verbose"
            ;;
        *)
            EXTRA_ARGS="${EXTRA_ARGS} ${arg}"
            ;;
    esac
done

echo ""
echo "Starting robot client..."
echo "Press Ctrl+C to stop"
echo ""

python deployment/trossen_client.py \
    --server-ip "${SERVER_IP}" \
    --server-port "${SERVER_PORT}" \
    --task "${TASK_INSTRUCTION}" \
    --left-arm-ip "${LEFT_ARM_IP}" \
    --right-arm-ip "${RIGHT_ARM_IP}" \
    --cam-high-serial "${CAM_HIGH_SERIAL}" \
    --cam-left-wrist-serial "${CAM_LEFT_WRIST_SERIAL}" \
    --cam-right-wrist-serial "${CAM_RIGHT_WRIST_SERIAL}" \
    ${EXTRA_ARGS}
