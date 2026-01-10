#!/bin/bash
# Setup script for ALOHA simulation environment

echo "Setting up ALOHA simulation environment..."

# Install gym-aloha
pip install gym-aloha

# Or install via LeRobot (alternative)
# pip install 'lerobot[aloha]'

# Install additional dependencies
pip install mujoco>=3.1.1

echo "ALOHA simulation setup complete!"
echo ""
echo "Test with:"
echo "  python -m gr00t.eval.sim.ALOHA.aloha_env"
