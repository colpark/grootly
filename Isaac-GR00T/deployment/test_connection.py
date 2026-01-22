#!/usr/bin/env python3
"""
Connection Test Utility for GR00T Policy Server

This script tests the connection between the robot client and
the GR00T inference server. Use this to verify network connectivity
and server status before running the full robot control loop.

Usage:
    python deployment/test_connection.py
    python deployment/test_connection.py --server-ip 130.199.95.27 --server-port 5559

What this tests:
    1. TCP connection to server
    2. ZeroMQ message exchange (ping)
    3. Action inference latency
    4. Modality configuration retrieval
"""

import argparse
import sys
import time
from typing import Dict, Optional

import numpy as np

# Add project root to path
sys.path.insert(0, ".")

from gr00t.policy.server_client import PolicyClient


def create_dummy_observation() -> Dict:
    """Create a dummy observation matching Trossen AI Mobile format."""
    obs = {
        "video": {
            "cam_high": np.zeros((1, 1, 256, 256, 3), dtype=np.uint8),
            "cam_left_wrist": np.zeros((1, 1, 256, 256, 3), dtype=np.uint8),
            "cam_right_wrist": np.zeros((1, 1, 256, 256, 3), dtype=np.uint8),
        },
        "state": {
            "base_odom": np.zeros((1, 1, 3), dtype=np.float32),
            "base_vel": np.zeros((1, 1, 2), dtype=np.float32),
            "left_arm": np.zeros((1, 1, 7), dtype=np.float32),
            "right_arm": np.zeros((1, 1, 7), dtype=np.float32),
        },
        "language": {
            "task": [["Test task instruction"]],
        },
    }
    return obs


def test_ping(client: PolicyClient) -> bool:
    """Test basic ping to server."""
    print("\n[1/4] Testing ping...")
    try:
        start = time.time()
        result = client.ping()
        latency = (time.time() - start) * 1000

        if result:
            print(f"      ✓ Ping successful (latency: {latency:.1f}ms)")
            return True
        else:
            print("      ✗ Ping failed (no response)")
            return False
    except Exception as e:
        print(f"      ✗ Ping failed: {e}")
        return False


def test_modality_config(client: PolicyClient) -> bool:
    """Test modality configuration retrieval."""
    print("\n[2/4] Testing modality config retrieval...")
    try:
        start = time.time()
        config = client.get_modality_config()
        latency = (time.time() - start) * 1000

        if config:
            print(f"      ✓ Modality config retrieved (latency: {latency:.1f}ms)")
            print(f"      Available modalities: {list(config.keys())}")
            return True
        else:
            print("      ⚠ Empty modality config returned")
            return True  # Not a failure, just empty
    except Exception as e:
        print(f"      ✗ Modality config failed: {e}")
        return False


def test_inference(client: PolicyClient, num_trials: int = 5) -> bool:
    """Test inference with dummy data."""
    print(f"\n[3/4] Testing inference ({num_trials} trials)...")

    obs = create_dummy_observation()
    latencies = []

    for i in range(num_trials):
        try:
            start = time.time()
            action, info = client.get_action(obs)
            latency = (time.time() - start) * 1000
            latencies.append(latency)

            if i == 0:
                # Print action structure on first successful inference
                print(f"      Action keys: {list(action.keys())}")
                for key, value in action.items():
                    if isinstance(value, np.ndarray):
                        print(f"        {key}: shape={value.shape}")

        except Exception as e:
            print(f"      ✗ Inference {i+1} failed: {e}")
            return False

    if latencies:
        avg_latency = np.mean(latencies)
        min_latency = np.min(latencies)
        max_latency = np.max(latencies)
        print(f"      ✓ All inferences successful")
        print(f"      Latency stats: avg={avg_latency:.1f}ms, min={min_latency:.1f}ms, max={max_latency:.1f}ms")
        return True

    return False


def test_reset(client: PolicyClient) -> bool:
    """Test policy reset."""
    print("\n[4/4] Testing policy reset...")
    try:
        start = time.time()
        result = client.reset()
        latency = (time.time() - start) * 1000

        print(f"      ✓ Reset successful (latency: {latency:.1f}ms)")
        return True
    except Exception as e:
        print(f"      ⚠ Reset failed (may not be implemented): {e}")
        return True  # Not critical


def main():
    parser = argparse.ArgumentParser(
        description="Test connection to GR00T Policy Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--server-ip",
        type=str,
        default="130.199.95.27",
        help="IP address of the GR00T policy server",
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=5559,
        help="Port of the GR00T policy server",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=15000,
        help="Connection timeout in milliseconds",
    )
    parser.add_argument(
        "--inference-trials",
        type=int,
        default=5,
        help="Number of inference trials to run",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("GR00T Policy Server Connection Test")
    print("=" * 60)
    print(f"Target: tcp://{args.server_ip}:{args.server_port}")
    print(f"Timeout: {args.timeout}ms")
    print("=" * 60)

    # Create client
    print("\nConnecting to server...")
    try:
        client = PolicyClient(
            host=args.server_ip,
            port=args.server_port,
            timeout_ms=args.timeout,
        )
        print("Client created successfully")
    except Exception as e:
        print(f"Failed to create client: {e}")
        sys.exit(1)

    # Run tests
    results = {
        "ping": test_ping(client),
        "modality_config": test_modality_config(client),
        "inference": test_inference(client, args.inference_trials),
        "reset": test_reset(client),
    }

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name:20s}: {status}")
        if not passed:
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("\n✓ All tests passed! Server is ready for robot deployment.")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed. Please check server status and configuration.")
        sys.exit(1)


if __name__ == "__main__":
    main()
