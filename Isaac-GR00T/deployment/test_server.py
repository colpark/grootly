#!/usr/bin/env python3
"""
Server Environment and Model Test Script

This script validates the server environment and tests model loading
before actual deployment. Run this on the GPU server to ensure
everything is properly configured.

Tests performed:
    1. Python environment and dependencies
    2. CUDA availability and GPU memory
    3. Modality config registration
    4. Checkpoint loading
    5. Dummy inference (optional)
    6. Server binding test (optional)

Usage:
    python deployment/test_server.py --task lego
    python deployment/test_server.py --task ball2 --full-test
    python deployment/test_server.py --task plug_stacking --test-server

Options:
    --task          Task name: lego, ball2, plug_stacking
    --checkpoint    Custom checkpoint path (overrides task default)
    --device        GPU device (default: cuda:0)
    --full-test     Run dummy inference test
    --test-server   Test server binding (requires port 5559 to be free)
    --port          Server port for binding test (default: 5559)
"""

import argparse
import sys
import os
import time
from typing import Dict, Optional, Tuple

# Add project root to path
sys.path.insert(0, ".")


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_result(test_name: str, passed: bool, message: str = ""):
    """Print test result."""
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {test_name:40s} [{status}]")
    if message:
        print(f"    → {message}")


def test_python_environment() -> Tuple[bool, Dict]:
    """Test Python version and basic environment."""
    print_header("1. Python Environment")

    results = {}
    all_passed = True

    # Python version
    version = sys.version_info
    version_ok = version.major == 3 and version.minor >= 10
    results["python_version"] = f"{version.major}.{version.minor}.{version.micro}"
    print_result("Python version >= 3.10", version_ok, results["python_version"])
    all_passed &= version_ok

    # Working directory
    cwd = os.getcwd()
    results["working_dir"] = cwd
    print_result("Working directory", True, cwd)

    return all_passed, results


def test_dependencies() -> Tuple[bool, Dict]:
    """Test required Python packages."""
    print_header("2. Dependencies")

    results = {}
    all_passed = True

    packages = [
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("numpy", "numpy"),
        ("pyzmq", "zmq"),
        ("msgpack", "msgpack"),
        ("tyro", "tyro"),
    ]

    for name, import_name in packages:
        try:
            module = __import__(import_name)
            version = getattr(module, "__version__", "unknown")
            results[name] = version
            print_result(f"{name}", True, f"v{version}")
        except ImportError as e:
            results[name] = None
            print_result(f"{name}", False, str(e))
            all_passed = False

    return all_passed, results


def test_cuda() -> Tuple[bool, Dict]:
    """Test CUDA availability and GPU."""
    print_header("3. CUDA / GPU")

    results = {}
    all_passed = True

    try:
        import torch

        # CUDA available
        cuda_available = torch.cuda.is_available()
        results["cuda_available"] = cuda_available
        print_result("CUDA available", cuda_available)
        all_passed &= cuda_available

        if cuda_available:
            # CUDA version
            cuda_version = torch.version.cuda
            results["cuda_version"] = cuda_version
            print_result("CUDA version", True, cuda_version)

            # GPU count
            gpu_count = torch.cuda.device_count()
            results["gpu_count"] = gpu_count
            print_result("GPU count", gpu_count > 0, str(gpu_count))

            # GPU info
            for i in range(gpu_count):
                name = torch.cuda.get_device_name(i)
                mem_total = torch.cuda.get_device_properties(i).total_memory / 1e9
                mem_free = (torch.cuda.get_device_properties(i).total_memory -
                           torch.cuda.memory_allocated(i)) / 1e9
                results[f"gpu_{i}"] = {"name": name, "memory_gb": mem_total}
                print_result(f"GPU {i}: {name}", True, f"{mem_total:.1f}GB total")
        else:
            print_result("GPU info", False, "CUDA not available")
            all_passed = False

    except Exception as e:
        results["error"] = str(e)
        print_result("CUDA test", False, str(e))
        all_passed = False

    return all_passed, results


def test_modality_config() -> Tuple[bool, Dict]:
    """Test modality config registration."""
    print_header("4. Modality Configuration")

    results = {}
    all_passed = True

    config_path = "./finetuning/trossen_modality_config.py"

    # Check file exists
    config_exists = os.path.exists(config_path)
    results["config_exists"] = config_exists
    print_result("Config file exists", config_exists, config_path)
    all_passed &= config_exists

    if config_exists:
        try:
            # Execute the config to register modality
            exec(open(config_path).read())
            results["config_loaded"] = True
            print_result("Config loaded successfully", True)

            # Verify registration
            from gr00t.data.embodiment_tags import EmbodimentTag
            from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS

            tag = EmbodimentTag.NEW_EMBODIMENT.value
            registered = tag in MODALITY_CONFIGS
            results["registered"] = registered
            print_result("NEW_EMBODIMENT registered", registered)
            all_passed &= registered

            if registered:
                config = MODALITY_CONFIGS[tag]
                modalities = list(config.keys())
                results["modalities"] = modalities
                print_result("Modalities", True, ", ".join(modalities))

        except Exception as e:
            results["error"] = str(e)
            print_result("Config loading", False, str(e))
            all_passed = False

    return all_passed, results


def test_checkpoint(task: str, checkpoint_path: Optional[str] = None) -> Tuple[bool, Dict]:
    """Test checkpoint existence and structure."""
    print_header("5. Checkpoint Validation")

    results = {}
    all_passed = True

    # Determine checkpoint path
    if checkpoint_path is None:
        task_checkpoints = {
            "lego": "./outputs/lego/checkpoint-10000",
            "ball2": "./outputs/ball2_groot/checkpoint-10000",
            "plug_stacking": "./outputs/plug_stacking/checkpoint-10000",
        }
        checkpoint_path = task_checkpoints.get(task, task_checkpoints["lego"])

    results["checkpoint_path"] = checkpoint_path
    print(f"  Checkpoint: {checkpoint_path}")

    # Check directory exists
    dir_exists = os.path.isdir(checkpoint_path)
    results["dir_exists"] = dir_exists
    print_result("Checkpoint directory exists", dir_exists)
    all_passed &= dir_exists

    if dir_exists:
        # Check for expected files
        expected_files = [
            "config.json",
            "model.safetensors",
        ]

        for filename in expected_files:
            filepath = os.path.join(checkpoint_path, filename)
            exists = os.path.exists(filepath)
            results[filename] = exists
            if exists:
                size_mb = os.path.getsize(filepath) / 1e6
                print_result(f"  {filename}", True, f"{size_mb:.1f}MB")
            else:
                # Try alternative names
                alt_found = False
                if filename == "model.safetensors":
                    for alt in ["pytorch_model.bin", "model.bin"]:
                        alt_path = os.path.join(checkpoint_path, alt)
                        if os.path.exists(alt_path):
                            size_mb = os.path.getsize(alt_path) / 1e6
                            print_result(f"  {alt} (alt)", True, f"{size_mb:.1f}MB")
                            alt_found = True
                            break
                if not alt_found:
                    print_result(f"  {filename}", False, "Not found")

        # List all files
        all_files = os.listdir(checkpoint_path)
        results["all_files"] = all_files
        print(f"\n  All files in checkpoint: {len(all_files)}")
        for f in sorted(all_files)[:10]:  # Show first 10
            print(f"    - {f}")
        if len(all_files) > 10:
            print(f"    ... and {len(all_files) - 10} more")

    return all_passed, results


def test_model_loading(task: str, checkpoint_path: Optional[str], device: str) -> Tuple[bool, Dict]:
    """Test actual model loading."""
    print_header("6. Model Loading")

    results = {}
    all_passed = True

    # Determine checkpoint path
    if checkpoint_path is None:
        task_checkpoints = {
            "lego": "./outputs/lego/checkpoint-10000",
            "ball2": "./outputs/ball2_groot/checkpoint-10000",
            "plug_stacking": "./outputs/plug_stacking/checkpoint-10000",
        }
        checkpoint_path = task_checkpoints.get(task, task_checkpoints["lego"])

    print(f"  Loading model from: {checkpoint_path}")
    print(f"  Device: {device}")

    try:
        import torch

        # Check GPU memory before loading
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
            mem_before = torch.cuda.memory_allocated() / 1e9
            print(f"  GPU memory before: {mem_before:.2f}GB")

        # Load the model
        start_time = time.time()

        from gr00t.policy.gr00t_policy import Gr00tPolicy
        from gr00t.data.embodiment_tags import EmbodimentTag

        policy = Gr00tPolicy(
            embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
            model_path=checkpoint_path,
            device=device,
            strict=False,
        )

        load_time = time.time() - start_time
        results["load_time"] = load_time
        print_result("Model loaded", True, f"{load_time:.1f}s")

        # Check GPU memory after loading
        if device.startswith("cuda"):
            mem_after = torch.cuda.memory_allocated() / 1e9
            mem_used = mem_after - mem_before
            results["gpu_memory_used"] = mem_used
            print_result("GPU memory used", True, f"{mem_used:.2f}GB")

        # Get modality config
        try:
            modality_config = policy.get_modality_config()
            results["modality_config"] = list(modality_config.keys())
            print_result("Modality config retrieved", True)
        except Exception as e:
            print_result("Modality config", False, str(e))

        # Cleanup
        del policy
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

        results["success"] = True

    except Exception as e:
        results["error"] = str(e)
        print_result("Model loading", False, str(e))
        all_passed = False

        # Print full traceback for debugging
        import traceback
        print("\n  Full traceback:")
        traceback.print_exc()

    return all_passed, results


def test_dummy_inference(task: str, checkpoint_path: Optional[str], device: str) -> Tuple[bool, Dict]:
    """Test dummy inference with fake data."""
    print_header("7. Dummy Inference Test")

    results = {}
    all_passed = True

    # Determine checkpoint path
    if checkpoint_path is None:
        task_checkpoints = {
            "lego": "./outputs/lego/checkpoint-10000",
            "ball2": "./outputs/ball2_groot/checkpoint-10000",
            "plug_stacking": "./outputs/plug_stacking/checkpoint-10000",
        }
        checkpoint_path = task_checkpoints.get(task, task_checkpoints["lego"])

    print(f"  Running inference test...")

    try:
        import torch
        import numpy as np

        from gr00t.policy.gr00t_policy import Gr00tPolicy
        from gr00t.data.embodiment_tags import EmbodimentTag

        # Load model
        policy = Gr00tPolicy(
            embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
            model_path=checkpoint_path,
            device=device,
            strict=False,
        )

        # Create dummy observation
        dummy_obs = {
            "video": {
                "cam_high": np.random.randint(0, 255, (1, 1, 256, 256, 3), dtype=np.uint8),
                "cam_left_wrist": np.random.randint(0, 255, (1, 1, 256, 256, 3), dtype=np.uint8),
                "cam_right_wrist": np.random.randint(0, 255, (1, 1, 256, 256, 3), dtype=np.uint8),
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

        # Run inference
        start_time = time.time()
        action, info = policy.get_action(dummy_obs)
        inference_time = (time.time() - start_time) * 1000

        results["inference_time_ms"] = inference_time
        print_result("Inference completed", True, f"{inference_time:.1f}ms")

        # Check action shape
        print("\n  Action output:")
        for key, value in action.items():
            if isinstance(value, np.ndarray):
                print(f"    {key}: shape={value.shape}, dtype={value.dtype}")
                results[f"action_{key}_shape"] = value.shape

        # Run multiple inferences for timing
        print("\n  Running 5 more inferences for timing...")
        times = []
        for i in range(5):
            start = time.time()
            policy.get_action(dummy_obs)
            times.append((time.time() - start) * 1000)

        avg_time = np.mean(times)
        std_time = np.std(times)
        results["avg_inference_ms"] = avg_time
        results["std_inference_ms"] = std_time
        print_result("Average inference time", True, f"{avg_time:.1f}ms ± {std_time:.1f}ms")

        # Cleanup
        del policy
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

        results["success"] = True

    except Exception as e:
        results["error"] = str(e)
        print_result("Inference test", False, str(e))
        all_passed = False

        import traceback
        print("\n  Full traceback:")
        traceback.print_exc()

    return all_passed, results


def test_server_binding(port: int) -> Tuple[bool, Dict]:
    """Test server socket binding."""
    print_header("8. Server Binding Test")

    results = {}
    all_passed = True

    print(f"  Testing port {port}...")

    try:
        import zmq

        context = zmq.Context()
        socket = context.socket(zmq.REP)

        # Try to bind
        address = f"tcp://0.0.0.0:{port}"
        socket.bind(address)

        results["bind_address"] = address
        print_result("Socket binding", True, address)

        # Cleanup
        socket.close()
        context.term()

        results["success"] = True

    except zmq.error.ZMQError as e:
        if "Address already in use" in str(e):
            results["error"] = "Port already in use"
            print_result("Socket binding", False, f"Port {port} already in use")
            print("    → Another process may be using this port")
            print(f"    → Try: lsof -i :{port}")
        else:
            results["error"] = str(e)
            print_result("Socket binding", False, str(e))
        all_passed = False

    except Exception as e:
        results["error"] = str(e)
        print_result("Socket binding", False, str(e))
        all_passed = False

    return all_passed, results


def main():
    parser = argparse.ArgumentParser(
        description="Test server environment and model before deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["lego", "ball2", "plug_stacking"],
        default="lego",
        help="Task name (determines default checkpoint path)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Custom checkpoint path (overrides task default)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="GPU device (default: cuda:0)",
    )
    parser.add_argument(
        "--full-test",
        action="store_true",
        help="Run full test including dummy inference",
    )
    parser.add_argument(
        "--test-server",
        action="store_true",
        help="Test server socket binding",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5559,
        help="Port for server binding test (default: 5559)",
    )
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="Skip model loading test (faster)",
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  GR00T SERVER ENVIRONMENT TEST")
    print("=" * 60)
    print(f"  Task: {args.task}")
    print(f"  Device: {args.device}")
    print(f"  Full test: {args.full_test}")
    print("=" * 60)

    all_results = {}
    all_passed = True

    # Run tests
    passed, results = test_python_environment()
    all_results["python"] = results
    all_passed &= passed

    passed, results = test_dependencies()
    all_results["dependencies"] = results
    all_passed &= passed

    passed, results = test_cuda()
    all_results["cuda"] = results
    all_passed &= passed

    passed, results = test_modality_config()
    all_results["modality_config"] = results
    all_passed &= passed

    passed, results = test_checkpoint(args.task, args.checkpoint)
    all_results["checkpoint"] = results
    all_passed &= passed

    if not args.skip_model:
        passed, results = test_model_loading(args.task, args.checkpoint, args.device)
        all_results["model_loading"] = results
        all_passed &= passed

        if args.full_test and passed:
            passed, results = test_dummy_inference(args.task, args.checkpoint, args.device)
            all_results["inference"] = results
            all_passed &= passed

    if args.test_server:
        passed, results = test_server_binding(args.port)
        all_results["server_binding"] = results
        all_passed &= passed

    # Summary
    print_header("TEST SUMMARY")

    test_names = {
        "python": "Python Environment",
        "dependencies": "Dependencies",
        "cuda": "CUDA / GPU",
        "modality_config": "Modality Config",
        "checkpoint": "Checkpoint",
        "model_loading": "Model Loading",
        "inference": "Inference",
        "server_binding": "Server Binding",
    }

    for key, name in test_names.items():
        if key in all_results:
            success = all_results[key].get("success", True)
            error = all_results[key].get("error")
            if error:
                print_result(name, False, error[:50])
            else:
                print_result(name, success)

    print("\n" + "=" * 60)
    if all_passed:
        print("  ✓ ALL TESTS PASSED - Server is ready for deployment!")
    else:
        print("  ✗ SOME TESTS FAILED - Please fix issues before deployment")
    print("=" * 60 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
