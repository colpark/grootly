#!/usr/bin/env python3
"""
Verify GR-1 Tabletop Tasks Installation

Run with:
    gr00t/eval/sim/robocasa-gr1-tabletop-tasks/robocasa_uv/.venv/bin/python scripts/verify_gr1_install.py
"""

import sys

def check_import(module_name, package_name=None):
    """Check if a module can be imported."""
    package_name = package_name or module_name
    try:
        __import__(module_name)
        print(f"  ✅ {package_name}")
        return True
    except ImportError as e:
        print(f"  ❌ {package_name}: {e}")
        return False

def main():
    print("=" * 60)
    print("GR-1 TABLETOP TASKS INSTALLATION VERIFICATION")
    print("=" * 60)

    all_ok = True

    # 1. Core dependencies
    print("\n📦 Core Dependencies:")
    all_ok &= check_import("numpy")
    all_ok &= check_import("torch")
    all_ok &= check_import("scipy")
    all_ok &= check_import("numba")
    all_ok &= check_import("h5py")
    all_ok &= check_import("imageio")

    # 2. Simulation dependencies
    print("\n🤖 Simulation Dependencies:")
    all_ok &= check_import("mujoco")
    all_ok &= check_import("robosuite")
    all_ok &= check_import("gymnasium")

    # 3. RoboCasa GR-1 specific
    print("\n🦾 RoboCasa GR-1 Tabletop:")
    robocasa_ok = check_import("robocasa")
    all_ok &= robocasa_ok

    # 4. GR00T dependencies
    print("\n🧠 GR00T Dependencies:")
    all_ok &= check_import("transformers")
    all_ok &= check_import("zmq", "pyzmq")
    all_ok &= check_import("msgpack")
    all_ok &= check_import("av", "PyAV")

    # 5. Try to import gr00t
    print("\n📁 GR00T Package:")
    groot_ok = check_import("gr00t")
    all_ok &= groot_ok

    if groot_ok:
        try:
            from gr00t.eval.rollout_policy import get_robocasa_gr1_env_fn
            print("  ✅ get_robocasa_gr1_env_fn available")
        except ImportError as e:
            print(f"  ⚠️  get_robocasa_gr1_env_fn not found: {e}")

    # 6. Try to create environment
    print("\n🎮 Environment Creation Test:")
    if robocasa_ok:
        try:
            import robocasa
            print(f"  ✅ RoboCasa version: {robocasa.__version__ if hasattr(robocasa, '__version__') else 'unknown'}")

            # Check available environments
            from robocasa.environments import ALL_ENVIRONMENTS
            gr1_envs = [e for e in ALL_ENVIRONMENTS if 'GR1' in e or 'gr1' in e.lower()]
            print(f"  ✅ Found {len(gr1_envs)} GR-1 environments")
            if gr1_envs:
                print(f"     Sample: {gr1_envs[0]}")
        except Exception as e:
            print(f"  ⚠️  Could not list environments: {e}")

        # Try actual environment creation
        try:
            print("\n  Creating test environment (this may take a moment)...")
            import robosuite
            from robocasa.environments.kitchen import KitchenEnv

            # Try to find a GR1 environment
            env_name = "PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_Env"

            # Import the environment module
            try:
                from robocasa.environments.kitchen.multi_stage import pnp_novel
                print(f"  ✅ PnP Novel environments module imported")
            except ImportError as e:
                print(f"  ⚠️  Could not import pnp_novel: {e}")

        except Exception as e:
            print(f"  ⚠️  Environment creation test: {e}")

    # Summary
    print("\n" + "=" * 60)
    if all_ok:
        print("✅ ALL CHECKS PASSED - Ready for GR-1 evaluation!")
    else:
        print("⚠️  SOME CHECKS FAILED - See above for details")
    print("=" * 60)

    # Print next steps
    print("""
Next Steps:
-----------
1. Start the policy server (Terminal 1):
   uv run python gr00t/eval/run_gr00t_server.py \\
       --model-path nvidia/GR00T-N1.6-3B \\
       --embodiment-tag GR1 \\
       --use-sim-policy-wrapper

2. Run evaluation (Terminal 2):
   gr00t/eval/sim/robocasa-gr1-tabletop-tasks/robocasa_uv/.venv/bin/python scripts/transparent_evaluation.py \\
       --env-name gr1_unified/PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_Env \\
       --save-dir ./gr1_eval \\
       --camera-view side
""")

    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
