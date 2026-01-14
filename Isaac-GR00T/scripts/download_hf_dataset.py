#!/usr/bin/env python3
"""
Download datasets from Hugging Face to local storage.

Usage:
    python scripts/download_hf_dataset.py TrossenRoboticsCommunity/trossen_ai_mobile_lego
    python scripts/download_hf_dataset.py TrossenRoboticsCommunity/trossen_ai_mobile_lego --name trossen_lego
    python scripts/download_hf_dataset.py TrossenRoboticsCommunity/trossen_ai_mobile_lego --output ./data/lego
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Download Hugging Face dataset")
    parser.add_argument(
        "repo_id",
        type=str,
        help="Hugging Face dataset repo ID (e.g., TrossenRoboticsCommunity/trossen_ai_mobile_lego)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Local folder name (default: last part of repo_id)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./inhouse/lerobot_dataset/lerobot/recorded_data",
        help="Output base directory",
    )
    args = parser.parse_args()

    # Determine local folder name
    if args.name:
        folder_name = args.name
    else:
        folder_name = args.repo_id.split("/")[-1]

    output_path = Path(args.output) / folder_name

    print(f"=" * 60)
    print(f"Downloading Hugging Face Dataset")
    print(f"=" * 60)
    print(f"Repo ID: {args.repo_id}")
    print(f"Output:  {output_path}")
    print(f"=" * 60)

    # Check if huggingface_hub is installed
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("\nERROR: huggingface_hub not installed.")
        print("Install with: pip install huggingface_hub")
        return 1

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Download
    print("\nStarting download...")
    try:
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            local_dir=str(output_path),
        )
        print(f"\nDownload complete!")
        print(f"Dataset saved to: {output_path}")

        # Show structure
        print(f"\nDataset structure:")
        for item in sorted(output_path.iterdir()):
            if item.is_dir():
                print(f"  {item.name}/")
            else:
                print(f"  {item.name}")

        # Check for meta/info.json
        info_file = output_path / "meta" / "info.json"
        if info_file.exists():
            import json
            with open(info_file) as f:
                info = json.load(f)
            print(f"\nDataset info:")
            print(f"  Episodes: {info.get('total_episodes', 'N/A')}")
            print(f"  Frames:   {info.get('total_frames', 'N/A')}")
            print(f"  FPS:      {info.get('fps', 'N/A')}")
            print(f"  Robot:    {info.get('robot_type', 'N/A')}")

    except Exception as e:
        print(f"\nERROR: Download failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
