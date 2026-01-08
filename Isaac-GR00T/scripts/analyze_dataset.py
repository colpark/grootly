#!/usr/bin/env python3
"""
GR00T Dataset Analyzer

Analyzes LeRobot-format datasets used for GR00T training and evaluation.
Provides visualization of:
1. Dataset structure and metadata
2. Sample video frames from episodes
3. State/action trajectories
4. Modality configurations
5. Dataset statistics

Usage:
    python scripts/analyze_dataset.py --dataset-path demo_data/gr1.PickNPlace --save-dir ./dataset_analysis

Works with any LeRobot v2 format dataset.
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

# Optional visualization dependencies
try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("WARNING: matplotlib not installed. Visualizations will be skipped.")

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# Try different video backends
VIDEO_BACKEND = None
try:
    import torchcodec
    VIDEO_BACKEND = "torchcodec"
except ImportError:
    try:
        import decord
        VIDEO_BACKEND = "decord"
    except ImportError:
        try:
            import imageio
            VIDEO_BACKEND = "imageio"
        except ImportError:
            print("WARNING: No video backend available (torchcodec, decord, or imageio).")


class DatasetAnalyzer:
    """Analyze LeRobot-format datasets for GR00T training."""

    def __init__(self, dataset_path: str, save_dir: str):
        self.dataset_path = Path(dataset_path)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")

        # Load metadata
        self.meta_dir = self.dataset_path / "meta"
        self.info = self._load_json("info.json")
        self.modality = self._load_json("modality.json")
        self.stats = self._load_json("stats.json")
        self.episodes = self._load_jsonl("episodes.jsonl")
        self.tasks = self._load_jsonl("tasks.jsonl")

        print(f"\n{'='*60}")
        print("DATASET ANALYZER INITIALIZED")
        print(f"{'='*60}")
        print(f"Dataset: {self.dataset_path}")
        print(f"Output: {self.save_dir}")

    def _load_json(self, filename: str) -> Dict:
        """Load JSON file from meta directory."""
        path = self.meta_dir / filename
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return {}

    def _load_jsonl(self, filename: str) -> List[Dict]:
        """Load JSONL file from meta directory."""
        path = self.meta_dir / filename
        if path.exists():
            with open(path) as f:
                return [json.loads(line) for line in f if line.strip()]
        return []

    def analyze_structure(self) -> Dict:
        """Analyze overall dataset structure."""
        print(f"\n{'='*60}")
        print("1. DATASET STRUCTURE")
        print(f"{'='*60}")

        structure = {
            "path": str(self.dataset_path),
            "format_version": self.info.get("codebase_version", "unknown"),
            "robot_type": self.info.get("robot_type", "unknown"),
            "total_episodes": self.info.get("total_episodes", len(self.episodes)),
            "total_frames": self.info.get("total_frames", 0),
            "total_tasks": self.info.get("total_tasks", len(self.tasks)),
            "fps": self.info.get("fps", "unknown"),
            "chunks_size": self.info.get("chunks_size", "unknown"),
        }

        print(f"""
Dataset Structure:
├── Format Version: {structure['format_version']}
├── Robot Type: {structure['robot_type']}
├── Total Episodes: {structure['total_episodes']}
├── Total Frames: {structure['total_frames']}
├── Total Tasks: {structure['total_tasks']}
├── FPS: {structure['fps']}
└── Chunk Size: {structure['chunks_size']}
""")

        # Directory structure
        print("Directory Layout:")
        print(f"├── meta/")
        for f in sorted(self.meta_dir.glob("*")):
            print(f"│   ├── {f.name}")

        data_dir = self.dataset_path / "data"
        if data_dir.exists():
            print(f"├── data/")
            parquet_files = list(data_dir.rglob("*.parquet"))
            print(f"│   └── {len(parquet_files)} parquet files")

        video_dir = self.dataset_path / "videos"
        if video_dir.exists():
            print(f"└── videos/")
            video_files = list(video_dir.rglob("*.mp4"))
            print(f"    └── {len(video_files)} video files")

        return structure

    def analyze_modality(self) -> Dict:
        """Analyze modality configuration."""
        print(f"\n{'='*60}")
        print("2. MODALITY CONFIGURATION")
        print(f"{'='*60}")

        print("\nThis defines how data is organized and mapped to model inputs:")

        # State modality
        if "state" in self.modality:
            print("\n🤖 STATE (Robot Proprioception):")
            print("-" * 40)
            total_state_dim = 0
            for name, config in self.modality["state"].items():
                start, end = config["start"], config["end"]
                dim = end - start
                total_state_dim += dim
                print(f"  {name:20s}: indices [{start:2d}:{end:2d}] ({dim} dims)")
            print(f"  {'TOTAL':20s}: {total_state_dim} dimensions")

        # Action modality
        if "action" in self.modality:
            print("\n🎮 ACTION (Robot Commands):")
            print("-" * 40)
            total_action_dim = 0
            for name, config in self.modality["action"].items():
                start, end = config["start"], config["end"]
                dim = end - start
                total_action_dim += dim
                print(f"  {name:20s}: indices [{start:2d}:{end:2d}] ({dim} dims)")
            print(f"  {'TOTAL':20s}: {total_action_dim} dimensions")

        # Video modality
        if "video" in self.modality:
            print("\n📷 VIDEO (Camera Inputs):")
            print("-" * 40)
            for name, config in self.modality["video"].items():
                original = config.get("original_key", "N/A")
                print(f"  {name}: maps to '{original}'")

        # Annotation/Language modality
        if "annotation" in self.modality:
            print("\n💬 ANNOTATION (Language/Labels):")
            print("-" * 40)
            for name, config in self.modality["annotation"].items():
                original = config.get("original_key", "(direct)")
                print(f"  {name}: {original}")

        # Save modality config
        with open(self.save_dir / "modality_config.json", "w") as f:
            json.dump(self.modality, f, indent=2)

        return self.modality

    def analyze_features(self) -> Dict:
        """Analyze feature specifications from info.json."""
        print(f"\n{'='*60}")
        print("3. FEATURE SPECIFICATIONS")
        print(f"{'='*60}")

        features = self.info.get("features", {})

        print("\nData types and shapes for each feature:")
        print("-" * 60)

        for name, spec in features.items():
            dtype = spec.get("dtype", "unknown")
            shape = spec.get("shape", [])
            names = spec.get("names", [])

            if dtype == "video":
                video_info = spec.get("video_info", {})
                fps = video_info.get("video.fps", "?")
                codec = video_info.get("video.codec", "?")
                print(f"\n📷 {name}")
                print(f"   Type: video ({codec})")
                print(f"   Shape: {shape} (H×W×C)")
                print(f"   FPS: {fps}")
            else:
                print(f"\n📊 {name}")
                print(f"   Type: {dtype}")
                print(f"   Shape: {shape}")
                if names and len(names) <= 10:
                    print(f"   Names: {names[:5]}{'...' if len(names) > 5 else ''}")

        return features

    def analyze_episodes(self) -> Dict:
        """Analyze episode metadata."""
        print(f"\n{'='*60}")
        print("4. EPISODE ANALYSIS")
        print(f"{'='*60}")

        if not self.episodes:
            print("No episode metadata found.")
            return {}

        lengths = [ep.get("length", 0) for ep in self.episodes]
        tasks_per_ep = [len(ep.get("tasks", [])) for ep in self.episodes]

        analysis = {
            "num_episodes": len(self.episodes),
            "total_frames": sum(lengths),
            "avg_length": np.mean(lengths),
            "min_length": min(lengths),
            "max_length": max(lengths),
            "std_length": np.std(lengths),
        }

        print(f"""
Episode Statistics:
├── Number of Episodes: {analysis['num_episodes']}
├── Total Frames: {analysis['total_frames']}
├── Average Length: {analysis['avg_length']:.1f} frames
├── Min Length: {analysis['min_length']} frames
├── Max Length: {analysis['max_length']} frames
└── Std Dev: {analysis['std_length']:.1f} frames
""")

        print("\nPer-Episode Details:")
        print("-" * 60)
        for ep in self.episodes[:10]:  # Show first 10
            idx = ep.get("episode_index", "?")
            length = ep.get("length", 0)
            tasks = ep.get("tasks", [])
            task_desc = tasks[0] if tasks else "N/A"
            if len(task_desc) > 50:
                task_desc = task_desc[:47] + "..."
            print(f"  Episode {idx:3d}: {length:4d} frames | {task_desc}")

        if len(self.episodes) > 10:
            print(f"  ... and {len(self.episodes) - 10} more episodes")

        # Plot episode lengths
        if HAS_MATPLOTLIB and lengths:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(range(len(lengths)), lengths, color='steelblue', alpha=0.7)
            ax.set_xlabel("Episode Index")
            ax.set_ylabel("Length (frames)")
            ax.set_title("Episode Lengths")
            ax.axhline(y=np.mean(lengths), color='r', linestyle='--', label=f'Mean: {np.mean(lengths):.1f}')
            ax.legend()
            plt.tight_layout()
            plt.savefig(self.save_dir / "episode_lengths.png", dpi=150)
            plt.close()
            print(f"\n  Saved: {self.save_dir / 'episode_lengths.png'}")

        return analysis

    def analyze_tasks(self) -> Dict:
        """Analyze task descriptions."""
        print(f"\n{'='*60}")
        print("5. TASK DESCRIPTIONS")
        print(f"{'='*60}")

        if not self.tasks:
            print("No task metadata found.")
            return {}

        print("\nLanguage instructions used for training:")
        print("-" * 60)

        task_map = {}
        for task in self.tasks:
            idx = task.get("task_index", "?")
            desc = task.get("task", "N/A")
            task_map[idx] = desc
            if desc != "valid":  # Skip validity markers
                print(f"  [{idx}] {desc}")

        print(f"\n  Total unique tasks: {len([t for t in self.tasks if t.get('task') != 'valid'])}")

        return task_map

    def analyze_statistics(self) -> Dict:
        """Analyze dataset statistics for normalization."""
        print(f"\n{'='*60}")
        print("6. NORMALIZATION STATISTICS")
        print(f"{'='*60}")

        if not self.stats:
            print("No statistics found.")
            return {}

        print("\nThese statistics are used to normalize inputs/outputs:")
        print("-" * 60)

        for key, stats in self.stats.items():
            if isinstance(stats, dict) and "mean" in stats:
                mean = stats.get("mean", [])
                std = stats.get("std", [])
                if isinstance(mean, list):
                    print(f"\n📊 {key}:")
                    print(f"   Dimensions: {len(mean)}")
                    print(f"   Mean range: [{min(mean):.4f}, {max(mean):.4f}]")
                    print(f"   Std range:  [{min(std):.4f}, {max(std):.4f}]")

        # Save full stats
        with open(self.save_dir / "statistics.json", "w") as f:
            json.dump(self.stats, f, indent=2)

        return self.stats

    def load_episode_data(self, episode_idx: int = 0) -> Optional[pd.DataFrame]:
        """Load parquet data for a specific episode."""
        data_path_pattern = self.info.get("data_path", "")
        if not data_path_pattern:
            return None

        chunk_idx = episode_idx // self.info.get("chunks_size", 1000)
        parquet_path = self.dataset_path / data_path_pattern.format(
            episode_chunk=chunk_idx,
            episode_index=episode_idx
        )

        if parquet_path.exists():
            return pd.read_parquet(parquet_path)
        return None

    def visualize_episode_data(self, episode_idx: int = 0):
        """Visualize state/action trajectories for an episode."""
        print(f"\n{'='*60}")
        print(f"7. EPISODE DATA VISUALIZATION (Episode {episode_idx})")
        print(f"{'='*60}")

        df = self.load_episode_data(episode_idx)
        if df is None:
            print(f"Could not load episode {episode_idx}")
            return

        print(f"\nDataFrame columns: {list(df.columns)}")
        print(f"Number of timesteps: {len(df)}")

        # Get state and action columns
        state_cols = [c for c in df.columns if 'state' in c.lower()]
        action_cols = [c for c in df.columns if c == 'action' or 'action' in c.lower()]

        if not HAS_MATPLOTLIB:
            print("Matplotlib not available for plotting")
            return

        # Plot state trajectory
        if state_cols and 'observation.state' in df.columns:
            states = np.stack(df['observation.state'].values)
            print(f"\nState shape: {states.shape}")

            fig, axes = plt.subplots(2, 1, figsize=(12, 8))

            # Plot first 7 dimensions (typically arm joints)
            ax = axes[0]
            for i in range(min(7, states.shape[1])):
                ax.plot(states[:, i], label=f'Joint {i}', alpha=0.7)
            ax.set_xlabel("Timestep")
            ax.set_ylabel("State Value")
            ax.set_title("State Trajectory (First 7 Dimensions)")
            ax.legend(loc='upper right', ncol=2, fontsize=8)
            ax.grid(True, alpha=0.3)

            # Plot actions if available
            if 'action' in df.columns:
                actions = np.stack(df['action'].values)
                print(f"Action shape: {actions.shape}")

                ax = axes[1]
                for i in range(min(7, actions.shape[1])):
                    ax.plot(actions[:, i], label=f'Action {i}', alpha=0.7)
                ax.set_xlabel("Timestep")
                ax.set_ylabel("Action Value")
                ax.set_title("Action Trajectory (First 7 Dimensions)")
                ax.legend(loc='upper right', ncol=2, fontsize=8)
                ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(self.save_dir / f"episode_{episode_idx}_trajectory.png", dpi=150)
            plt.close()
            print(f"\n  Saved: {self.save_dir / f'episode_{episode_idx}_trajectory.png'}")

        # Save sample data as CSV for inspection
        sample_df = df.head(20)
        # Convert array columns to strings for CSV
        for col in sample_df.columns:
            if sample_df[col].dtype == object:
                sample_df[col] = sample_df[col].apply(lambda x: str(x)[:100] if isinstance(x, (list, np.ndarray)) else x)
        sample_df.to_csv(self.save_dir / f"episode_{episode_idx}_sample.csv", index=False)
        print(f"  Saved: {self.save_dir / f'episode_{episode_idx}_sample.csv'}")

    def extract_video_frames(self, episode_idx: int = 0, num_frames: int = 6):
        """Extract and visualize sample frames from episode video."""
        print(f"\n{'='*60}")
        print(f"8. VIDEO FRAME EXTRACTION (Episode {episode_idx})")
        print(f"{'='*60}")

        video_path_pattern = self.info.get("video_path", "")
        if not video_path_pattern:
            print("No video path pattern found")
            return

        # Find available video keys
        video_dir = self.dataset_path / "videos"
        if not video_dir.exists():
            print("No videos directory found")
            return

        chunk_idx = episode_idx // self.info.get("chunks_size", 1000)

        # Find all video files for this episode
        video_files = []
        for video_key_dir in (video_dir / f"chunk-{chunk_idx:03d}").glob("*"):
            if video_key_dir.is_dir():
                video_file = video_key_dir / f"episode_{episode_idx:06d}.mp4"
                if video_file.exists():
                    video_files.append((video_key_dir.name, video_file))

        if not video_files:
            print(f"No video files found for episode {episode_idx}")
            return

        print(f"\nFound {len(video_files)} camera view(s):")
        for name, path in video_files:
            print(f"  - {name}: {path.name}")

        if not VIDEO_BACKEND:
            print("\nNo video backend available to extract frames")
            return

        if not HAS_MATPLOTLIB:
            print("\nMatplotlib not available for visualization")
            return

        # Extract frames from first video
        video_key, video_path = video_files[0]
        print(f"\nExtracting frames from: {video_key}")

        try:
            if VIDEO_BACKEND == "decord":
                import decord
                vr = decord.VideoReader(str(video_path))
                total_frames = len(vr)
                indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
                frames = [vr[i].asnumpy() for i in indices]

            elif VIDEO_BACKEND == "imageio":
                import imageio
                reader = imageio.get_reader(str(video_path))
                total_frames = reader.count_frames()
                indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
                frames = [reader.get_data(i) for i in indices]
                reader.close()

            else:
                print(f"Video backend '{VIDEO_BACKEND}' not fully supported")
                return

            print(f"  Total frames in video: {total_frames}")
            print(f"  Extracted {len(frames)} sample frames")
            print(f"  Frame shape: {frames[0].shape}")

            # Plot frames
            fig, axes = plt.subplots(1, len(frames), figsize=(3 * len(frames), 3))
            if len(frames) == 1:
                axes = [axes]

            for ax, frame, idx in zip(axes, frames, indices):
                ax.imshow(frame)
                ax.set_title(f"Frame {idx}", fontsize=10)
                ax.axis('off')

            plt.suptitle(f"Episode {episode_idx}: {video_key}", fontsize=12)
            plt.tight_layout()
            plt.savefig(self.save_dir / f"episode_{episode_idx}_frames.png", dpi=150)
            plt.close()
            print(f"\n  Saved: {self.save_dir / f'episode_{episode_idx}_frames.png'}")

            # Save individual frames
            for i, (frame, idx) in enumerate(zip(frames, indices)):
                frame_path = self.save_dir / f"episode_{episode_idx}_frame_{idx:04d}.png"
                plt.imsave(str(frame_path), frame)
            print(f"  Saved {len(frames)} individual frame images")

        except Exception as e:
            print(f"Error extracting frames: {e}")

    def generate_report(self) -> Path:
        """Generate comprehensive markdown report."""
        print(f"\n{'='*60}")
        print("GENERATING COMPREHENSIVE REPORT")
        print(f"{'='*60}")

        report = f"""# GR00T Dataset Analysis Report

**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Dataset**: `{self.dataset_path}`

---

## 1. Overview

This report analyzes a LeRobot v2 format dataset used for GR00T VLA model training.

| Property | Value |
|----------|-------|
| Format Version | {self.info.get('codebase_version', 'N/A')} |
| Robot Type | {self.info.get('robot_type', 'N/A')} |
| Total Episodes | {self.info.get('total_episodes', 'N/A')} |
| Total Frames | {self.info.get('total_frames', 'N/A')} |
| FPS | {self.info.get('fps', 'N/A')} |

---

## 2. Dataset Structure

```
{self.dataset_path.name}/
├── meta/
│   ├── info.json          # Dataset configuration
│   ├── episodes.jsonl     # Per-episode metadata
│   ├── tasks.jsonl        # Task descriptions
│   ├── modality.json      # Data layout mapping
│   └── stats.json         # Normalization statistics
├── data/
│   └── chunk-XXX/
│       └── episode_XXXXXX.parquet  # State/action data
└── videos/
    └── chunk-XXX/
        └── <camera_name>/
            └── episode_XXXXXX.mp4  # Video frames
```

---

## 3. Modality Configuration

The `modality.json` defines how raw data maps to model inputs:

### State (Robot Proprioception)
"""
        if "state" in self.modality:
            report += "| Body Part | Index Range | Dimensions |\n|-----------|-------------|------------|\n"
            for name, config in self.modality["state"].items():
                dim = config["end"] - config["start"]
                report += f"| {name} | [{config['start']}:{config['end']}] | {dim} |\n"

        report += """
### Action (Robot Commands)
"""
        if "action" in self.modality:
            report += "| Body Part | Index Range | Dimensions |\n|-----------|-------------|------------|\n"
            for name, config in self.modality["action"].items():
                dim = config["end"] - config["start"]
                report += f"| {name} | [{config['start']}:{config['end']}] | {dim} |\n"

        report += """
### Video (Camera Inputs)
"""
        if "video" in self.modality:
            report += "| Key | Original Data Path |\n|-----|-------------------|\n"
            for name, config in self.modality["video"].items():
                report += f"| {name} | {config.get('original_key', 'N/A')} |\n"

        report += """
---

## 4. Task Descriptions

Language instructions used for training:

"""
        for task in self.tasks:
            if task.get("task") != "valid":
                report += f"- [{task.get('task_index')}] {task.get('task')}\n"

        report += """
---

## 5. Data Flow: How the Model Uses This Data

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING DATA FLOW                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. LOAD EPISODE                                                │
│     parquet file → DataFrame with state, action, timestamps    │
│     video file → Frame sequence (H×W×3 RGB images)             │
│                                                                 │
│  2. APPLY MODALITY MAPPING                                      │
│     state[0:7] → left_arm joints                               │
│     state[7:13] → left_hand joints                             │
│     ... (as defined in modality.json)                          │
│                                                                 │
│  3. NORMALIZE                                                   │
│     normalized = (raw - mean) / std                            │
│     (using stats.json values)                                  │
│                                                                 │
│  4. CREATE TRAINING SAMPLE                                      │
│     ┌─────────────────┐                                        │
│     │ Video frames    │ → Vision Encoder → Visual tokens       │
│     │ State vector    │ → State Encoder → State tokens         │
│     │ Task text       │ → Text Encoder → Language tokens       │
│     └────────┬────────┘                                        │
│              │                                                  │
│              ▼                                                  │
│     ┌─────────────────┐                                        │
│     │ Fused Features  │ → Diffusion Transformer                │
│     └────────┬────────┘                                        │
│              │                                                  │
│              ▼                                                  │
│     ┌─────────────────┐                                        │
│     │ Action Chunk    │ → Loss vs Ground Truth Actions         │
│     │ (16 future      │                                        │
│     │  timesteps)     │                                        │
│     └─────────────────┘                                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Generated Files

| File | Description |
|------|-------------|
| `modality_config.json` | Modality configuration |
| `statistics.json` | Normalization statistics |
| `episode_lengths.png` | Episode length distribution |
| `episode_X_trajectory.png` | State/action plots |
| `episode_X_frames.png` | Sample video frames |
| `episode_X_sample.csv` | Sample data rows |

---

## 7. Key Takeaways

1. **Format**: LeRobot v2 with GR00T extensions
2. **Video**: Stored as H.264 MP4, loaded frame-by-frame
3. **State/Action**: Stored in Parquet, indexed by body part
4. **Language**: Task descriptions map to episode annotations
5. **Normalization**: Mean/std statistics pre-computed for training

---

*Report generated by GR00T Dataset Analyzer*
"""

        report_path = self.save_dir / "dataset_analysis_report.md"
        with open(report_path, "w") as f:
            f.write(report)

        print(f"Report saved to: {report_path}")
        return report_path

    def run_full_analysis(self, episode_idx: int = 0):
        """Run complete dataset analysis."""
        self.analyze_structure()
        self.analyze_modality()
        self.analyze_features()
        self.analyze_episodes()
        self.analyze_tasks()
        self.analyze_statistics()
        self.visualize_episode_data(episode_idx)
        self.extract_video_frames(episode_idx)
        self.generate_report()

        print(f"\n{'='*60}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*60}")
        print(f"All outputs saved to: {self.save_dir}")


def main():
    parser = argparse.ArgumentParser(description="Analyze GR00T training datasets")
    parser.add_argument("--dataset-path", type=str, required=True,
                        help="Path to LeRobot format dataset")
    parser.add_argument("--save-dir", type=str, default="./dataset_analysis",
                        help="Directory to save analysis outputs")
    parser.add_argument("--episode", type=int, default=0,
                        help="Episode index to analyze in detail")

    args = parser.parse_args()

    analyzer = DatasetAnalyzer(
        dataset_path=args.dataset_path,
        save_dir=args.save_dir
    )

    analyzer.run_full_analysis(episode_idx=args.episode)


if __name__ == "__main__":
    main()
