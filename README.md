# Grootly - Isaac GR00T Analysis & Documentation

Analysis and documentation for NVIDIA's Isaac GR00T N1.6 robotics foundation model.

## Overview

This repository contains comprehensive analysis of the [Isaac GR00T N1.6](https://github.com/NVIDIA/Isaac-GR00T) codebase, including:

- Data processing pipeline documentation
- Input/output specifications
- Model architecture analysis
- Quick reference guides

## Reports

| Report | Description |
|--------|-------------|
| [01_codebase_analysis.md](reports/01_codebase_analysis.md) | Complete architecture analysis with 8-stage data pipeline |
| [02_data_flow_quick_reference.md](reports/02_data_flow_quick_reference.md) | Visual diagrams and quick code snippets |
| [03_io_specifications.md](reports/03_io_specifications.md) | Detailed input/output formats with validation |

## Quick Summary

### Data Flow
```
Raw Data (LeRobot v2) → Episode Loading → Step Extraction →
Processing → Collation → Model Inference → Action Output
```

### Input Format
```python
observation = {
    "video": {"ego_view": np.ndarray(B, T, H, W, 3)},  # uint8
    "state": {"joint_state": np.ndarray(B, T, D)},     # float32
    "language": {"task": [["pick up cube"]]}           # List[List[str]]
}
```

### Output Format
```python
action_dict = {
    "joint_action": np.ndarray(B, horizon, 7),   # radians
    "gripper_action": np.ndarray(B, horizon, 1)  # 0-1 range
}
```

## Setup

1. Clone this repository:
```bash
git clone https://github.com/colpark/grootly.git
cd grootly
```

2. Clone Isaac GR00T (not included in this repo):
```bash
git clone https://github.com/NVIDIA/Isaac-GR00T.git
```

3. Follow the [official setup guide](https://github.com/NVIDIA/Isaac-GR00T#installation)

## License

Analysis documentation is provided as-is. Isaac GR00T is licensed under [NVIDIA's license](https://github.com/NVIDIA/Isaac-GR00T/blob/main/LICENSE).
