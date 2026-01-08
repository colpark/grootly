# Grootly - Isaac GR00T Analysis & Documentation

Analysis and documentation for NVIDIA's Isaac GR00T N1.6 robotics foundation model.

## Overview

This repository contains:

- **Isaac-GR00T/** - Complete NVIDIA Isaac GR00T N1.6 source code
- **reports/** - Comprehensive analysis and documentation

## Repository Structure

```
grootly/
├── Isaac-GR00T/           # NVIDIA Isaac GR00T N1.6 source
│   ├── gr00t/             # Main package
│   │   ├── data/          # Data loading & preprocessing
│   │   ├── model/         # Model architecture
│   │   ├── policy/        # Inference interface
│   │   └── experiment/    # Training scripts
│   ├── examples/          # Robot-specific configs
│   ├── getting_started/   # Official documentation
│   └── scripts/           # Deployment tools
│
└── reports/               # Analysis documentation
    ├── 01_codebase_analysis.md
    ├── 02_data_flow_quick_reference.md
    └── 03_io_specifications.md
```

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

2. Install dependencies:
```bash
cd Isaac-GR00T
pip install -e .
```

3. See [Isaac-GR00T/README.md](Isaac-GR00T/README.md) for detailed setup instructions.

## License

- Analysis documentation (reports/) is provided as-is
- Isaac GR00T source code is licensed under [NVIDIA's license](Isaac-GR00T/LICENSE)
