# Evaluation Sets Overview & Quick Inference Guide

**Purpose:** Compare available evaluation benchmarks and provide step-by-step instructions for quick inference replication.

---

## Available Evaluation Sets

| Benchmark | Robot | Type | Setup Complexity | Dataset Size | Success Rate |
|-----------|-------|------|------------------|--------------|--------------|
| **LIBERO** | Franka Panda | Simulation | ⭐ Low | ~2GB per suite | 94-98% |
| SimplerEnv (Bridge) | WidowX | Simulation | ⭐⭐ Medium | ~5GB | 62% avg |
| SimplerEnv (Fractal) | Google Robot | Simulation | ⭐⭐ Medium | ~10GB | 68% avg |
| BEHAVIOR | R1 Pro Humanoid | Simulation | ⭐⭐⭐ High | Large | 26% avg |
| SO100 | SO100/SO101 | Real Robot | ⭐⭐⭐⭐ Hardware | Small | Variable |
| GR00T-WholeBodyControl | Unitree G1 | Simulation | ⭐⭐⭐ High | Large | Variable |

---

## Recommendation: LIBERO

**Why LIBERO is best for quick inference:**

1. **Lowest Setup Complexity** - Single HuggingFace download, no additional sim setup needed for open-loop eval
2. **Smallest Dataset** - ~2GB per task suite, quick to download
3. **Highest Success Rates** - 94-98% means the model reliably produces good results
4. **Well-Documented** - Clear instructions, modality.json provided
5. **Multiple Suites** - Choose from Spatial, Object, Goal, or 10 (Long) tasks
6. **Open-Loop Eval** - Can run inference without full simulation environment

### LIBERO Task Suites

| Suite | Focus | Tasks | Success Rate |
|-------|-------|-------|--------------|
| **Spatial** | Spatial reasoning | 10 | 97.65% |
| **Goal** | Goal-conditioned | 10 | 97.5% |
| **Object** | Object generalization | 10 | 98.45% |
| **10 (Long)** | Multi-step long-horizon | 10 | 94.35% |

**Recommended starting point:** `libero_spatial` or `libero_goal` (highest success, simpler tasks)

---

## Quick Inference: Step-by-Step Sequence

### Option A: Open-Loop Evaluation (Fastest - No Simulation)

Compares predicted actions vs ground-truth actions on dataset. No robot/sim interaction.

```
┌─────────────────────────────────────────────────────────────┐
│  Step 1: Download Dataset (~2GB)                            │
│  └── huggingface-cli download LIBERO dataset               │
├─────────────────────────────────────────────────────────────┤
│  Step 2: Download Model (~6GB)                              │
│  └── Auto-downloads nvidia/GR00T-N1.6-3B on first run      │
├─────────────────────────────────────────────────────────────┤
│  Step 3: Copy modality.json                                 │
│  └── Configure dataset for GR00T format                    │
├─────────────────────────────────────────────────────────────┤
│  Step 4: Run Open-Loop Eval                                 │
│  └── python gr00t/eval/open_loop_eval.py                   │
├─────────────────────────────────────────────────────────────┤
│  Step 5: View Results                                       │
│  └── Check /tmp/open_loop_eval/traj_*.jpeg plots           │
└─────────────────────────────────────────────────────────────┘
```

**Time Estimate:** ~30-60 minutes (mostly download time)

### Option B: Closed-Loop Simulation Evaluation (Full Pipeline)

Runs policy in actual LIBERO simulation environment.

```
┌─────────────────────────────────────────────────────────────┐
│  Steps 1-3: Same as Option A                                │
├─────────────────────────────────────────────────────────────┤
│  Step 4: Setup LIBERO Simulation                            │
│  └── bash gr00t/eval/sim/LIBERO/setup_libero.sh            │
├─────────────────────────────────────────────────────────────┤
│  Step 5: Start Policy Server (Terminal 1)                   │
│  └── python gr00t/eval/run_gr00t_server.py                 │
├─────────────────────────────────────────────────────────────┤
│  Step 6: Run Evaluation Client (Terminal 2)                 │
│  └── python rollout scripts                                │
└─────────────────────────────────────────────────────────────┘
```

**Time Estimate:** ~2-3 hours (includes sim setup)

---

## Detailed Commands: Option A (Open-Loop)

### Step 1: Download LIBERO Dataset

```bash
# Choose one suite (spatial recommended for first run)
huggingface-cli download \
    --repo-type dataset IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot \
    --local-dir examples/LIBERO/libero_spatial_no_noops_1.0.0_lerobot/
```

### Step 2: Configure Dataset

```bash
# Copy modality.json to dataset
cp examples/LIBERO/modality.json \
   examples/LIBERO/libero_spatial_no_noops_1.0.0_lerobot/meta/
```

### Step 3: Run Open-Loop Evaluation

```bash
# Using pretrained model (auto-downloads from HuggingFace)
uv run python gr00t/eval/open_loop_eval.py \
    --dataset-path examples/LIBERO/libero_spatial_no_noops_1.0.0_lerobot \
    --embodiment-tag LIBERO_PANDA \
    --model-path nvidia/GR00T-N1.6-3B \
    --traj-ids 0 1 2 \
    --action-horizon 16 \
    --steps 200
```

### Step 4: View Results

```bash
# Results saved to /tmp/open_loop_eval/
ls /tmp/open_loop_eval/
# traj_0.jpeg, traj_1.jpeg, traj_2.jpeg

# View plots (macOS)
open /tmp/open_loop_eval/traj_0.jpeg
```

---

## Expected Output

### Console Output
```
INFO:root:Dataset length: 50
INFO:root:Running evaluation on trajectories: [0, 1, 2]
INFO:root:Running trajectory: 0
INFO:root:inferencing at step: 0
INFO:root:inferencing at step: 16
...
INFO:root:Unnormalized Action MSE across single traj: 0.00234
INFO:root:Unnormalized Action MAE across single traj: 0.0312
INFO:root:Average MSE across all trajs: 0.00256
INFO:root:Average MAE across all trajs: 0.0298
```

### Plot Output
- X-axis: Timesteps
- Y-axis: Action values (joint positions)
- Blue line: Ground truth actions
- Orange line: Predicted actions
- Red dots: Inference points (every action_horizon steps)

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 16GB | 24GB+ |
| RAM | 32GB | 64GB |
| Disk | 20GB free | 50GB free |
| GPU | RTX 3090 | RTX 4090/A100 |

**Note:** Model runs at ~20 Hz on RTX 4090, ~27 Hz on RTX 5090

---

## Alternative: Use Pre-finetuned Checkpoint

If you want to skip finetuning and use NVIDIA's provided checkpoint:

```bash
# LIBERO doesn't have a pre-finetuned checkpoint, but SimplerEnv does:
# nvidia/GR00T-N1.6-bridge (for WidowX)
# nvidia/GR00T-N1.6-fractal (for Google Robot)

# For LIBERO, use base model:
--model-path nvidia/GR00T-N1.6-3B
```

---

## Troubleshooting

### Issue: CUDA Out of Memory
```bash
# Reduce batch size or use CPU (slower)
--device cpu
```

### Issue: Video Decoding Error
```bash
# Install torchcodec or use decord backend
pip install torchcodec
# or change video_backend in code to "decord"
```

### Issue: Model Download Fails
```bash
# Login to HuggingFace first
huggingface-cli login
```

---

## Summary: Recommended Sequence

1. **Download LIBERO spatial dataset** (~2GB, 5-10 min)
2. **Copy modality.json** (10 seconds)
3. **Run open_loop_eval.py** (auto-downloads model on first run)
4. **View trajectory plots** in /tmp/open_loop_eval/

**Total time to first inference: ~30-60 minutes** (depending on download speed)

This gives you predicted vs ground-truth action comparisons without needing simulation setup.
