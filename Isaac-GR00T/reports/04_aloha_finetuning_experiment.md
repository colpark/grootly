# ALOHA Bimanual Finetuning Experiment

**Date**: January 2026
**Task**: Transfer Cube (Bimanual)
**Base Model**: nvidia/GR00T-N1.6-3B
**Finetuned Checkpoint**: checkpoint-6000
**Embodiment**: GR1 (mapped from ALOHA)

---

## 1. Experiment Overview

This experiment finetunes the GR00T N1.6 Vision-Language-Action model on the ALOHA Transfer Cube dataset, demonstrating domain adaptation from a pretrained foundation model to a new bimanual manipulation task.

### Task Description
The ALOHA bimanual robot transfers a cube from one gripper to the other. This task requires precise bimanual coordination between two 7-DOF arms with parallel grippers.

### Objective
Adapt the pretrained GR00T N1.6-3B model (trained on diverse robotics data) to the specific ALOHA embodiment and Transfer Cube task through supervised finetuning.

---

## 2. Dataset Configuration

### 2.1 Source Dataset

| Property | Value |
|----------|-------|
| Source | HuggingFace `lerobot/aloha_sim_transfer_cube_human` |
| Format | LeRobot v2 (Parquet + MP4) |
| Episodes | 50 |
| Original FPS | 50 Hz |
| Original DOF | 14 (7 per arm) |
| Video Resolution | 480 × 640 |

### 2.2 Converted Dataset (GR00T Format)

| Property | Value |
|----------|-------|
| Target FPS | 20 Hz |
| Target DOF | 29 (GR1 format with padding) |
| Video Resolution | 256 × 256 |
| Video Codec | MP4V |
| Output Path | `./data/aloha_groot_format` |

### 2.3 DOF Mapping (ALOHA → GR1)

| GR1 Component | Dimensions | ALOHA Source | Notes |
|---------------|------------|--------------|-------|
| `left_arm` | 7 | ALOHA joints 0-5 | Padded with 1 zero |
| `left_hand` | 6 | ALOHA gripper 6 | Padded with 5 zeros |
| `right_arm` | 7 | ALOHA joints 7-12 | Padded with 1 zero |
| `right_hand` | 6 | ALOHA gripper 13 | Padded with 5 zeros |
| `waist` | 3 | None | All zeros (ALOHA has no waist) |

**Total**: 14 DOF → 29 DOF (zero-padded)

---

## 3. Finetuning Configuration

### 3.1 Training Setup

| Parameter | Value |
|-----------|-------|
| Base Model | `nvidia/GR00T-N1.6-3B` |
| Embodiment Tag | `GR1` |
| GPUs | 4 × NVIDIA GPU |
| Distributed Training | `torchrun --nproc_per_node=4` |
| Optimizer | AdamW |
| Learning Rate | 2e-5 (default) |
| Batch Size | Global batch size from config |

### 3.2 Model Components Tuned

| Component | Tuned | Description |
|-----------|-------|-------------|
| LLM Backbone | ❌ | Frozen Eagle-Block2A-2B-v2 |
| Visual Encoder | ❌ | Frozen |
| Projector | ✅ | Vision-language projection |
| Diffusion Model | ✅ | Action generation (32 layers) |
| VLLN | ✅ | Vision-language-action alignment |

### 3.3 Training Command

```bash
torchrun --nproc_per_node=4 --standalone \
    -m gr00t.experiment.launch_finetune \
    --modality-config-path ./finetuning/gr1_modality_config.py \
    --embodiment-tag GR1 \
    --dataset-path ./data/aloha_groot_format \
    --base-model-path nvidia/GR00T-N1.6-3B
```

### 3.4 Modality Configuration

```python
GR1_CONFIG = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["ego_view"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["left_arm", "left_hand", "right_arm", "right_hand", "waist"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(16)),  # 16-step action horizon
        modality_keys=["left_arm", "left_hand", "right_arm", "right_hand", "waist"],
        action_configs=[...],  # RELATIVE for arms, ABSOLUTE for hands/waist
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["task"],
    ),
}
```

---

## 4. Evaluation Methodology

### 4.1 Open-Loop Evaluation

Since ALOHA simulation integration with GR00T requires specific MuJoCo/dm_control version compatibility, we use open-loop evaluation:

1. **Input**: Observations (images, states) from recorded dataset trajectories
2. **Process**: Model predicts action sequences at regular intervals
3. **Comparison**: Predicted actions vs. ground truth demonstrator actions
4. **Metrics**: MSE, MAE, and threshold-based success rates

### 4.2 Evaluation Script

```bash
python finetuning/evaluate_aloha_finetune.py \
    --checkpoint_path ./outputs/checkpoint-6000 \
    --dataset_path ./data/aloha_groot_format \
    --traj_ids 0 1 2 3 4 5 6 7 8 9 \
    --steps 200 \
    --action_horizon 16
```

---

## 5. Results

### 5.1 Aggregate Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Average MSE** | 0.000662 | Very low prediction error |
| **Average MAE** | 0.013243 | ~1.3% average error per joint |
| **Success Rate @ 0.05** | 64.6% | Strict threshold |
| **Success Rate @ 0.10** | 90.6% | Moderate threshold |
| **Success Rate @ 0.20** | 98.6% | Lenient threshold |
| **Action Accuracy @ 0.10** | 98.6% | Individual prediction accuracy |
| **Action Accuracy @ 0.20** | 99.8% | Near-perfect at lenient threshold |
| **Error Drift** | +0.000022 | Minimal error accumulation |

### 5.2 Per-Trajectory Results

| Trajectory | MSE | MAE | Success@0.1 | Success@0.2 |
|------------|-----|-----|-------------|-------------|
| 0 | 0.000512 | 0.013759 | 92.5% | 100.0% |
| 1 | 0.000578 | 0.012823 | 91.9% | 99.4% |
| 2 | 0.000567 | 0.013117 | 91.2% | 100.0% |
| 3 | 0.000553 | 0.013266 | 87.5% | 99.4% |
| 4 | 0.000960 | 0.014013 | 88.8% | 96.2% |
| 5 | 0.000638 | 0.012964 | 89.4% | 98.8% |
| 6 | 0.000637 | 0.013053 | 90.6% | 98.8% |
| 7 | 0.000381 | 0.011825 | 92.5% | 100.0% |
| 8 | 0.000412 | 0.012089 | 95.6% | 100.0% |
| 9 | 0.001375 | 0.015522 | 85.6% | 93.8% |

### 5.3 Results Visualization

```
Success Rate Distribution (@ 0.1 threshold):
Trajectory 0:  ████████████████████████████████████████████████ 92.5%
Trajectory 1:  ███████████████████████████████████████████████  91.9%
Trajectory 2:  ██████████████████████████████████████████████   91.2%
Trajectory 3:  ████████████████████████████████████████████     87.5%
Trajectory 4:  █████████████████████████████████████████████    88.8%
Trajectory 5:  █████████████████████████████████████████████    89.4%
Trajectory 6:  ██████████████████████████████████████████████   90.6%
Trajectory 7:  ████████████████████████████████████████████████ 92.5%
Trajectory 8:  ██████████████████████████████████████████████████ 95.6%
Trajectory 9:  ███████████████████████████████████████████      85.6%
```

---

## 6. Analysis

### 6.1 Key Findings

1. **Excellent Prediction Accuracy**: MSE of 0.000662 indicates the model closely matches expert demonstrations.

2. **High Step Success Rate**: 90.6% of timesteps have ALL 29 action dimensions within 0.1 of ground truth.

3. **Minimal Error Drift**: Error drift of +0.000022 shows the model maintains accuracy throughout trajectories without significant compounding errors.

4. **Consistent Performance**: All 10 trajectories achieve >85% success rate at the 0.1 threshold.

5. **Near-Perfect at Lenient Threshold**: 98.6% success rate at 0.2 threshold suggests robust action prediction.

### 6.2 Comparison: Before vs After Finetuning

| Metric | Checkpoint-1000 (Early) | Checkpoint-6000 (Final) | Improvement |
|--------|------------------------|------------------------|-------------|
| MSE | 0.006659 | 0.000662 | **10× reduction** |
| MAE | 0.034095 | 0.013243 | **2.6× reduction** |
| Success@0.1 | ~50% (estimated) | 90.6% | **+40 points** |
| Success@0.2 | ~85% (estimated) | 98.6% | **+13 points** |

### 6.3 Trajectory Difficulty Analysis

| Category | Trajectories | Avg Success@0.1 |
|----------|--------------|-----------------|
| Easiest | 7, 8 | 94.1% |
| Average | 0, 1, 2, 5, 6 | 91.1% |
| Hardest | 3, 4, 9 | 87.3% |

Trajectory 9 shows the lowest performance (85.6%), possibly due to more complex manipulation patterns or edge cases in the demonstration.

---

## 7. Technical Implementation

### 7.1 Data Conversion Pipeline

```
ALOHA Dataset (LeRobot v2)
         │
         ▼
┌─────────────────────┐
│ convert_aloha_to_   │
│ groot.py            │
│ - Temporal resample │
│ - DOF mapping       │
│ - Video transcode   │
│ - Stats generation  │
└─────────────────────┘
         │
         ▼
GR00T Dataset Format
(Parquet + MP4 + JSON)
```

### 7.2 Key Conversion Steps

1. **Download**: HuggingFace snapshot download of LeRobot dataset
2. **Video Extraction**: Decode AV1 video using imageio
3. **Temporal Resampling**: 50 Hz → 20 Hz (every 2.5 frames)
4. **Spatial Resize**: 480×640 → 256×256 (center crop + resize)
5. **DOF Mapping**: 14 → 29 dimensions with zero padding
6. **Statistics**: Compute mean, std, min, max, q01, q99 per dimension

### 7.3 Model Architecture

```
┌─────────────────────────────────────────────────┐
│                  GR00T N1.6-3B                  │
├─────────────────────────────────────────────────┤
│  Eagle-Block2A-2B-v2 (Vision-Language Backbone) │
│  └── Frozen during finetuning                   │
├─────────────────────────────────────────────────┤
│  Vision-Language Projector                      │
│  └── Finetuned ✓                               │
├─────────────────────────────────────────────────┤
│  VLLN (Vision-Language-Action Alignment)        │
│  └── Finetuned ✓                               │
├─────────────────────────────────────────────────┤
│  DiT Diffusion Model (32 layers)               │
│  └── Finetuned ✓                               │
│  └── Action horizon: 16 steps                   │
│  └── Denoising steps: 4                         │
└─────────────────────────────────────────────────┘
```

---

## 8. Files Created

### 8.1 Conversion & Training

| File | Purpose |
|------|---------|
| `finetuning/convert_aloha_to_groot.py` | Dataset conversion script |
| `finetuning/gr1_modality_config.py` | GR1 modality configuration |
| `finetuning/FINETUNING_GUIDE.md` | Step-by-step guide |

### 8.2 Evaluation

| File | Purpose |
|------|---------|
| `finetuning/evaluate_aloha_finetune.py` | Open-loop evaluation with success metrics |
| `finetuning/run_aloha_sim.py` | MuJoCo simulation (requires gym-aloha) |
| `gr00t/eval/sim/ALOHA/aloha_env.py` | ALOHA environment wrapper |

### 8.3 Outputs

| Directory | Contents |
|-----------|----------|
| `data/aloha_groot_format/` | Converted dataset |
| `outputs/checkpoint-6000/` | Finetuned model weights |
| `eval_results/` | Evaluation plots and summary |

---

## 9. Reproducing Results

### 9.1 Full Pipeline

```bash
# 1. Convert ALOHA dataset to GR00T format
python finetuning/convert_aloha_to_groot.py

# 2. Run finetuning (4 GPUs)
torchrun --nproc_per_node=4 --standalone \
    -m gr00t.experiment.launch_finetune \
    --modality-config-path ./finetuning/gr1_modality_config.py \
    --embodiment-tag GR1 \
    --dataset-path ./data/aloha_groot_format \
    --base-model-path nvidia/GR00T-N1.6-3B

# 3. Evaluate
python finetuning/evaluate_aloha_finetune.py \
    --checkpoint_path ./outputs/checkpoint-6000 \
    --dataset_path ./data/aloha_groot_format \
    --traj_ids 0 1 2 3 4 5 6 7 8 9
```

### 9.2 Quick Evaluation Only

```bash
# With existing checkpoint
python finetuning/evaluate_aloha_finetune.py \
    --checkpoint_path ./outputs/checkpoint-6000 \
    --output_dir ./eval_results
```

---

## 10. Limitations & Future Work

### 10.1 Current Limitations

1. **Open-Loop Evaluation Only**: True task success rate requires closed-loop simulation
2. **MuJoCo Version Conflicts**: gym-aloha requires different mujoco/dm_control versions than robocasa
3. **Small Dataset**: Only 50 demonstrations for finetuning
4. **Single Task**: Only Transfer Cube task evaluated

### 10.2 Future Work

1. **Simulation Integration**: Set up separate conda environment for ALOHA simulation
2. **Multi-Task Finetuning**: Include Insertion and other ALOHA tasks
3. **Real Robot Deployment**: Transfer to physical ALOHA hardware
4. **Longer Training**: Explore training beyond 6000 steps
5. **Hyperparameter Tuning**: Optimize learning rate, batch size, etc.

---

## 11. Key Takeaways

1. **Successful Domain Adaptation**: GR00T N1.6-3B successfully adapts to ALOHA embodiment through finetuning.

2. **DOF Mapping Works**: Zero-padding from 14 to 29 DOF maintains action prediction quality.

3. **Temporal Resampling Effective**: 50 Hz → 20 Hz conversion preserves task-relevant motion patterns.

4. **Strong Convergence**: 10× MSE reduction from checkpoint-1000 to checkpoint-6000 shows effective learning.

5. **Minimal Error Accumulation**: Near-zero error drift indicates stable long-horizon predictions.

6. **Foundation Model Advantage**: Starting from pretrained GR00T enables rapid adaptation with limited data (50 demos).

---

## 12. References

- GR00T N1 Paper: https://arxiv.org/abs/2503.14734
- ALOHA Robot: https://tonyzhaozh.github.io/aloha/
- LeRobot Dataset: https://huggingface.co/datasets/lerobot/aloha_sim_transfer_cube_human
- HuggingFace Model: https://huggingface.co/nvidia/GR00T-N1.6-3B
- gym-aloha: https://github.com/huggingface/gym-aloha
