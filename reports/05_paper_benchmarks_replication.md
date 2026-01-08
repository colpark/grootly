# GR00T N1 Paper Benchmarks - Replication Guide

**Paper:** [GR00T N1: An Open Foundation Model for Generalist Humanoid Robots](https://arxiv.org/abs/2503.14734)

---

## Complete Benchmark Results from Paper

### Summary Table (100 Demonstrations)

| Benchmark | Tasks | Diffusion Policy | GR00T N1 | Available in Repo |
|-----------|-------|------------------|----------|-------------------|
| **RoboCasa Kitchen** | 24 | 25.6% | 32.1% | ✅ Zero-shot |
| **DexMimicGen** | 9 | 56.1% | 66.5% | ❌ Not included |
| **GR-1 Tabletop** | 24 | 32.7% | 50.0% | ✅ Zero-shot |

---

## Detailed Per-Task Results

### 1. RoboCasa Kitchen (24 Tasks)

**Robot:** Panda + Omron Gripper
**Environment:** Realistic kitchen simulation
**Demos per task:** 100 (paper), Zero-shot available (repo)

| Task | DP 100 | GR00T 100 | N1.6 Zero-shot |
|------|--------|-----------|----------------|
| Close Drawer | 88.2% | 96.1% | **100.0%** |
| Close Single Door | 46.1% | 67.7% | **96.0%** |
| Close Double Door | 26.5% | 43.1% | **88.5%** |
| Coffee Press Button | 46.1% | 56.9% | **98.5%** |
| Turn Off Microwave | 53.9% | 57.8% | **96.0%** |
| Turn On Microwave | 51.0% | 73.5% | **91.5%** |
| Turn Off Sink Faucet | 63.7% | 67.7% | **93.5%** |
| Turn On Sink Faucet | 27.5% | 59.8% | **89.0%** |
| Turn Sink Spout | 11.8% | 42.2% | **87.0%** |
| Open Drawer | 42.2% | 42.2% | **81.1%** |
| Open Single Door | 42.2% | 54.9% | **81.5%** |
| Turn On Stove | 22.6% | 25.5% | **76.5%** |
| Coffee Serve Mug | 28.4% | 34.3% | **63.5%** |
| PnP Counter to Stove | 1.0% | 0.0% | **63.2%** |
| PnP Stove to Counter | 2.9% | 0.0% | **54.5%** |
| PnP Bottle to Cabinet | - | - | **51.5%** |
| PnP Counter to Sink | 0.0% | 1.0% | **46.0%** |
| PnP Sink to Counter | 8.8% | 5.9% | **50.0%** |
| PnP Cabinet to Counter | 4.9% | 3.9% | **41.0%** |
| PnP Counter to Cabinet | 2.9% | 6.9% | **47.5%** |
| Open Double Door | 9.8% | 12.8% | **39.0%** |
| Coffee Setup Mug | 19.6% | 2.0% | **31.0%** |
| Turn Off Stove | 10.8% | 15.7% | **31.0%** |
| PnP Microwave to Counter | 2.0% | 0.0% | **24.5%** |
| PnP Counter to Microwave | 2.0% | 0.0% | **19.0%** |
| **Average** | **25.6%** | **32.1%** | **66.22%** |

### 2. DexMimicGen Cross-Embodiment (9 Tasks)

**Robot:** Various (cross-embodiment)
**Demos per task:** 100
**⚠️ NOT available in repository**

| Task | DP 100 | GR00T 100 |
|------|--------|-----------|
| Can Sort | 93.1% | **98.0%** |
| Coffee | 68.1% | **79.4%** |
| Lift Tray | 25.0% | **77.5%** |
| Pouring | 62.3% | **71.6%** |
| Transport | 25.0% | **48.0%** |
| Three Piece Assembly | 32.5% | **43.1%** |
| Drawer Cleanup | 16.7% | **42.2%** |
| Threading | 18.3% | **37.3%** |
| Box Cleanup | 80.8% | 29.4% |
| **Average** | **46.9%** | **58.5%** |

### 3. GR-1 Tabletop (24 Tasks)

**Robot:** Fourier GR-1 Humanoid
**Environment:** Tabletop manipulation
**Demos per task:** 100 (paper), Zero-shot available (repo)

| Task | DP 100 | GR00T 100 | N1.6 Zero-shot |
|------|--------|-----------|----------------|
| Plate to Plate | - | - | **78.7%** |
| Tray to Plate | - | - | **71.0%** |
| Cutting Board to Pan | 48.0% | 65.7% | **68.5%** |
| Cutting Board to Pot | 37.3% | 57.8% | **65.0%** |
| Tray to Pot | - | - | **64.5%** |
| Placemat to Plate | 23.5% | 37.3% | **63.0%** |
| Cutting Board to Basket | 42.2% | 61.8% | **58.0%** |
| Placemat to Basket | 25.5% | 45.1% | **58.5%** |
| Placemat to Bowl | 18.6% | 39.2% | **57.5%** |
| Tray to Tiered Basket | - | - | **57.0%** |
| Plate to Bowl | - | - | **57.0%** |
| Bottle to Cabinet | - | - | **51.5%** |
| Tray to Cardboard Box | - | - | **51.5%** |
| Plate to Pan | - | - | **51.0%** |
| Cutting Board to Tiered Basket | 13.7% | 23.5% | **46.5%** |
| Cutting Board to Cardboard Box | 15.7% | 30.4% | **46.5%** |
| Plate to Cardboard Box | - | - | **43.5%** |
| Potato to Microwave | - | - | **41.5%** |
| Tray to Tiered Shelf | - | - | **31.5%** |
| Placemat to Tiered Shelf | - | - | **28.5%** |
| Wine to Cabinet | - | - | **16.5%** |
| Milk to Microwave | - | - | **14.0%** |
| Can to Drawer | - | - | **13.0%** |
| Cup to Drawer | - | - | **8.5%** |
| **Average** | **32.7%** | **50.0%** | **47.6%** |

---

## Top 3 Recommendations for Replication

### Ranking Criteria
- ✅ Available in repository
- ✅ Zero-shot evaluation (no finetuning needed)
- ✅ Good success rate
- ✅ Small scale / quick to run

---

## 🥇 #1: RoboCasa Kitchen (Selected Tasks)

**Why:** Highest success rates, zero-shot ready, well-documented

### Best Tasks for Quick Replication

| Task | Success Rate | Complexity |
|------|--------------|------------|
| Close Drawer | **100.0%** | ⭐ Simple |
| Coffee Press Button | **98.5%** | ⭐ Simple |
| Close Single Door | **96.0%** | ⭐ Simple |
| Turn Off Microwave | **96.0%** | ⭐ Simple |
| Turn Off Sink Faucet | **93.5%** | ⭐⭐ Medium |

### Dataset Details
- **Size:** ~500MB per task subset
- **Format:** RoboCasa simulation format
- **Robot:** Panda arm with Omron gripper
- **Action space:** 7-DOF + gripper
- **Observation:** RGB images + proprioception

### Setup Commands
```bash
# Install simulation environment
sudo apt install libegl1-mesa-dev libglu1-mesa
bash gr00t/eval/sim/robocasa/setup_RoboCasa.sh

# Run evaluation (Terminal 1 - Server)
uv run python gr00t/eval/run_gr00t_server.py \
    --model-path nvidia/GR00T-N1.6-3B \
    --embodiment-tag ROBOCASA_PANDA_OMRON \
    --use-sim-policy-wrapper

# Run evaluation (Terminal 2 - Client)
gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python gr00t/eval/rollout_policy.py \
    --n_episodes 10 \
    --env_name robocasa_panda_omron/CloseDrawer_PandaOmron_Env \
    --n_action_steps 8 \
    --n_envs 5
```

---

## 🥈 #2: GR-1 Tabletop (Selected Tasks)

**Why:** Humanoid robot (paper's main focus), zero-shot ready, variety of tasks

### Best Tasks for Quick Replication

| Task | Success Rate | Complexity |
|------|--------------|------------|
| Plate to Plate | **78.7%** | ⭐⭐ Medium |
| Tray to Plate | **71.0%** | ⭐⭐ Medium |
| Cutting Board to Pan | **68.5%** | ⭐⭐ Medium |
| Cutting Board to Pot | **65.0%** | ⭐⭐ Medium |
| Tray to Pot | **64.5%** | ⭐⭐ Medium |

### Dataset Details
- **Size:** ~1GB per task (1000 demos each)
- **Format:** RoboCasa GR1 format
- **Robot:** Fourier GR-1 humanoid (bimanual)
- **Action space:** Arms + waist + hands
- **Observation:** RGB images + proprioception

### Setup Commands
```bash
# Install simulation environment
bash gr00t/eval/sim/robocasa-gr1-tabletop-tasks/setup_RoboCasaGR1TabletopTasks.sh

# Run evaluation (Terminal 1 - Server)
uv run python gr00t/eval/run_gr00t_server.py \
    --model-path nvidia/GR00T-N1.6-3B \
    --embodiment-tag GR1 \
    --use-sim-policy-wrapper

# Run evaluation (Terminal 2 - Client)
gr00t/eval/sim/robocasa-gr1-tabletop-tasks/robocasa_uv/.venv/bin/python \
    gr00t/eval/rollout_policy.py \
    --n_episodes 10 \
    --env_name gr1_unified/PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_Env \
    --n_action_steps 8 \
    --n_envs 5
```

---

## 🥉 #3: RoboCasa Kitchen - Articulated Tasks

**Why:** Different skill category, still high success, tests different capabilities

### Best Articulated Tasks

| Task | Success Rate | Skill Type |
|------|--------------|------------|
| Turn On Microwave | **91.5%** | Button press |
| Turn On Sink Faucet | **89.0%** | Handle rotation |
| Turn Sink Spout | **87.0%** | Object manipulation |
| Open Drawer | **81.1%** | Pull action |
| Open Single Door | **81.5%** | Swing action |

### Why This Category
- Tests **articulated object manipulation** (different from pick-and-place)
- High success rates for replication confidence
- Validates model's understanding of different motion primitives

---

## Replication Priority Order

For quickest path to replicating paper results:

| Priority | Benchmark | Task | Expected Success | Time to Run |
|----------|-----------|------|------------------|-------------|
| 1 | RoboCasa | CloseDrawer | ~100% | 10 min |
| 2 | RoboCasa | CoffeePressButton | ~98% | 10 min |
| 3 | RoboCasa | TurnOnMicrowave | ~91% | 10 min |
| 4 | GR-1 | PlateToPlateSplitA | ~79% | 15 min |
| 5 | GR-1 | CuttingBoardToPan | ~68% | 15 min |

---

## Hardware Requirements

| Component | RoboCasa | GR-1 Tabletop |
|-----------|----------|---------------|
| GPU VRAM | 16GB+ | 16GB+ |
| RAM | 32GB | 32GB |
| Disk | 10GB | 15GB |
| Display | Required (sim) | Required (sim) |

---

## What's NOT Available for Replication

| Benchmark | Reason |
|-----------|--------|
| DexMimicGen | Not included in repository |
| Real GR-1 | Requires physical robot hardware |
| YAM Bimanual | Requires physical robot hardware |
| Agibot Genie-1 | Requires physical robot hardware |
| Unitree G1 | Requires IsaacLab setup (separate) |

---

## Summary

**For paper replication, use:**
1. **RoboCasa Kitchen** - Best success rates, quickest setup
2. **GR-1 Tabletop** - Paper's main humanoid benchmark
3. Start with high-success tasks (>90%) to validate setup works

**Estimated time to first successful replication:** 1-2 hours (including sim setup)
