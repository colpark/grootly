#!/usr/bin/env python3
"""
Plot evaluation results across checkpoints with training details.

This script reads evaluation results from multiple checkpoint evaluations
and generates comparison plots showing training progress. It also extracts
training configuration, hyperparameters, and loss curves from the training output.

Usage:
    python finetuning/plot_checkpoint_results.py --eval_dir ./eval_results/ball2_groot
    python finetuning/plot_checkpoint_results.py --eval_dir ./eval_results/ball2_groot --checkpoint_dir ./outputs/ball2_groot
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


def find_checkpoint_results(eval_dir: Path) -> dict[int, dict]:
    """Find all checkpoint evaluation results."""
    results = {}

    # Look for step_* folders
    for step_dir in sorted(eval_dir.glob("step_*")):
        match = re.match(r"step_(\d+)", step_dir.name)
        if match:
            step = int(match.group(1))
            summary_file = step_dir / "evaluation_summary.txt"

            if summary_file.exists():
                metrics = parse_summary_file(summary_file)
                if metrics:
                    results[step] = metrics
                    print(f"  Found step {step}: MSE={metrics.get('mse', 'N/A'):.6f}, MAE={metrics.get('mae', 'N/A'):.6f}")

    return results


def parse_summary_file(summary_file: Path) -> dict:
    """Parse evaluation_summary.txt to extract metrics."""
    metrics = {}

    try:
        content = summary_file.read_text()

        # Extract basic metrics
        mse_match = re.search(r"Average MSE:\s+([\d.]+)", content)
        if mse_match:
            metrics['mse'] = float(mse_match.group(1))

        mae_match = re.search(r"Average MAE:\s+([\d.]+)", content)
        if mae_match:
            metrics['mae'] = float(mae_match.group(1))

        # Extract success rates
        for threshold in ['0.05', '0.10', '0.20', '0.50']:
            pattern = rf"@ {threshold} threshold:\s+([\d.]+)%"
            match = re.search(pattern, content)
            if match:
                metrics[f'success_{threshold.replace(".", "")}'] = float(match.group(1))

        # Extract accuracy rates
        for threshold in ['0.05', '0.10', '0.20', '0.50']:
            pattern = rf"Accuracy @ {threshold} threshold:\s+([\d.]+)%"
            match = re.search(pattern, content)
            if match:
                metrics[f'accuracy_{threshold.replace(".", "")}'] = float(match.group(1))

        # Extract per-component MAE
        base_match = re.search(r"Base Velocity:\s+([\d.]+)", content)
        if base_match:
            metrics['base_mae'] = float(base_match.group(1))

        left_match = re.search(r"Left Arm:\s+([\d.]+)", content)
        if left_match:
            metrics['left_arm_mae'] = float(left_match.group(1))

        right_match = re.search(r"Right Arm:\s+([\d.]+)", content)
        if right_match:
            metrics['right_arm_mae'] = float(right_match.group(1))

    except Exception as e:
        print(f"  Error parsing {summary_file}: {e}")

    return metrics


def load_training_config(checkpoint_dir: Path) -> dict:
    """Load training configuration from checkpoint directory."""
    config = {}

    # Try to load from experiment_cfg/conf.yaml (OmegaConf format)
    conf_yaml = checkpoint_dir / "experiment_cfg" / "conf.yaml"
    config_yaml = checkpoint_dir / "experiment_cfg" / "config.yaml"

    if conf_yaml.exists() and YAML_AVAILABLE:
        try:
            with open(conf_yaml) as f:
                config = yaml.safe_load(f)
            print(f"  Loaded config from: {conf_yaml}")
        except Exception as e:
            print(f"  Error loading {conf_yaml}: {e}")
    elif config_yaml.exists() and YAML_AVAILABLE:
        try:
            with open(config_yaml) as f:
                config = yaml.safe_load(f)
            print(f"  Loaded config from: {config_yaml}")
        except Exception as e:
            print(f"  Error loading {config_yaml}: {e}")

    return config


def load_trainer_state(checkpoint_dir: Path) -> dict:
    """Load trainer state from the latest checkpoint."""
    trainer_state = {}

    # Find all checkpoints and get the latest one
    checkpoints = sorted(checkpoint_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))

    if checkpoints:
        latest_checkpoint = checkpoints[-1]
        trainer_state_file = latest_checkpoint / "trainer_state.json"

        if trainer_state_file.exists():
            try:
                with open(trainer_state_file) as f:
                    trainer_state = json.load(f)
                print(f"  Loaded trainer state from: {trainer_state_file}")
            except Exception as e:
                print(f"  Error loading {trainer_state_file}: {e}")

    return trainer_state


def extract_training_losses(trainer_state: dict) -> tuple[list, list]:
    """Extract training loss history from trainer state."""
    steps = []
    losses = []

    log_history = trainer_state.get("log_history", [])

    for entry in log_history:
        if "loss" in entry and "step" in entry:
            steps.append(entry["step"])
            losses.append(entry["loss"])

    return steps, losses


def extract_hyperparameters(config: dict) -> dict:
    """Extract key hyperparameters from config."""
    hyperparams = {}

    # Training parameters
    training = config.get("training", {})
    if isinstance(training, dict):
        hyperparams["max_steps"] = training.get("max_steps", config.get("max_steps"))
        hyperparams["learning_rate"] = training.get("learning_rate")
        hyperparams["global_batch_size"] = training.get("global_batch_size")
        hyperparams["batch_size"] = training.get("batch_size")
        hyperparams["gradient_accumulation_steps"] = training.get("gradient_accumulation_steps")
        hyperparams["weight_decay"] = training.get("weight_decay")
        hyperparams["warmup_ratio"] = training.get("warmup_ratio")
        hyperparams["lr_scheduler_type"] = training.get("lr_scheduler_type")
        hyperparams["num_gpus"] = training.get("num_gpus")
        hyperparams["fp16"] = training.get("fp16")
        hyperparams["bf16"] = training.get("bf16")
        hyperparams["gradient_checkpointing"] = training.get("gradient_checkpointing")
        hyperparams["optim"] = training.get("optim")
        hyperparams["save_steps"] = training.get("save_steps", config.get("save_steps"))
        hyperparams["logging_steps"] = training.get("logging_steps")
        hyperparams["eval_steps"] = training.get("eval_steps")

    # Model parameters
    model = config.get("model", {})
    if isinstance(model, dict):
        hyperparams["model_type"] = model.get("type") or model.get("_target_", "").split(".")[-1]
        hyperparams["base_model"] = model.get("base_model_path") or model.get("pretrained_model_path")
        hyperparams["action_horizon"] = model.get("action_horizon")
        hyperparams["backbone_lora_rank"] = model.get("backbone_lora_rank")
        hyperparams["llm_lora_rank"] = model.get("llm_lora_rank")
        hyperparams["tune_llm"] = model.get("tune_llm")
        hyperparams["tune_visual"] = model.get("tune_visual")
        hyperparams["tune_projector"] = model.get("tune_projector")

    # Data parameters
    data = config.get("data", {})
    if isinstance(data, dict):
        hyperparams["dataset_path"] = data.get("dataset_path")
        hyperparams["seed"] = data.get("seed")

    # Clean up None values
    hyperparams = {k: v for k, v in hyperparams.items() if v is not None}

    return hyperparams


def estimate_model_params(config: dict) -> dict:
    """Estimate model size and trainable parameters."""
    model_info = {}

    model = config.get("model", {})
    if isinstance(model, dict):
        base_model = model.get("base_model_path", "")

        # Estimate based on model name
        if "3B" in base_model or "3b" in base_model:
            model_info["estimated_total_params"] = "~3B"
        elif "1B" in base_model or "1b" in base_model:
            model_info["estimated_total_params"] = "~1B"
        elif "7B" in base_model or "7b" in base_model:
            model_info["estimated_total_params"] = "~7B"

        # LoRA info
        backbone_lora = model.get("backbone_lora_rank")
        llm_lora = model.get("llm_lora_rank")

        if backbone_lora or llm_lora:
            model_info["finetuning_method"] = "LoRA"
            if backbone_lora:
                model_info["backbone_lora_rank"] = backbone_lora
            if llm_lora:
                model_info["llm_lora_rank"] = llm_lora
        else:
            model_info["finetuning_method"] = "Full Finetuning"

        # What's being tuned
        tuned_components = []
        if model.get("tune_llm"):
            tuned_components.append("LLM")
        if model.get("tune_visual"):
            tuned_components.append("Visual Backbone")
        if model.get("tune_projector"):
            tuned_components.append("Projector")
        if tuned_components:
            model_info["tuned_components"] = tuned_components

    return model_info


def plot_training_loss(loss_steps: list, losses: list, output_dir: Path, dataset_name: str):
    """Plot training loss curve."""
    if not loss_steps or not losses:
        print("  No training loss data available")
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(loss_steps, losses, '-', color='#2E86AB', linewidth=1.5, alpha=0.8)

    # Add smoothed line
    if len(losses) > 20:
        window = min(50, len(losses) // 5)
        smoothed = pd.Series(losses).rolling(window=window, center=True).mean()
        ax.plot(loss_steps, smoothed, '-', color='#E94F37', linewidth=2.5, label=f'Smoothed (window={window})')
        ax.legend(loc='upper right', fontsize=10)

    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(f'{dataset_name}: Training Loss', fontsize=14)
    ax.grid(True, alpha=0.3)

    # Add annotations
    min_loss_idx = np.argmin(losses)
    ax.annotate(f'Min: {losses[min_loss_idx]:.4f}',
                xy=(loss_steps[min_loss_idx], losses[min_loss_idx]),
                xytext=(10, 10), textcoords='offset points',
                fontsize=9, color='green',
                arrowprops=dict(arrowstyle='->', color='green', alpha=0.7))

    plt.tight_layout()
    plt.savefig(output_dir / 'training_loss.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: training_loss.png")


def plot_training_curves(results: dict[int, dict], output_dir: Path, dataset_name: str):
    """Generate training curve plots."""
    if not results:
        print("No results to plot!")
        return

    steps = sorted(results.keys())

    # Set up style
    plt.style.use('seaborn-v0_8-whitegrid')
    colors = plt.cm.tab10.colors

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. MSE and MAE over training
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    mse_values = [results[s].get('mse', np.nan) for s in steps]
    mae_values = [results[s].get('mae', np.nan) for s in steps]

    axes[0].plot(steps, mse_values, 'o-', color=colors[0], linewidth=2, markersize=8)
    axes[0].set_xlabel('Training Step', fontsize=12)
    axes[0].set_ylabel('Mean Squared Error', fontsize=12)
    axes[0].set_title(f'{dataset_name}: MSE over Training', fontsize=14)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, mae_values, 's-', color=colors[1], linewidth=2, markersize=8)
    axes[1].set_xlabel('Training Step', fontsize=12)
    axes[1].set_ylabel('Mean Absolute Error', fontsize=12)
    axes[1].set_title(f'{dataset_name}: MAE over Training', fontsize=14)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'eval_loss_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: eval_loss_curves.png")

    # 2. Success rates at different thresholds
    fig, ax = plt.subplots(figsize=(10, 6))

    thresholds = ['005', '010', '020', '050']
    threshold_labels = ['0.05', '0.10', '0.20', '0.50']

    for i, (thresh, label) in enumerate(zip(thresholds, threshold_labels)):
        values = [results[s].get(f'success_{thresh}', np.nan) for s in steps]
        if not all(np.isnan(values)):
            ax.plot(steps, values, 'o-', color=colors[i], linewidth=2, markersize=8, label=f'Success @ {label}')

    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title(f'{dataset_name}: Success Rates over Training', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(output_dir / 'success_rates.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: success_rates.png")

    # 3. Accuracy rates at different thresholds
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (thresh, label) in enumerate(zip(thresholds, threshold_labels)):
        values = [results[s].get(f'accuracy_{thresh}', np.nan) for s in steps]
        if not all(np.isnan(values)):
            ax.plot(steps, values, 's-', color=colors[i], linewidth=2, markersize=8, label=f'Accuracy @ {label}')

    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Accuracy Rate (%)', fontsize=12)
    ax.set_title(f'{dataset_name}: Accuracy Rates over Training', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_rates.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: accuracy_rates.png")

    # 4. Per-component MAE breakdown
    fig, ax = plt.subplots(figsize=(10, 6))

    base_values = [results[s].get('base_mae', np.nan) for s in steps]
    left_values = [results[s].get('left_arm_mae', np.nan) for s in steps]
    right_values = [results[s].get('right_arm_mae', np.nan) for s in steps]

    if not all(np.isnan(base_values)):
        ax.plot(steps, base_values, 'o-', color=colors[0], linewidth=2, markersize=8, label='Base Velocity')
    if not all(np.isnan(left_values)):
        ax.plot(steps, left_values, 's-', color=colors[1], linewidth=2, markersize=8, label='Left Arm')
    if not all(np.isnan(right_values)):
        ax.plot(steps, right_values, '^-', color=colors[2], linewidth=2, markersize=8, label='Right Arm')

    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Mean Absolute Error', fontsize=12)
    ax.set_title(f'{dataset_name}: Per-Component MAE over Training', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'component_mae.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: component_mae.png")

    # 5. Combined overview plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # MSE
    axes[0, 0].plot(steps, mse_values, 'o-', color=colors[0], linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Training Step')
    axes[0, 0].set_ylabel('MSE')
    axes[0, 0].set_title('Mean Squared Error')
    axes[0, 0].grid(True, alpha=0.3)

    # MAE
    axes[0, 1].plot(steps, mae_values, 's-', color=colors[1], linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Training Step')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].set_title('Mean Absolute Error')
    axes[0, 1].grid(True, alpha=0.3)

    # Success rates
    for i, (thresh, label) in enumerate(zip(thresholds, threshold_labels)):
        values = [results[s].get(f'success_{thresh}', np.nan) for s in steps]
        if not all(np.isnan(values)):
            axes[1, 0].plot(steps, values, 'o-', color=colors[i], linewidth=2, markersize=6, label=f'@ {label}')
    axes[1, 0].set_xlabel('Training Step')
    axes[1, 0].set_ylabel('Success Rate (%)')
    axes[1, 0].set_title('Success Rates (all dims < threshold)')
    axes[1, 0].legend(loc='best', fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0, 100)

    # Component MAE
    if not all(np.isnan(base_values)):
        axes[1, 1].plot(steps, base_values, 'o-', color=colors[0], linewidth=2, markersize=6, label='Base')
    if not all(np.isnan(left_values)):
        axes[1, 1].plot(steps, left_values, 's-', color=colors[1], linewidth=2, markersize=6, label='Left Arm')
    if not all(np.isnan(right_values)):
        axes[1, 1].plot(steps, right_values, '^-', color=colors[2], linewidth=2, markersize=6, label='Right Arm')
    axes[1, 1].set_xlabel('Training Step')
    axes[1, 1].set_ylabel('MAE')
    axes[1, 1].set_title('Per-Component MAE')
    axes[1, 1].legend(loc='best', fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(f'{dataset_name} - Training Progress Overview', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'training_overview.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: training_overview.png")


def create_summary_table(
    results: dict[int, dict],
    output_dir: Path,
    dataset_name: str,
    hyperparams: dict = None,
    model_info: dict = None,
    loss_steps: list = None,
    losses: list = None,
):
    """Create a comprehensive summary CSV and markdown report."""
    steps = sorted(results.keys())

    # Build DataFrame
    data = []
    for step in steps:
        r = results[step]
        data.append({
            'Step': step,
            'MSE': r.get('mse', np.nan),
            'MAE': r.get('mae', np.nan),
            'Success@0.05': r.get('success_005', np.nan),
            'Success@0.10': r.get('success_010', np.nan),
            'Success@0.20': r.get('success_020', np.nan),
            'Accuracy@0.10': r.get('accuracy_010', np.nan),
            'Base MAE': r.get('base_mae', np.nan),
            'Left Arm MAE': r.get('left_arm_mae', np.nan),
            'Right Arm MAE': r.get('right_arm_mae', np.nan),
        })

    df = pd.DataFrame(data)

    # Save CSV
    csv_path = output_dir / 'checkpoint_summary.csv'
    df.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"  Saved: checkpoint_summary.csv")

    # Save comprehensive markdown report
    md_path = output_dir / 'training_report.md'
    with open(md_path, 'w') as f:
        f.write(f"# {dataset_name} Training Report\n\n")

        # Training Configuration Section
        if hyperparams or model_info:
            f.write("## Training Configuration\n\n")

            if model_info:
                f.write("### Model\n\n")
                f.write("| Parameter | Value |\n")
                f.write("|-----------|-------|\n")
                for k, v in model_info.items():
                    if isinstance(v, list):
                        v = ", ".join(v)
                    f.write(f"| {k.replace('_', ' ').title()} | {v} |\n")
                f.write("\n")

            if hyperparams:
                f.write("### Hyperparameters\n\n")
                f.write("| Parameter | Value |\n")
                f.write("|-----------|-------|\n")

                # Group hyperparameters
                training_params = ['max_steps', 'learning_rate', 'global_batch_size', 'batch_size',
                                   'gradient_accumulation_steps', 'weight_decay', 'warmup_ratio',
                                   'lr_scheduler_type', 'optim']
                compute_params = ['num_gpus', 'fp16', 'bf16', 'gradient_checkpointing']
                schedule_params = ['save_steps', 'logging_steps', 'eval_steps']

                for k in training_params:
                    if k in hyperparams:
                        f.write(f"| {k.replace('_', ' ').title()} | {hyperparams[k]} |\n")

                for k in compute_params:
                    if k in hyperparams:
                        f.write(f"| {k.replace('_', ' ').title()} | {hyperparams[k]} |\n")

                for k in schedule_params:
                    if k in hyperparams:
                        f.write(f"| {k.replace('_', ' ').title()} | {hyperparams[k]} |\n")

                # Other params
                for k, v in hyperparams.items():
                    if k not in training_params + compute_params + schedule_params:
                        f.write(f"| {k.replace('_', ' ').title()} | {v} |\n")

                f.write("\n")

        # Training Loss Section
        if loss_steps and losses:
            f.write("## Training Loss\n\n")
            f.write(f"- **Initial Loss**: {losses[0]:.4f}\n")
            f.write(f"- **Final Loss**: {losses[-1]:.4f}\n")
            f.write(f"- **Minimum Loss**: {min(losses):.4f} (step {loss_steps[losses.index(min(losses))]})\n")
            f.write(f"- **Total Steps Logged**: {len(losses)}\n\n")
            f.write("![Training Loss](training_loss.png)\n\n")

        # Evaluation Metrics Section
        f.write("## Evaluation Metrics\n\n")
        f.write("### Metrics Over Training\n\n")

        f.write("| Step | MSE | MAE | Success@0.05 | Success@0.10 | Success@0.20 | Base MAE | Left MAE | Right MAE |\n")
        f.write("|------|-----|-----|--------------|--------------|--------------|----------|----------|----------|\n")

        for _, row in df.iterrows():
            f.write(f"| {int(row['Step'])} | {row['MSE']:.4f} | {row['MAE']:.4f} | ")
            f.write(f"{row['Success@0.05']:.1f}% | {row['Success@0.10']:.1f}% | {row['Success@0.20']:.1f}% | ")
            f.write(f"{row['Base MAE']:.4f} | {row['Left Arm MAE']:.4f} | {row['Right Arm MAE']:.4f} |\n")

        f.write("\n### Best Checkpoints\n\n")

        # Find best checkpoints
        best_mse_idx = df['MSE'].idxmin()
        best_mae_idx = df['MAE'].idxmin()
        best_success_idx = df['Success@0.10'].idxmax()

        f.write(f"- **Lowest MSE**: Step {int(df.loc[best_mse_idx, 'Step'])} (MSE = {df.loc[best_mse_idx, 'MSE']:.6f})\n")
        f.write(f"- **Lowest MAE**: Step {int(df.loc[best_mae_idx, 'Step'])} (MAE = {df.loc[best_mae_idx, 'MAE']:.6f})\n")
        f.write(f"- **Highest Success@0.10**: Step {int(df.loc[best_success_idx, 'Step'])} ({df.loc[best_success_idx, 'Success@0.10']:.1f}%)\n\n")

        # Plots Section
        f.write("## Plots\n\n")
        f.write("### Training Overview\n")
        f.write("![Training Overview](training_overview.png)\n\n")

        if loss_steps and losses:
            f.write("### Training Loss Curve\n")
            f.write("![Training Loss](training_loss.png)\n\n")

        f.write("### Evaluation Loss Curves\n")
        f.write("![Eval Loss Curves](eval_loss_curves.png)\n\n")

        f.write("### Success Rates\n")
        f.write("![Success Rates](success_rates.png)\n\n")

        f.write("### Accuracy Rates\n")
        f.write("![Accuracy Rates](accuracy_rates.png)\n\n")

        f.write("### Per-Component MAE\n")
        f.write("![Component MAE](component_mae.png)\n\n")

        # Files Section
        f.write("## Generated Files\n\n")
        f.write("| File | Description |\n")
        f.write("|------|-------------|\n")
        f.write("| `training_report.md` | This comprehensive report |\n")
        f.write("| `checkpoint_summary.csv` | Raw metrics data |\n")
        f.write("| `training_overview.png` | Combined 2x2 metrics overview |\n")
        if loss_steps and losses:
            f.write("| `training_loss.png` | Training loss curve |\n")
        f.write("| `eval_loss_curves.png` | MSE and MAE over checkpoints |\n")
        f.write("| `success_rates.png` | Success rates at thresholds |\n")
        f.write("| `accuracy_rates.png` | Accuracy rates at thresholds |\n")
        f.write("| `component_mae.png` | Per-component MAE breakdown |\n")

    print(f"  Saved: training_report.md")

    # Print best results
    print(f"\n  Best Results:")
    print(f"    Lowest MSE: Step {int(df.loc[best_mse_idx, 'Step'])} ({df.loc[best_mse_idx, 'MSE']:.6f})")
    print(f"    Lowest MAE: Step {int(df.loc[best_mae_idx, 'Step'])} ({df.loc[best_mae_idx, 'MAE']:.6f})")
    print(f"    Best Success@0.10: Step {int(df.loc[best_success_idx, 'Step'])} ({df.loc[best_success_idx, 'Success@0.10']:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Plot checkpoint evaluation results with training details")
    parser.add_argument(
        "--eval_dir",
        type=str,
        default="./eval_results/ball2_groot",
        help="Directory containing step_* evaluation folders",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Directory containing training checkpoints (for config and loss curves)",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Dataset name for plot titles (default: inferred from path)",
    )
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)

    # Infer checkpoint_dir if not provided
    if args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir)
    else:
        # Try to infer from eval_dir name
        dataset_name_part = eval_dir.name  # e.g., "ball2_groot"
        checkpoint_dir = Path(f"./outputs/{dataset_name_part}")
        if not checkpoint_dir.exists():
            checkpoint_dir = None

    # Infer dataset name from path
    if args.dataset_name:
        dataset_name = args.dataset_name
    else:
        dataset_name = eval_dir.name.replace('_', ' ').title()

    print(f"=" * 60)
    print(f"Training Analysis: {dataset_name}")
    print(f"=" * 60)
    print(f"Eval directory: {eval_dir}")
    if checkpoint_dir:
        print(f"Checkpoint directory: {checkpoint_dir}")
    print()

    if not eval_dir.exists():
        print(f"ERROR: Directory not found: {eval_dir}")
        return

    # Load training configuration
    config = {}
    trainer_state = {}
    hyperparams = {}
    model_info = {}
    loss_steps = []
    losses = []

    if checkpoint_dir and checkpoint_dir.exists():
        print("Loading training configuration...")
        config = load_training_config(checkpoint_dir)
        trainer_state = load_trainer_state(checkpoint_dir)

        if config:
            hyperparams = extract_hyperparameters(config)
            model_info = estimate_model_params(config)

            print(f"\n  Hyperparameters found: {len(hyperparams)}")
            for k, v in list(hyperparams.items())[:5]:
                print(f"    {k}: {v}")
            if len(hyperparams) > 5:
                print(f"    ... and {len(hyperparams) - 5} more")

        if trainer_state:
            loss_steps, losses = extract_training_losses(trainer_state)
            print(f"\n  Training loss entries: {len(losses)}")
            if losses:
                print(f"    Initial loss: {losses[0]:.4f}")
                print(f"    Final loss: {losses[-1]:.4f}")
                print(f"    Min loss: {min(losses):.4f}")
        print()

    # Find evaluation results
    print("Finding checkpoint evaluation results...")
    results = find_checkpoint_results(eval_dir)

    if not results:
        print("ERROR: No valid evaluation results found!")
        return

    print(f"\nFound {len(results)} checkpoint evaluations")
    print(f"Steps: {sorted(results.keys())}")
    print()

    # Generate plots
    print("Generating plots...")

    # Plot training loss if available
    if loss_steps and losses:
        plot_training_loss(loss_steps, losses, eval_dir, dataset_name)

    # Plot evaluation metrics
    plot_training_curves(results, eval_dir, dataset_name)

    # Create summary tables and report
    create_summary_table(results, eval_dir, dataset_name, hyperparams, model_info, loss_steps, losses)

    print()
    print(f"=" * 60)
    print("COMPLETE")
    print(f"=" * 60)
    print(f"Results saved to: {eval_dir}")
    print(f"\nView the full report: {eval_dir}/training_report.md")


if __name__ == "__main__":
    main()
