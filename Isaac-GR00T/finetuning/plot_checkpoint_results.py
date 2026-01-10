#!/usr/bin/env python3
"""
Plot evaluation results across checkpoints.

This script reads evaluation results from multiple checkpoint evaluations
and generates comparison plots showing training progress.

Usage:
    python finetuning/plot_checkpoint_results.py --eval_dir ./eval_results/ball2_groot
    python finetuning/plot_checkpoint_results.py --eval_dir ./eval_results/plug_stacking
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
            # Look for Accuracy line (second occurrence of threshold)
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
    plt.savefig(output_dir / 'loss_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: loss_curves.png")

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

    # 6. Create summary table
    create_summary_table(results, output_dir, dataset_name)


def create_summary_table(results: dict[int, dict], output_dir: Path, dataset_name: str):
    """Create a summary CSV and markdown table."""
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

    # Save markdown table
    md_path = output_dir / 'checkpoint_summary.md'
    with open(md_path, 'w') as f:
        f.write(f"# {dataset_name} Checkpoint Evaluation Summary\n\n")
        f.write("## Metrics Over Training\n\n")

        # Format table
        f.write("| Step | MSE | MAE | Success@0.05 | Success@0.10 | Success@0.20 | Base MAE | Left MAE | Right MAE |\n")
        f.write("|------|-----|-----|--------------|--------------|--------------|----------|----------|----------|\n")

        for _, row in df.iterrows():
            f.write(f"| {int(row['Step'])} | {row['MSE']:.4f} | {row['MAE']:.4f} | ")
            f.write(f"{row['Success@0.05']:.1f}% | {row['Success@0.10']:.1f}% | {row['Success@0.20']:.1f}% | ")
            f.write(f"{row['Base MAE']:.4f} | {row['Left Arm MAE']:.4f} | {row['Right Arm MAE']:.4f} |\n")

        f.write("\n## Best Checkpoints\n\n")

        # Find best checkpoints
        best_mse_idx = df['MSE'].idxmin()
        best_mae_idx = df['MAE'].idxmin()
        best_success_idx = df['Success@0.10'].idxmax()

        f.write(f"- **Lowest MSE**: Step {int(df.loc[best_mse_idx, 'Step'])} (MSE = {df.loc[best_mse_idx, 'MSE']:.6f})\n")
        f.write(f"- **Lowest MAE**: Step {int(df.loc[best_mae_idx, 'Step'])} (MAE = {df.loc[best_mae_idx, 'MAE']:.6f})\n")
        f.write(f"- **Highest Success@0.10**: Step {int(df.loc[best_success_idx, 'Step'])} ({df.loc[best_success_idx, 'Success@0.10']:.1f}%)\n")

        f.write("\n## Plots\n\n")
        f.write("- `training_overview.png` - Combined overview of all metrics\n")
        f.write("- `loss_curves.png` - MSE and MAE over training\n")
        f.write("- `success_rates.png` - Success rates at different thresholds\n")
        f.write("- `accuracy_rates.png` - Accuracy rates at different thresholds\n")
        f.write("- `component_mae.png` - Per-component MAE breakdown\n")

    print(f"  Saved: checkpoint_summary.md")

    # Print best results
    print(f"\n  Best Results:")
    print(f"    Lowest MSE: Step {int(df.loc[best_mse_idx, 'Step'])} ({df.loc[best_mse_idx, 'MSE']:.6f})")
    print(f"    Lowest MAE: Step {int(df.loc[best_mae_idx, 'Step'])} ({df.loc[best_mae_idx, 'MAE']:.6f})")
    print(f"    Best Success@0.10: Step {int(df.loc[best_success_idx, 'Step'])} ({df.loc[best_success_idx, 'Success@0.10']:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Plot checkpoint evaluation results")
    parser.add_argument(
        "--eval_dir",
        type=str,
        default="./eval_results/ball2_groot",
        help="Directory containing step_* evaluation folders",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Dataset name for plot titles (default: inferred from path)",
    )
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)

    # Infer dataset name from path
    if args.dataset_name:
        dataset_name = args.dataset_name
    else:
        dataset_name = eval_dir.name.replace('_', ' ').title()

    print(f"=" * 60)
    print(f"Plotting Checkpoint Results: {dataset_name}")
    print(f"=" * 60)
    print(f"Eval directory: {eval_dir}")
    print()

    if not eval_dir.exists():
        print(f"ERROR: Directory not found: {eval_dir}")
        return

    print("Finding checkpoint results...")
    results = find_checkpoint_results(eval_dir)

    if not results:
        print("ERROR: No valid results found!")
        return

    print(f"\nFound {len(results)} checkpoint evaluations")
    print(f"Steps: {sorted(results.keys())}")
    print()

    print("Generating plots...")
    plot_training_curves(results, eval_dir, dataset_name)

    print()
    print(f"=" * 60)
    print("COMPLETE")
    print(f"=" * 60)
    print(f"Results saved to: {eval_dir}")


if __name__ == "__main__":
    main()
