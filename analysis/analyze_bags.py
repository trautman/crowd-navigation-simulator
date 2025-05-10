#!/usr/bin/env python3
# analysis/analyze_bags.py

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def load_metrics(data_dir):
    """
    Load density metrics files from data_dir.
    Returns a list of density lists (one per trial).
    """
    files = sorted([
        f for f in os.listdir(data_dir)
        if f.startswith('density_') and f.endswith('.txt')
    ])
    metrics = []
    for fname in files:
        path = os.path.join(data_dir, fname)
        with open(path, 'r') as f:
            densities = [float(line.strip()) for line in f if line.strip()]
            metrics.append(densities)
    return metrics

def compute_stats(metrics):
    """
    Given a list of density lists, compute per-trial mean and std.
    """
    means = np.array([np.mean(m) for m in metrics])
    stds  = np.array([np.std(m)  for m in metrics])
    return means, stds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-dir',
        default=os.path.join('..','data','density'),
        help="Directory containing metrics_*.txt files"
    )
    args = parser.parse_args()
    data_dir = args.data_dir

    if not os.path.isdir(data_dir):
        print(f"Error: directory '{data_dir}' not found.")
        return

    metrics = load_metrics(data_dir)
    if not metrics:
        print(f"No metrics files found in '{data_dir}'.")
        return

    means, stds = compute_stats(metrics)
    num_trials = len(means)

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Histogram with side-by-side count and percent
    bin_width = 0.05
    bins = np.arange(0, means.max() + bin_width, bin_width)
    counts, edges, patches = axes[0].hist(
        means, bins=bins, edgecolor='black', alpha=0.7)

    # Ensure y-axis is integer ticks and add headroom
    axes[0].yaxis.set_major_locator(MaxNLocator(integer=True))
    max_count = counts.max()
    axes[0].set_ylim(0, max_count * 1.15)

    # Annotate each bar with "count, percent"
    for count, patch in zip(counts, patches):
        if count > 0:
            x = patch.get_x() + patch.get_width() / 2
            pct = (count / num_trials) * 100
            axes[0].text(
                x, count,
                f"{int(count)}, {pct:.0f}%",
                ha='center', va='bottom',
                fontsize=10
            )

    axes[0].set_xlabel('Mean Trial Density (ped/m²)')
    axes[0].set_ylabel('Number of Trials')
    axes[0].set_title('Histogram of Mean Densities')

    # Right: Mean ± std per trial (sorted)
    idx = np.argsort(means)
    sorted_means = means[idx]
    sorted_stds  = stds[idx]
    trials = np.arange(1, num_trials + 1)
    axes[1].errorbar(
        trials, sorted_means, yerr=sorted_stds,
        fmt='-o', ecolor='gray', capsize=5)
    axes[1].set_xlabel('Trial (sorted by mean density)')
    axes[1].set_ylabel('Mean Density (ped/m²)')
    axes[1].set_title('Mean ± Std per Trial')
    axes[1].yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
