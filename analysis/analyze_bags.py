import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def load_density(data_dir):
    """
    Load density files from data_dir.
    Returns a list of lists of density values, one per trial.
    """
    files = sorted([
        f for f in os.listdir(data_dir)
        if f.startswith('density_trial_') and f.endswith('.txt')
    ])
    metrics = []
    for fname in files:
        path = os.path.join(data_dir, fname)
        with open(path, 'r') as f:
            vals = [float(line.strip()) for line in f if line.strip()]
            metrics.append(vals)
    return metrics


def load_safety(data_dir):
    """
    Load safety distance files from data_dir.
    Returns a list of lists of distances, one per trial.
    """
    files = sorted([
        f for f in os.listdir(data_dir)
        if f.startswith('safety_distances_trial_') and f.endswith('.txt')
    ])
    metrics = []
    for fname in files:
        path = os.path.join(data_dir, fname)
        with open(path, 'r') as f:
            dists = [float(line.strip()) for line in f if line.strip()]
            metrics.append(dists)
    return metrics


def compute_stats(metrics):
    """
    Given a list of lists (one per trial), compute per-trial mean and std.
    """
    means = np.array([np.mean(m) for m in metrics])
    stds  = np.array([np.std(m)  for m in metrics])
    return means, stds


def main():
    parser = argparse.ArgumentParser(
        description='Analyze density and safety-distance metrics from trials.'
    )
    parser.add_argument(
        '--data-dir', dest='data_root', required=True,
        help='Path to the root data folder containing "density/" and "safety_distances/".'
    )
    args = parser.parse_args()
    data_root    = args.data_root
    density_dir  = os.path.join(data_root, 'density')
    safety_dir   = os.path.join(data_root, 'safety_distances')

    # Load and analyze density
    print(f"→ Looking for density files in: {density_dir}")
    print("   Found:", sorted(os.listdir(density_dir)))
    density_metrics = load_density(density_dir)
    if not density_metrics:
        print(f"ERROR: No density files found in '{density_dir}'")
        return
    density_means, density_stds = compute_stats(density_metrics)

    # Plot histogram and errorbar for density
    num_trials = len(density_means)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram of mean densities
    bin_width = 0.05
    bins = np.arange(0, density_means.max() + bin_width, bin_width)
    counts, edges, patches = axes[0].hist(
        density_means, bins=bins, edgecolor='black', alpha=0.7
    )
    axes[0].yaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0].set_ylim(0, counts.max() * 1.15)
    for count, patch in zip(counts, patches):
        if count > 0:
            x = patch.get_x() + patch.get_width() / 2
            pct = (count / num_trials) * 100
            axes[0].text(
                x, count, f"{int(count)}, {pct:.0f}%",
                ha='center', va='bottom', fontsize=10
            )
    axes[0].set_xlabel('Mean Trial Density (ped/m²)')
    axes[0].set_ylabel('Number of Trials')
    axes[0].set_title('Histogram of Mean Densities')

    # Mean ± std per trial (sorted)
    idx = np.argsort(density_means)
    sorted_means = density_means[idx]
    sorted_stds  = density_stds[idx]
    trials = np.arange(1, num_trials + 1)
    axes[1].errorbar(
        trials, sorted_means, yerr=sorted_stds,
        fmt='-o', ecolor='gray', capsize=5
    )
    axes[1].set_xlabel('Trial (sorted by mean density)')
    axes[1].set_ylabel('Mean Density (ped/m²)')
    axes[1].set_title('Mean ± Std per Trial')
    axes[1].yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.show()

    # Load and analyze safety distances
    print(f"→ Looking for safety files in: {safety_dir}")
    print("   Found:", sorted(os.listdir(safety_dir)))
    safety_metrics = load_safety(safety_dir)
    if not safety_metrics:
        print(f"ERROR: No safety-distance files found in '{safety_dir}'")
        return
    safety_means = np.array([np.mean(m) for m in safety_metrics])
    safety_mins  = np.array([np.min(m)  for m in safety_metrics])

    # Scatter: mean density vs mean safety distance
    plt.figure(figsize=(6, 6))
    plt.scatter(
        density_means, safety_means,
        s=50, edgecolor='black'
    )
    plt.xlabel('Mean Trial Density (ped/m²)')
    plt.ylabel('Mean Safety Distance (m)')
    plt.title('Average Safety Distance vs Mean Density')
    plt.grid(True)
    plt.show()

    # Scatter: mean density vs minimum safety distance
    plt.figure(figsize=(6, 6))
    plt.scatter(
        density_means, safety_mins,
        s=50, edgecolor='black'
    )
    plt.xlabel('Mean Trial Density (ped/m²)')
    plt.ylabel('Min Safety Distance (m)')
    plt.title('Minimum Safety Distance vs Mean Density')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
