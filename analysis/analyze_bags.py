import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def load_metric(data_dir, prefix):
    """
    Load metric files from data_dir matching prefix, return list of lists per trial.
    """
    files = sorted([
        f for f in os.listdir(data_dir)
        if f.startswith(prefix) and f.endswith('.txt')
    ])
    metrics = []
    for fname in files:
        path = os.path.join(data_dir, fname)
        with open(path, 'r') as f:
            vals = [float(line.strip()) for line in f if line.strip()]
            metrics.append(vals)
    return metrics

def compute_stats(metrics):
    """
    Compute per-trial mean and std from list-of-lists.
    """
    means = np.array([np.mean(m) for m in metrics])
    stds  = np.array([np.std(m)  for m in metrics])
    return means, stds

def main():
    parser = argparse.ArgumentParser(
        description='Analyze trial metrics versus crowd density.'
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--scatter', action='store_true', help='Show scatter plots of each metric.')
    group.add_argument('--binned_with_means', action='store_true', help='Show binned mean+std plots of each metric.')
    parser.add_argument(
        '--data-dir', dest='data_root', required=True,
        help='Root data folder with subfolders: density, safety_distances, translational_velocity, path_length, travel_time, time_not_moving.'
    )
    args = parser.parse_args()
    data_root = args.data_root
    # Subfolder paths
    density_dir     = os.path.join(data_root, 'density')
    safety_dir      = os.path.join(data_root, 'safety_distances')
    trans_vel_dir   = os.path.join(data_root, 'translational_velocity')
    path_len_dir    = os.path.join(data_root, 'path_length')
    travel_time_dir = os.path.join(data_root, 'travel_time')
    stop_time_dir   = os.path.join(data_root, 'time_not_moving')

    # Load density metrics
    density_metrics = load_metric(density_dir, 'density_trial_')
    if not density_metrics:
        print(f"ERROR: No density files in '{density_dir}'")
        return
    density_means, density_stds = compute_stats(density_metrics)

    # Plot density distribution and error bars
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # Histogram
    bin_width = 0.05
    bins = np.arange(0, density_means.max() * 1.05 + bin_width, bin_width)
    counts, edges, patches = axes[0].hist(
        density_means, bins=bins, edgecolor='black', alpha=0.7
    )
    axes[0].yaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0].set_xlim(0, density_means.max() * 1.05)
    axes[0].set_ylim(0, counts.max() * 1.15)
    axes[0].set_xlabel('Mean Density (ped/m²)')
    axes[0].set_ylabel('Number of Trials')
    axes[0].set_title('Density Distribution')
    # Errorbar per trial
    idx = np.argsort(density_means)
    axes[1].errorbar(
        np.arange(1, len(density_means)+1), density_means[idx], yerr=density_stds[idx],
        fmt='-o', ecolor='gray', capsize=5
    )
    axes[1].set_xlim(0, len(density_means)+1)
    axes[1].set_ylim(0, density_means.max() * 1.05)
    axes[1].set_xlabel('Trial (sorted)')
    axes[1].set_ylabel('Mean Density (ped/m²)')
    axes[1].set_title('Density Mean ± Std')
    plt.tight_layout()
    plt.show()

    # Prepare other metrics
    safety_metrics = load_metric(safety_dir, 'safety_distances_trial_')
    safety_means = np.array([np.mean(m) for m in safety_metrics])
    safety_mins  = np.array([np.min(m)  for m in safety_metrics])
    tv_metrics   = load_metric(trans_vel_dir, 'translational_velocity_trial_')
    tv_means     = np.array([np.mean(m) for m in tv_metrics])
    pl_metrics   = load_metric(path_len_dir, 'path_length_trial_')
    pl_totals    = np.array([np.sum(m)  for m in pl_metrics])
    efficiencies = 10.0 / pl_totals
    tn_metrics   = load_metric(stop_time_dir, 'time_not_moving_trial_')
    tn_totals    = np.array([np.sum(m)  for m in tn_metrics])
    tt_metrics   = load_metric(travel_time_dir, 'travel_time_trial_')
    tt_totals    = np.array([np.sum(m)  for m in tt_metrics])

    # List of (name, values, ylabel)
    metrics_info = [
        ('Avg Safety Distance', safety_means, 'Avg Safety Distance (m)'),
        ('Min Safety Distance', safety_mins,  'Min Safety Distance (m)'),
        ('Avg Translational Velocity', tv_means, 'Avg Translational Velocity (m/s)'),
        ('Total Path Length', pl_totals,      'Total Path Length (m)'),
        ('Efficiency', efficiencies,          'Normalized Efficiency (10m/path_length)'),
        ('Time Not Moving', tn_totals,        'Total Time Not Moving (s)'),
        ('Travel Time', tt_totals,            'Total Travel Time (s)')
    ]

    # Common x-limits
    xlim_low, xlim_high = 0, density_means.max() * 1.05

    # Plot each metric
    for name, values, ylabel in metrics_info:
        plt.figure(figsize=(6,6))
        if args.scatter:
            plt.scatter(density_means, values, s=50, edgecolor='black')
        else:
            # bin and compute means+stds
            edges = np.arange(0, density_means.max() * 1.05 + bin_width, bin_width)
            centers, means, stds = [], [], []
            for i in range(len(edges)-1):
                mask = (density_means >= edges[i]) & (density_means < edges[i+1])
                if np.any(mask):
                    centers.append((edges[i]+edges[i+1])/2)
                    means.append(np.mean(values[mask]))
                    stds.append(np.std(values[mask]))
            centers = np.array(centers)
            means   = np.array(means)
            stds    = np.array(stds)
            plt.errorbar(centers, means, yerr=stds, fmt='-o', capsize=5)
        plt.xlabel('Mean Density (ped/m²)')
        plt.ylabel(ylabel)
        plt.title(f'{name} vs Density' + (' (binned)' if args.binned_with_means else ''))
        plt.xlim(xlim_low, xlim_high)
        plt.ylim(0, values.max() * 1.05)
        plt.show()

if __name__ == '__main__':
    main()
