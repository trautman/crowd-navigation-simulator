import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.stats import linregress

def load_metric(data_dir, prefix):
    """
    Load metric files from data_dir matching prefix, return list of lists per trial.
    """
    if not os.path.isdir(data_dir):
        print(f"WARNING: directory '{data_dir}' not found.")
        return []
    files = sorted([
        f for f in os.listdir(data_dir)
        if f.startswith(prefix) and f.endswith('.txt')
    ])
    metrics = []
    for fname in files:
        with open(os.path.join(data_dir, fname), 'r') as f:
            vals = [float(line.strip()) for line in f if line.strip()]
            metrics.append(vals)
    return metrics

def compute_stats(metrics):
    """
    Compute per-trial mean and std from list-of-lists.
    """
    if len(metrics) == 0:
        return np.array([]), np.array([])
    means = np.array([np.mean(m) for m in metrics])
    stds  = np.array([np.std(m)  for m in metrics])
    return means, stds

def main():
    parser = argparse.ArgumentParser(
        description='Analyze trial metrics versus crowd density.'
    )
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--scatter', action='store_true', help='Show scatter plots of selected metrics.')
    mode_group.add_argument('--binned', action='store_true', help='Show binned mean+95% CI plots of selected metrics.')
    parser.add_argument('--data-dir', dest='data_root', required=True,
                        help='Root data folder with subfolders for each metric.')
    # metric selection flags
    parser.add_argument('--all',                       action='store_true', help='Select all metrics.')
    parser.add_argument('--density',                   action='store_true', help='Plot density distribution and error bars.')
    parser.add_argument('--average_safety_distance',   action='store_true', help='Plot average safety distance vs density.')
    parser.add_argument('--min_safety_distance',       action='store_true', help='Plot minimum safety distance vs density.')
    parser.add_argument('--translational_velocity',    action='store_true', help='Plot average translational velocity vs density.')
    parser.add_argument('--path_length',               action='store_true', help='Plot total path length vs density.')
    parser.add_argument('--efficiency',                action='store_true', help='Plot normalized path efficiency vs density.')
    parser.add_argument('--time_not_moving',           action='store_true', help='Plot total time not moving vs density.')
    parser.add_argument('--travel_time',               action='store_true', help='Plot total travel time vs density.')
    args = parser.parse_args()

    # if --all, enable all metrics
    metric_flags = ['density','average_safety_distance','min_safety_distance',
                    'translational_velocity','path_length','efficiency',
                    'time_not_moving','travel_time']
    if args.all:
        for mf in metric_flags:
            setattr(args, mf, True)
    if not args.all and not any(getattr(args, mf) for mf in metric_flags):
        parser.error('No metric specified; use --all or at least one metric flag.')

    data_root = args.data_root
    # define metric directories
    dirs = {
        'density':        os.path.join(data_root, 'density'),
        'safety':         os.path.join(data_root, 'safety_distances'),
        'trans_vel':      os.path.join(data_root, 'translational_velocity'),
        'path_len':       os.path.join(data_root, 'path_length'),
        'travel_time':    os.path.join(data_root, 'travel_time'),
        'stop_time':      os.path.join(data_root, 'time_not_moving')
    }
    # load density
    density_metrics = load_metric(dirs['density'], 'density_trial_')
    density_means, density_stds = compute_stats(density_metrics)
    # density plots
    if args.density:
        bin_width = 0.05
        bins = np.arange(0, density_means.max()*1.05 + bin_width, bin_width)
        fig, axes = plt.subplots(1,2,figsize=(12,5))
        counts, edges, patches = axes[0].hist(density_means, bins=bins, edgecolor='black', alpha=0.7)
        axes[0].set_xlim(0, density_means.max()*1.05)
        axes[0].set_ylim(0, counts.max()*1.15)
        num_trials = len(density_means)
        y_off = 0.01 * axes[0].get_ylim()[1]
        for count, patch in zip(counts, patches):
            if count > 0:
                x = patch.get_x() + patch.get_width()/2
                pct = (count/num_trials)*100
                axes[0].text(x, count+y_off, f"{pct:.0f}%", ha='center', va='bottom')
                axes[0].text(x, count-y_off, f"{int(count)}", ha='center', va='top')
        axes[0].set_xlabel('Mean Density (ped/m²)')
        axes[0].set_ylabel('Number of Trials')
        axes[0].set_title('Density Distribution')
        idx = np.argsort(density_means)
        axes[1].errorbar(np.arange(1,len(density_means)+1), density_means[idx], yerr=density_stds[idx], fmt='-o', ecolor='gray', capsize=5)
        axes[1].set_xlim(0,len(density_means)+1)
        axes[1].set_ylim(0, density_means.max()*1.05)
        axes[1].set_xlabel('Trial (sorted)')
        axes[1].set_ylabel('Mean Density (ped/m²)')
        axes[1].set_title('Density Mean ± Std')
        plt.tight_layout(); plt.show()

    # prepare other metrics
    safety_metrics = load_metric(dirs['safety'], 'safety_distances_trial_')
    safety_means   = np.array([np.mean(m) for m in safety_metrics])
    safety_mins    = np.array([np.min(m)  for m in safety_metrics])
    tv_metrics     = load_metric(dirs['trans_vel'], 'translational_velocity_trial_')
    tv_means       = np.array([np.mean(m) for m in tv_metrics])
    pl_metrics     = load_metric(dirs['path_len'], 'path_length_trial_')
    pl_totals      = np.array([np.sum(m)  for m in pl_metrics])
    efficiencies   = 10.0 / pl_totals
    tn_metrics     = load_metric(dirs['stop_time'], 'time_not_moving_trial_')
    tn_totals      = np.array([np.sum(m)  for m in tn_metrics])
    tt_metrics     = load_metric(dirs['travel_time'], 'travel_time_trial_')
    tt_totals      = np.array([np.sum(m)  for m in tt_metrics])

    metrics_map = {
        'average_safety_distance': (safety_means, 'Avg Safety Distance (m)',        'Average Safety Distance'),
        'min_safety_distance':     (safety_mins,  'Min Safety Distance (m)',        'Minimum Safety Distance'),
        'translational_velocity':  (tv_means,     'Avg Translational Velocity (m/s)', 'Avg Translational Velocity'),
        'path_length':             (pl_totals,    'Total Path Length (m)',          'Total Path Length'),
        'efficiency':              (efficiencies,'Normalized Efficiency (10m/path_length)','Normalized Efficiency'),
        'time_not_moving':         (tn_totals,    'Total Time Not Moving (s)',      'Time Not Moving'),
        'travel_time':             (tt_totals,    'Total Travel Time (s)',          'Travel Time')
    }
    bin_width = 0.05
    xlim_low, xlim_high = 0, density_means.max()*1.05

    for flag, (values, ylabel, name) in metrics_map.items():
        if getattr(args, flag):
            plt.figure(figsize=(6,6))
            if args.scatter:
                plt.scatter(density_means, values, s=50, edgecolor='black', label='Data')
            else:
                edges = np.arange(0, density_means.max()*1.05 + bin_width, bin_width)
                centers, m_means, m_cis = [], [], []
                for i in range(len(edges)-1):
                    mask = (density_means >= edges[i]) & (density_means < edges[i+1])
                    if np.any(mask):
                        data  = values[mask]
                        n     = data.size
                        mu    = data.mean()
                        sigma = data.std(ddof=0)
                        ci95  = 1.96 * sigma / np.sqrt(n)
                        centers.append((edges[i]+edges[i+1])/2)
                        m_means.append(mu)
                        m_cis.append(ci95)
                centers = np.array(centers)
                m_means = np.array(m_means)
                m_cis   = np.array(m_cis)
                plt.errorbar(centers, m_means, yerr=m_cis, fmt='-o', capsize=5, label='Mean ± 95% CI')
                # linear fit on the binned means
                slope, intercept, r_value, p_value, _ = linregress(centers, m_means)
                fit_line = slope * centers + intercept
                plt.plot(centers, fit_line, 'r--', label=f'Fit: y={slope:.3f}x+{intercept:.3f}\n' +
                         f'r={r_value:.3f}, p={p_value:.3f}')
            plt.xlabel('Mean Density (ped/m²)')
            plt.ylabel(ylabel)
            title = f'{name} vs Density'
            if args.binned: title += ' (binned)'
            plt.title(title)
            plt.xlim(xlim_low, xlim_high)
            plt.ylim(0, values.max()*1.05)
            plt.legend(loc='best')
            plt.show()

if __name__ == '__main__':
    main()
