import os
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.gridspec as gridspec
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
        path = os.path.join(data_dir, fname)
        with open(path, 'r') as f:
            vals = [float(line.strip()) for line in f if line.strip()]
            metrics.append(vals)
    return metrics

def compute_stats(metrics):
    """
    Compute per-trial mean and std from list-of-lists.
    """
    if len(metrics) == 0:
        return np.array([]), np.array([])
    # means = np.array([np.mean(m) for m in metrics])
    # stds  = np.array([np.std(m)  for m in metrics])
    means = np.array([np.mean(m) for m in metrics])
    # use sample‐std (ddof=1) to avoid underestimating the spread
    stds  = np.array([np.std(m, ddof=1) if len(m)>1 else 0.0 for m in metrics])
    return means, stds

def main():
    parser = argparse.ArgumentParser(
        description='Analyze trial metrics versus crowd density.'
    )
    # mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group = parser.add_mutually_exclusive_group(required=False)
    mode_group.add_argument('--scatter', action='store_true', help='Show scatter plots of selected metrics.')
    mode_group.add_argument('--binned', action='store_true', help='Show binned mean+95% CI plots of selected metrics.')
    parser.add_argument('--data-dir', dest='data_root', required=True,
                        help='Root data folder with subfolders for each metric.')
    # metric selection flags
    parser.add_argument('--all',                     action='store_true', help='Select all metrics.')
    parser.add_argument('--density',                 action='store_true', help='Plot density distribution and error bars.')
    parser.add_argument('--average_safety_distance', action='store_true', help='Plot average safety distance vs density.')
    parser.add_argument('--min_safety_distance',     action='store_true', help='Plot minimum safety distance vs density.')
    parser.add_argument('--translational_velocity',  action='store_true', help='Plot average translational velocity vs density.')
    parser.add_argument('--path_length',             action='store_true', help='Plot total path length vs density.')
    parser.add_argument('--efficiency',              action='store_true', help='Plot normalized path efficiency vs density.')
    parser.add_argument('--time_not_moving',         action='store_true', help='Plot total time not moving vs density.')
    parser.add_argument('--travel_time',             action='store_true', help='Plot total travel time vs density.')
    parser.add_argument('--write_to_yaml',           action='store_true', help='Write YAML files for binned metrics.')
    parser.add_argument('--site',                    help='Site name to embed in YAML and filename (e.g. Arcade)')
    parser.add_argument('--baseline',                help='Baseline name to embed in YAML and filename (e.g. BRNE)')
    parser.add_argument('--state',                   help='State name to embed in YAML and filename (e.g. ORCA)')    
    parser.add_argument('--giant_plot',  action='store_true', help='Show all selected metrics as a grid of subplots.')

    args = parser.parse_args()

    # If the user only wants YAML, skip the scatter/binned requirement
    if not args.write_to_yaml:
        # in all other cases we must have either --scatter or --binned
        if not (args.scatter or args.binned):
            parser.error('Must specify one of --scatter, --binned, or --write_to_yaml.')
    else:
        # in YAML mode, require the three identifiers
        if not (args.site and args.baseline and args.state):
            parser.error('--write_to_yaml requires --site, --baseline, and --state.')


    # if --all, enable all metrics
    metric_flags = ['density','average_safety_distance','min_safety_distance',
                    'translational_velocity','path_length','efficiency',
                    'time_not_moving','travel_time']
    if args.all:
        for mf in metric_flags:
            setattr(args, mf, True)
    # if not args.all and not any(getattr(args, mf) for mf in metric_flags):
    #     parser.error('No metric specified; use --all or at least one metric flag.')
    if not args.all and not any(getattr(args, mf) for mf in metric_flags) and not args.write_to_yaml:
        parser.error('No metric specified; use --all, a metric flag, or --write_to_yaml.')


    data_root = args.data_root
    # create YAML output folder if needed
    # yaml_dir = os.path.join(data_root, 'yaml_files')
    # if args.write_to_yaml:
    #     os.makedirs(yaml_dir, exist_ok=True)
    yaml_dir = os.path.join(data_root, 'yaml_files')
    if args.write_to_yaml:
        os.makedirs(yaml_dir, exist_ok=True)

    # define metric directories
    dirs = {
        'density':     os.path.join(data_root, 'density'),
        'safety':      os.path.join(data_root, 'safety_distances'),
        'trans_vel':   os.path.join(data_root, 'translational_velocity'),
        'path_len':    os.path.join(data_root, 'path_length'),
        'travel_time': os.path.join(data_root, 'travel_time'),
        'stop_time':   os.path.join(data_root, 'time_not_moving')
    }
    # load density metrics
    density_metrics = load_metric(dirs['density'], 'density_trial_')
    density_means, density_stds = compute_stats(density_metrics)
    # density plots
    if args.density and not args.giant_plot:
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
        'average_safety_distance': (safety_means, 'Avg Safety Distance (m)', 'Average Safety Distance'),
        'min_safety_distance':     (safety_mins,  'Min Safety Distance (m)',     'Minimum Safety Distance'),
        'translational_velocity':  (tv_means,     'Avg Translational Velocity (m/s)', 'Avg Translational Velocity'),
        'path_length':             (pl_totals,    'Total Path Length (m)',       'Total Path Length'),
        'efficiency':              (efficiencies, 'Normalized Efficiency (10m/path_length)', 'Normalized Efficiency'),
        'time_not_moving':         (tn_totals,    'Total Time Not Moving (s)',   'Time Not Moving'),
        'travel_time':             (tt_totals,    'Total Travel Time (s)',       'Travel Time')
    }


    # ── WRITE ALL BINS TO YAML AND EXIT ──────────────────────────
    if args.write_to_yaml:
        # we only need one bin_width and density_means for all metrics
        bin_width = 0.05
        edges = np.arange(0, density_means.max()*1.05 + bin_width, bin_width)
        for flag, (values, ylabel, name) in metrics_map.items():
            centers, m_means, m_stds, m_ns = [], [], [], []
            for i in range(len(edges)-1):
                mask = (density_means >= edges[i]) & (density_means < edges[i+1])
                if mask.any():
                    d     = values[mask]
                    n     = d.size
                    mu    = d.mean()
                    # sigma = d.std(ddof=0)
                    sigma = d.std(ddof=1) if len(d)>1 else 0.0
                    # centers.append((edges[i]+edges[i+1]) / 2)
                    centers.append(edges[i])
                    m_means.append(mu)
                    m_stds.append(sigma)
                    m_ns.append(n)

            # out_file = os.path.join(yaml_dir, f"sim_arcade_BRNE_ORCA_{flag}.yaml")
            # use the user-passed identifiers in the filename
            fname    = f"sim_{args.site}_{args.baseline}_{args.state}_{flag}.yaml"
            out_file = os.path.join(yaml_dir, fname)
            with open(out_file, 'w') as f:
                # f.write("site: Arcade\n")
                # f.write("state: ORCA\n")
                # f.write("baseline: BRNE\n")
                f.write(f"site: {args.site}\n")
                f.write(f"state: {args.state}\n")
                f.write(f"baseline: {args.baseline}\n")
                f.write(f"metric: {name}\n")
                f.write("bin: [" + ", ".join(f"{b:.2f}" for b in centers) + "]\n")
                f.write("mean: ["+ ", ".join(f"{m:.3f}" for m in m_means) + "]\n")
                f.write("std: [" + ", ".join(f"{s:.3f}" for s in m_stds)   + "]\n")
                f.write("N: ["   + ", ".join(str(n) for n in m_ns)           + "]\n")
            print(f"Saved YAML → {out_file}")
        return



    if (args.binned and args.giant_plot) or (args.scatter and args.giant_plot):
        # which metrics (excluding density) to plot below the top row
        metric_keys = [k for k in metrics_map if k != 'density']
        n_metrics   = len(metric_keys)
        n_cols      = 2
        # one top row + enough rows to fit the rest in 2 columns
        n_rows      = 1 + (n_metrics + n_cols - 1)//n_cols

        fig = plt.figure(figsize=(10, 4*n_rows))
        gs  = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                                 hspace=0.5, wspace=0.3)



        # ── Top‐row density panels ───────────────────────────────────────

        # Histogram (left)
        ax0 = fig.add_subplot(gs[0, 0])
        bin_width = 0.05
        bins = np.arange(0, density_means.max()*1.05 + bin_width, bin_width)
        counts, edges, patches = ax0.hist(
            density_means, bins=bins, edgecolor='black', alpha=0.7
        )
        ax0.set_xlim(0, density_means.max()*1.05)
        ax0.set_ylim(0, counts.max()*1.15)
        num_trials = len(density_means)
        y_off = 0.01 * ax0.get_ylim()[1]
        for count, patch in zip(counts, patches):
            if count > 0:
                x = patch.get_x() + patch.get_width()/2
                pct = (count/num_trials)*100
                ax0.text(x, count+y_off, f"{pct:.0f}%", ha='center', va='bottom')
                ax0.text(x, count-y_off, f"{int(count)}", ha='center', va='top')
        ax0.set_xlabel('Mean Density (ped/m²)')
        ax0.set_ylabel('Number of Trials')
        ax0.set_title('Density Distribution')
          
        # Mean ± Std vs Trial (right)
        ax1 = fig.add_subplot(gs[0, 1])
        idx = np.argsort(density_means)
        ax1.errorbar(
            np.arange(1, num_trials+1),
            density_means[idx],
            yerr=density_stds[idx],
            fmt='-o',
            ecolor='gray',
            capsize=5
        )
        ax1.set_xlim(0, num_trials+1)
        ax1.set_ylim(0, density_means.max()*1.05)
        ax1.set_xlabel('Trial (sorted)')
        ax1.set_ylabel('Mean Density (ped/m²)')
        ax1.set_title('Density Mean ± Std')


        # ── Other metrics beneath ──────────────────────────────────────
        for idx, key in enumerate(metric_keys):
            vals   = metrics_map[key][0]
            ylabel = metrics_map[key][1]
            title  = metrics_map[key][2]
            row = 1 + idx//n_cols
            col = idx % n_cols
            ax  = fig.add_subplot(gs[row, col])

            if args.scatter:
                ax.scatter(density_means, vals, s=50, edgecolor='black')
                ax.set_ylim(0, vals.max()*1.05)
            else:
                # binned mean±CI
                edges = np.arange(0, density_means.max()*1.05 + bin_width, bin_width)
                ctrs, mus, cis = [], [], []
                for i in range(len(edges)-1):
                    mask = (density_means>=edges[i]) & (density_means<edges[i+1])
                    if mask.any():
                        d     = vals[mask]
                        mu    = d.mean()
                        # sigma = d.std(ddof=0)
                        sigma = d.std(ddof=1) if len(d)>1 else 0.0
                        ci95  = 1.96*sigma/np.sqrt(d.size)
                        ctrs.append((edges[i]+edges[i+1])/2)
                        mus.append(mu)
                        cis.append(ci95)
                ctrs = np.array(ctrs); mus = np.array(mus); cis = np.array(cis)
                ax.errorbar(ctrs, mus, yerr=cis, fmt='-o', capsize=5)
                # linear fit
                if ctrs.size > 1:
                    sl, ic, r, p, _ = linregress(ctrs, mus)
                    ax.plot(ctrs, sl*ctrs+ic, 'r--',
                            label=f'y={sl:.2f}x+{ic:.2f}, r²={r*r:.2g}, p={p:.2g}')
                ax.legend(loc='best')
                ax.set_ylim(0, mus.max()*1.05)

            ax.set_xlim(0, density_means.max()*1.05)
            ax.set_title(title + (' (scatter)' if args.scatter else ' (binned)'))
            ax.set_xlabel('Density (ped/m²)')
            ax.set_ylabel(ylabel)

        plt.tight_layout()
        plt.show()
        # sys.exit(0)
        return    



    bin_width = 0.05
    xlim_low, xlim_high = 0, density_means.max()*1.05

    for flag, (values, ylabel, name) in metrics_map.items():
        if getattr(args, flag):
            plt.figure(figsize=(6,6))
            if args.scatter:
                plt.scatter(density_means, values, s=50, edgecolor='black', label='Data')
            else:
                edges = np.arange(0, density_means.max()*1.05 + bin_width, bin_width)
                centers, m_means, m_stds, m_ns, m_cis = [], [], [], [], []
                for i in range(len(edges)-1):
                    mask = (density_means >= edges[i]) & (density_means < edges[i+1])
                    # if np.any(mask):
                    #     data  = values[mask]
                    #     n     = data.size
                    #     mu    = data.mean()
                    #     # sigma = data.std(ddof=0)
                    #     sigma = d.std(ddof=1) if len(d)>1 else 0.0
                    #     ci95  = 1.96 * sigma / np.sqrt(n)
                    #     centers.append((edges[i]+edges[i+1])/2)
                    #     m_means.append(mu)
                    #     m_stds.append(sigma)
                    #     m_ns.append(n)
                    #     m_cis.append(ci95)
                    if np.any(mask):
                        data  = values[mask]
                        n     = data.size
                        mu    = data.mean()
                        # use sample-std on 'data', not the undefined 'd'
                        sigma = data.std(ddof=1) if n>1 else 0.0
                        ci95  = 1.96 * sigma / np.sqrt(n)
                        centers.append((edges[i]+edges[i+1])/2)
                        m_means.append(mu)
                        m_stds.append(sigma)
                        m_ns.append(n)
                        m_cis.append(ci95)                    
                centers = np.array(centers)
                m_means = np.array(m_means)
                m_stds  = np.array(m_stds)
                m_ns    = np.array(m_ns)
                m_cis   = np.array(m_cis)
                plt.errorbar(centers, m_means, yerr=m_cis, fmt='-o', capsize=5, label='Mean ± 95% CI')
                slope, intercept, r_value, p_value, _ = linregress(centers, m_means)
                fit_line = slope * centers + intercept
                plt.plot(centers, fit_line, 'r--',
                         label=f'y={slope:.3f}x+{intercept:.3f}, r={r_value:.3f}, p={p_value:.3f}')
                if args.write_to_yaml:
                    out_file = os.path.join(yaml_dir, f"sim_arcade_BRNE_ORCA_{flag}.yaml")
                    with open(out_file, 'w') as f:
                        f.write("site: Arcade\n")
                        f.write("state: ORCA\n")
                        f.write("baseline: BRNE\n")
                        f.write(f"metric: {name}\n")
                        # inline lists exactly as in reference
                        f.write("mean: [" +
                                ", ".join(f"{v:.3f}" for v in m_means) +
                                "]\n")
                        f.write("std: [" +
                                ", ".join(f"{v:.3f}" for v in m_stds) +
                                "]\n")
                        f.write("N: [" +
                                ", ".join(str(int(n)) for n in m_ns) +
                                "]\n")
                        f.write("bin: [" +
                                ", ".join(f"{b:.2f}" for b in centers) +
                                "]\n")
                    print(f"Saved YAML for {name} → {out_file}")
            

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
