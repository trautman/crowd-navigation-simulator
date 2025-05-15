#!/usr/bin/env python3
import os
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(
        description="Clean out trials whose total path length exceeds a threshold."
    )
    parser.add_argument(
        "--data-dir", "-d",
        required=True,
        help="Root data folder containing subfolders: density, path_length, safety_distances, time_not_moving, translational_velocity, travel_time"
    )
    parser.add_argument(
        "--thresh", "-t",
        type=float,
        required=True,
        help="Path‐length threshold (sum of all entries in path_length_trial_{i}.txt). All trials above this will be deleted."
    )
    args = parser.parse_args()

    root = args.data_dir
    path_dir = os.path.join(root, "path_length")
    if not os.path.isdir(path_dir):
        print(f"Error: path_length directory not found at {path_dir}", file=sys.stderr)
        sys.exit(1)

    # 1) find all trials whose summed path length > threshold
    bad_trials = []
    for fname in os.listdir(path_dir):
        if not fname.startswith("path_length_trial_") or not fname.endswith(".txt"):
            continue
        trial_str = fname[len("path_length_trial_"):-4]
        try:
            trial = int(trial_str)
        except ValueError:
            continue

        # load and sum
        vals = []
        with open(os.path.join(path_dir, fname), "r") as f:
            for line in f:
                line = line.strip()
                if not line: 
                    continue
                try:
                    vals.append(float(line))
                except ValueError:
                    continue
        total = sum(vals)
        if total > args.thresh:
            bad_trials.append(trial)

    bad_trials.sort()
    if not bad_trials:
        print(f"No trials exceeded path_length threshold {args.thresh:.3f}.")
        return

    print("Trials to remove (path_length > {:.3f}):".format(args.thresh),
          ", ".join(str(t) for t in bad_trials))

    # 2) delete corresponding files in each metric folder
    metric_folders = {
        "density":               "density_trial_",
        "path_length":           "path_length_trial_",
        "safety_distances":      "safety_distances_trial_",
        "time_not_moving":       "time_not_moving_trial_",
        "translational_velocity":"translational_velocity_trial_",
        "travel_time":           "travel_time_trial_",
    }

    for trial in bad_trials:
        for folder, prefix in metric_folders.items():
            dirpath = os.path.join(root, folder)
            fname   = f"{prefix}{trial}.txt"
            fpath   = os.path.join(dirpath, fname)
            if os.path.exists(fpath):
                os.remove(fpath)
                print(f"  Deleted {folder}/{fname}")
            else:
                print(f"  [Warning] not found: {folder}/{fname}")

if __name__ == "__main__":
    main()
