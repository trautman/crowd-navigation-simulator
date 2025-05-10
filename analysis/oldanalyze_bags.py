#!/usr/bin/env python3
import argparse
import math

def compute_mean_std(data):
    n = len(data)
    if n == 0:
        return float('nan'), float('nan')
    mean = sum(data) / n
    var  = sum((x - mean) ** 2 for x in data) / n
    std  = math.sqrt(var)
    return mean, std

def main():
    p = argparse.ArgumentParser(
        description="Read metrics.txt files (one density per line) and compute mean & std."
    )
    p.add_argument(
        'metrics_files',
        nargs='*',
        default=['metrics.txt'],
        help="Path(s) to metrics.txt (default: ./metrics.txt)"
    )
    args = p.parse_args()

    for path in args.metrics_files:
        densities = []
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        densities.append(float(line))
                    except ValueError:
                        # skip any malformed lines
                        continue
        except FileNotFoundError:
            print(f"{path}: file not found")
            continue

        mean, std = compute_mean_std(densities)
        print(f"{path}: mean = {mean:.6f}, std = {std:.6f}")

if __name__ == '__main__':
    main()
