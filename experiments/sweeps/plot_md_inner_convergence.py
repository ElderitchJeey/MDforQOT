# experiments/sweeps/plot_md_inner_convergence.py
#
# Plot summary of inner-loop sweep CSV:
# - hit rate per block i
# - median hit gibbs calls per block i (among hit samples)
#
# Run:
#   python -m experiments.sweeps.plot_md_inner_convergence --N 2 --d 4 --eps 0.5 --num_seeds 10

import os
import csv
import argparse
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


def to_bool(x) -> bool:
    return str(x).strip().lower() in ("true", "1", "yes")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=2)
    parser.add_argument("--d", type=int, default=4)
    parser.add_argument("--eps", type=float, default=0.5)
    parser.add_argument("--num_seeds", type=int, default=10)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--figures_dir", type=str, default="figures")
    args = parser.parse_args()

    os.makedirs(args.figures_dir, exist_ok=True)

    in_csv = os.path.join(
        args.results_dir,
        f"sweep_md_inner_conv_N{args.N}_d{args.d}_eps{args.eps:g}_seeds{args.num_seeds}.csv",
    )
    if not os.path.exists(in_csv):
        raise FileNotFoundError(f"CSV not found: {in_csv}")

    rows = []
    with open(in_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    by_i = defaultdict(list)
    for r in rows:
        by_i[int(r["i"])].append(r)

    is_ = sorted(by_i.keys())

    hit_rate = []
    med_hit_calls = []
    support_hit = []

    for i in is_:
        Rs = by_i[i]
        hits = [to_bool(r["hit"]) for r in Rs]
        hit_rate.append(float(np.mean(hits)))

        calls = [int(r["hit_gibbs_calls"]) for j, r in enumerate(Rs) if hits[j] and int(r["hit_gibbs_calls"]) >= 0]
        support_hit.append(len(calls))
        med_hit_calls.append(float(np.median(calls)) if len(calls) > 0 else np.nan)

    # Plot 1: hit rate vs i
    out1 = os.path.join(args.figures_dir, f"md_inner_hit_rate_N{args.N}_d{args.d}_eps{args.eps:g}.png")
    plt.figure()
    plt.plot(is_, hit_rate, marker="o", linewidth=2)
    plt.xlabel("block index i")
    plt.ylabel("hit rate (reach tol_inner within M_max)")
    plt.ylim(0.0, 1.05)
    plt.xticks(is_)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out1, dpi=200)
    print(f"Wrote: {out1}")

    # Plot 2: median hit calls vs i
    out2 = os.path.join(args.figures_dir, f"md_inner_median_hit_calls_N{args.N}_d{args.d}_eps{args.eps:g}.png")
    plt.figure()
    plt.plot(is_, med_hit_calls, marker="s", linewidth=2)
    plt.xlabel("block index i")
    plt.ylabel("median Gibbs calls to hit tol (hit samples only)")
    plt.xticks(is_)
    plt.grid(True, alpha=0.3)

    for x, y, n in zip(is_, med_hit_calls, support_hit):
        if np.isfinite(y):
            plt.annotate(f"n={n}", (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(out2, dpi=200)
    print(f"Wrote: {out2}")


if __name__ == "__main__":
    main()
