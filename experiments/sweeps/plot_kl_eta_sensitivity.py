# experiments/sweeps/plot_kl_eta_sensitivity.py

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
    parser.add_argument("--d", type=int, default=3)
    parser.add_argument("--num_seeds", type=int, default=5)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--figures_dir", type=str, default="figures")
    args = parser.parse_args()

    in_csv = os.path.join(args.results_dir, f"sweep_kl_eta_N{args.N}_d{args.d}_seed{args.num_seeds}.csv")
    os.makedirs(args.figures_dir, exist_ok=True)

    rows = []
    with open(in_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    by_eta = defaultdict(list)
    for r in rows:
        by_eta[float(r["eta"])].append(r)

    etas = sorted(by_eta.keys())

    conv_rate = []
    mean_calls = []
    same_rate = []
    support_calls = []
    support_same = []

    for eta in etas:
        Rs = by_eta[eta]
        conv = [to_bool(r["converged"]) for r in Rs]
        conv_rate.append(float(np.mean(conv)))

        # mean calls among converged
        calls = [int(r["gibbs_calls"]) for i, r in enumerate(Rs) if conv[i]]
        mean_calls.append(float(np.mean(calls)) if len(calls) > 0 else np.nan)
        support_calls.append(len(calls))

        # same-limit vs ref among BOTH converged (enforced in same_limit(require_converged=True))
        # We still restrict to cases where current converged AND ref converged.
        ref_conv = [to_bool(r["ref_converged"]) for r in Rs]
        mask = [conv[i] and ref_conv[i] for i in range(len(Rs))]
        vals = [to_bool(r["same_limit_vs_ref"]) for r in Rs]
        xs = [vals[i] for i in range(len(vals)) if mask[i]]
        same_rate.append(float(np.mean(xs)) if len(xs) > 0 else np.nan)
        support_same.append(len(xs))

    # -----------------------
    # Plot 1: conv rate
    # -----------------------
    out1 = os.path.join(args.figures_dir, f"kl_eta_conv_rate_N{args.N}_d{args.d}.png")
    plt.figure()
    plt.plot(etas, conv_rate, marker="o", linewidth=2)
    plt.xlabel("step size η")
    plt.ylabel("convergence rate")
    plt.ylim(0.0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out1, dpi=200)
    print(f"Wrote: {out1}")

    # -----------------------
    # Plot 2: mean calls (converged)
    # -----------------------
    out2 = os.path.join(args.figures_dir, f"kl_eta_mean_calls_N{args.N}_d{args.d}.png")
    plt.figure()
    plt.plot(etas, mean_calls, marker="s", linewidth=2)
    plt.xlabel("step size η")
    plt.ylabel("mean Gibbs calls (converged runs only)")
    plt.grid(True, alpha=0.3)

    # annotate support (optional)
    for x, y, n in zip(etas, mean_calls, support_calls):
        if np.isfinite(y):
            plt.annotate(f"n={n}", (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(out2, dpi=200)
    print(f"Wrote: {out2}")

    # -----------------------
    # Plot 3: same-limit rate vs ref
    # -----------------------
    out3 = os.path.join(args.figures_dir, f"kl_eta_same_limit_vs_ref_N{args.N}_d{args.d}.png")
    plt.figure()
    plt.plot(etas, same_rate, marker="^", linewidth=2)
    plt.xlabel("step size η")
    plt.ylabel("same-limit rate vs η_ref (both converged)")
    plt.ylim(0.0, 1.05)
    plt.grid(True, alpha=0.3)

    for x, y, n in zip(etas, same_rate, support_same):
        if np.isfinite(y):
            plt.annotate(f"n={n}", (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(out3, dpi=200)
    print(f"Wrote: {out3}")


if __name__ == "__main__":
    main()
