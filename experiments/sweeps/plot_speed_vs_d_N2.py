# experiments/sweeps/plot_speed_vs_d_N2.py
#
# Read results/sweep_speed_vs_d_N2.csv and plot:
#   Fig0: convergence rate vs d (N=2)
#   Fig1: median gibbs calls vs d (converged runs only)
#   Fig2: median time (sec) vs d (converged runs only)
#   Fig3: median time/gibbs_call vs d (converged runs only)
#
# Run:
#   python -m experiments.sweeps.plot_speed_vs_d_N2

import os
import csv
import argparse
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


def to_bool(x) -> bool:
    return str(x).strip().lower() in ("true", "1", "yes")


def safe_median(vals):
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.median(vals)) if len(vals) > 0 else np.nan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_csv", type=str, default=os.path.join("results", "sweep_speed_vs_d_N2.csv"))
    parser.add_argument("--figures_dir", type=str, default="figures")
    args = parser.parse_args()

    os.makedirs(args.figures_dir, exist_ok=True)

    rows = []
    with open(args.in_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    # group by d
    by_d = defaultdict(list)
    for r in rows:
        by_d[int(r["d"])].append(r)

    ds = sorted(by_d.keys())

    # helpers for each algo
    algos = [
        ("BGDA", "bgda"),
        ("KL-descent", "kl"),
        ("MD-Sinkhorn", "md"),
    ]

    conv_rate = {key: [] for _, key in algos}
    med_calls = {key: [] for _, key in algos}
    med_time = {key: [] for _, key in algos}
    med_time_per_call = {key: [] for _, key in algos}

    for d in ds:
        Rs = by_d[d]
        for _, key in algos:
            conv = [to_bool(r[f"{key}_converged"]) for r in Rs]
            conv_rate[key].append(float(np.mean(conv)))

            calls = []
            times = []
            t_over_c = []
            for i, r in enumerate(Rs):
                if not conv[i]:
                    continue
                c = float(r[f"{key}_gibbs_calls"])
                t = float(r[f"{key}_time_sec"])
                if c > 0 and np.isfinite(c):
                    calls.append(c)
                    if np.isfinite(t):
                        times.append(t)
                        t_over_c.append(t / c)

            med_calls[key].append(safe_median(calls))
            med_time[key].append(safe_median(times))
            med_time_per_call[key].append(safe_median(t_over_c))

    # ---------------------------------------------------------
    # Fig0: convergence rate vs d
    # ---------------------------------------------------------
    out0 = os.path.join(args.figures_dir, "Fig0_conv_rate_vs_d_N2.png")
    plt.figure()
    for label, key in algos:
        plt.plot(ds, conv_rate[key], marker="o", linewidth=2, label=label)
    plt.xlabel("dimension d (N=2)")
    plt.ylabel("convergence rate")
    plt.ylim(0.0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out0, dpi=200)
    print(f"Wrote: {out0}")

    # ---------------------------------------------------------
    # Fig1: median Gibbs calls vs d (converged only)
    # ---------------------------------------------------------
    out1 = os.path.join(args.figures_dir, "Fig1_median_calls_vs_d_N2.png")
    plt.figure()
    for label, key in algos:
        plt.plot(ds, med_calls[key], marker="s", linewidth=2, label=label)
    plt.xlabel("dimension d (N=2)")
    plt.ylabel("median Gibbs calls (converged runs only)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out1, dpi=200)
    print(f"Wrote: {out1}")

    # ---------------------------------------------------------
    # Fig2: median time vs d (converged only)
    # ---------------------------------------------------------
    out2 = os.path.join(args.figures_dir, "Fig2_median_time_vs_d_N2.png")
    plt.figure()
    for label, key in algos:
        plt.plot(ds, med_time[key], marker="^", linewidth=2, label=label)
    plt.xlabel("dimension d (N=2)")
    plt.ylabel("median time (sec, converged runs only)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out2, dpi=200)
    print(f"Wrote: {out2}")

    # ---------------------------------------------------------
    # Fig3: median time/call vs d (converged only)
    # ---------------------------------------------------------
    out3 = os.path.join(args.figures_dir, "Fig3_median_time_per_call_vs_d_N2.png")
    plt.figure()
    for label, key in algos:
        plt.plot(ds, med_time_per_call[key], marker="d", linewidth=2, label=label)
    plt.xlabel("dimension d (N=2)")
    plt.ylabel("median time / Gibbs call (sec/call, converged runs only)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out3, dpi=200)
    print(f"Wrote: {out3}")


if __name__ == "__main__":
    main()
