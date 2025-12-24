# experiments/sweeps/plot_convergence_rate_N2.py

import os
import csv
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


def to_bool(x):
    return str(x).strip().lower() in ("true", "1", "yes")


def main():
    in_csv = os.path.join("results", "sweep_vs_bgda_N2.csv")
    out_dir = "figures"
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, "convergence_rate_N2.png")

    rows = []
    with open(in_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    # group by d
    by_d = defaultdict(list)
    for r in rows:
        by_d[int(r["d"])].append(r)

    ds = sorted(by_d.keys())

    bgda_rate, kl_rate, md_rate = [], [], []

    for d in ds:
        Rs = by_d[d]
        bgda_rate.append(np.mean([to_bool(r["bgda_converged"]) for r in Rs]))
        kl_rate.append(np.mean([to_bool(r["kl_converged"]) for r in Rs]))
        md_rate.append(np.mean([to_bool(r["md_converged"]) for r in Rs]))

    # ---- plot ----
    plt.figure()
    plt.plot(ds, bgda_rate, marker="o", linewidth=2, label="BGDA")
    plt.plot(ds, kl_rate, marker="s", linewidth=2, label="KL-descent")
    plt.plot(ds, md_rate, marker="^", linewidth=2, label="MD-Sinkhorn")

    plt.xlabel("dimension d (N=2)")
    plt.ylabel("convergence rate")
    plt.ylim(0.0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    print(f"Wrote figure: {out_png}")


if __name__ == "__main__":
    main()
