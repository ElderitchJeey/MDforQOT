# experiments/sweeps/plot_same_limit_vs_N_d3.py

import os
import csv
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


def to_bool(x):
    return str(x).strip().lower() in ("true", "1", "yes")


def main():
    in_csv = os.path.join("results", "sweep_KL_vs_MD_multimarginal_d3.csv")
    out_dir = "figures"
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, "same_limit_vs_N_d3.png")

    rows = []
    with open(in_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    by_N = defaultdict(list)
    for r in rows:
        by_N[int(r["N"])].append(r)

    Ns = sorted(by_N.keys())
    same_rate = []
    support = []

    for N in Ns:
        Rs = by_N[N]
        mask = [to_bool(r["both_converged"]) for r in Rs]
        vals = [to_bool(r["same_limit"]) for r in Rs]

        if any(mask):
            rate = np.mean([vals[i] for i in range(len(vals)) if mask[i]])
            cnt = sum(mask)
        else:
            rate = float("nan")
            cnt = 0

        same_rate.append(rate)
        support.append(cnt)

    # ---- plot ----
    plt.figure()
    plt.plot(Ns, same_rate, marker="o", linewidth=2)
    plt.xlabel("number of marginals N (d=3)")
    plt.ylabel("same-limit rate (KL vs MD, when both converged)")
    plt.ylim(0.0, 1.05)
    plt.grid(True, alpha=0.3)

    # optional: annotate sample size
    for x, y, c in zip(Ns, same_rate, support):
        plt.annotate(f"n={c}", (x, y), textcoords="offset points", xytext=(0, 6), ha="center")

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    print(f"Wrote figure: {out_png}")


if __name__ == "__main__":
    main()
