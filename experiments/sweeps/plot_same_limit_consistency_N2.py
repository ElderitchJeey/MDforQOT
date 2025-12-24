# experiments/sweeps/plot_same_limit_consistency_N2.py

import os
import csv
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


def to_bool(x):
    return str(x).strip().lower() in ("true", "1", "yes")


def rate_when_both(mask, values):
    xs = [values[i] for i in range(len(values)) if mask[i]]
    return np.mean(xs) if len(xs) > 0 else np.nan


def main():
    in_csv = os.path.join("results", "sweep_vs_bgda_N2.csv")
    out_dir = "figures"
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, "same_limit_consistency_N2.png")

    rows = []
    with open(in_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    by_d = defaultdict(list)
    for r in rows:
        by_d[int(r["d"])].append(r)

    ds = sorted(by_d.keys())

    kl_vs_bgda, md_vs_bgda, md_vs_kl = [], [], []

    for d in ds:
        Rs = by_d[d]

        bgda_c = [to_bool(r["bgda_converged"]) for r in Rs]
        kl_c   = [to_bool(r["kl_converged"]) for r in Rs]
        md_c   = [to_bool(r["md_converged"]) for r in Rs]

        kl_same = [to_bool(r["kl_same_limit"]) for r in Rs]
        md_same = [to_bool(r["md_same_limit"]) for r in Rs]
        mdkl_same = [to_bool(r["md_vs_kl_same_limit"]) for r in Rs]

        kl_vs_bgda.append(rate_when_both(
            [bgda_c[i] and kl_c[i] for i in range(len(Rs))],
            kl_same
        ))

        md_vs_bgda.append(rate_when_both(
            [bgda_c[i] and md_c[i] for i in range(len(Rs))],
            md_same
        ))

        md_vs_kl.append(rate_when_both(
            [md_c[i] and kl_c[i] for i in range(len(Rs))],
            mdkl_same
        ))

    # ---- plot ----
    plt.figure()
    plt.plot(ds, kl_vs_bgda, marker="s", linewidth=2, label="KL vs BGDA")
    plt.plot(ds, md_vs_bgda, marker="^", linewidth=2, label="MD vs BGDA")
    plt.plot(ds, md_vs_kl, marker="o", linewidth=2, label="MD vs KL")

    plt.xlabel("dimension d (N=2)")
    plt.ylabel("same-limit rate (when both converged)")
    plt.ylim(0.0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    print(f"Wrote figure: {out_png}")


if __name__ == "__main__":
    main()
