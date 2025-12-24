# experiments/sweeps/plot_eps_summary.py
#
# Usage:
#   python -m experiments.sweeps.plot_eps_summary --N 3 --d 3
#
# It reads:
#   results/sweep_eps_KL_vs_MD_N{N}_d{d}.csv
#
# And produces:
#   figures/convergence_rate_vs_eps_N{N}_d{d}.png
#   figures/median_gibbs_calls_vs_eps_N{N}_d{d}.png

import os
import csv
import argparse
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


def to_bool(x: str) -> bool:
    return str(x).strip().lower() in ("true", "1", "yes")


def main():
    parser = argparse.ArgumentParser(description="Plot eps-sweep summary for KL vs MD.")
    parser.add_argument("--N", type=int, default=3, help="number of marginals")
    parser.add_argument("--d", type=int, default=3, help="local dimension")
    parser.add_argument(
        "--results_dir", type=str, default="results",
        help="directory containing sweep CSV files"
    )
    parser.add_argument(
        "--figures_dir", type=str, default="figures",
        help="directory to save figures"
    )
    parser.add_argument(
        "--annotate_counts", action="store_true",
        help="annotate convergence counts (k/total) above points"
    )
    args = parser.parse_args()

    N, d = args.N, args.d
    results_dir = args.results_dir
    figures_dir = args.figures_dir

    os.makedirs(figures_dir, exist_ok=True)

    in_csv = os.path.join(results_dir, f"sweep_eps_KL_vs_MD_N{N}_d{d}.csv")
    if not os.path.exists(in_csv):
        raise FileNotFoundError(
            f"CSV not found: {in_csv}\n"
            f"Make sure you ran the corresponding sweep and that the filename matches."
        )

    # -----------------------
    # Read CSV
    # -----------------------
    rows = []
    with open(in_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    if len(rows) == 0:
        raise RuntimeError(f"No data rows found in {in_csv}")

    # group by eps
    by_eps = defaultdict(list)
    for r in rows:
        by_eps[float(r["eps"])].append(r)

    eps_list = sorted(by_eps.keys())

    # metrics per eps
    kl_conv_rate, md_conv_rate = [], []
    kl_med_calls, md_med_calls = [], []
    kl_nconv, md_nconv = [], []
    total_per_eps = []

    for eps in eps_list:
        Rs = by_eps[eps]
        n = len(Rs)
        total_per_eps.append(n)

        kl_conv = [to_bool(r["kl_converged"]) for r in Rs]
        md_conv = [to_bool(r["md_converged"]) for r in Rs]

        kl_conv_rate.append(float(np.mean(kl_conv)))
        md_conv_rate.append(float(np.mean(md_conv)))

        # calls only among converged runs
        kl_calls = [int(Rs[i]["kl_gibbs_calls"]) for i in range(n) if kl_conv[i]]
        md_calls = [int(Rs[i]["md_gibbs_calls"]) for i in range(n) if md_conv[i]]

        kl_med_calls.append(float(np.median(kl_calls)) if len(kl_calls) > 0 else np.nan)
        md_med_calls.append(float(np.median(md_calls)) if len(md_calls) > 0 else np.nan)

        kl_nconv.append(int(np.sum(kl_conv)))
        md_nconv.append(int(np.sum(md_conv)))

    # -----------------------
    # Figure 1: convergence rate vs eps
    # -----------------------
    out_png1 = os.path.join(figures_dir, f"convergence_rate_vs_eps_N{N}_d{d}.png")

    plt.figure()
    plt.plot(eps_list, kl_conv_rate, marker="s", linewidth=2, label="KL-descent")
    plt.plot(eps_list, md_conv_rate, marker="o", linewidth=2, label="MD-Sinkhorn")

    plt.xscale("log")
    plt.xlabel("entropic regularization ε")
    plt.ylabel("convergence rate")
    plt.ylim(0.0, 1.05)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()

    if args.annotate_counts:
        for x, y, k, tot in zip(eps_list, kl_conv_rate, kl_nconv, total_per_eps):
            plt.annotate(
                f"{k}/{tot}",
                (x, y),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(out_png1, dpi=200)
    print(f"Wrote figure: {out_png1}")

    # -----------------------
    # Figure 2: median gibbs calls vs eps (converged runs only)
    # -----------------------
    out_png2 = os.path.join(figures_dir, f"median_gibbs_calls_vs_eps_N{N}_d{d}.png")

    plt.figure()
    plt.plot(eps_list, kl_med_calls, marker="s", linewidth=2, label="KL-descent (median, converged)")
    plt.plot(eps_list, md_med_calls, marker="o", linewidth=2, label="MD-Sinkhorn (median, converged)")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("entropic regularization ε")
    plt.ylabel("median Gibbs calls (converged runs only)")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_png2, dpi=200)
    print(f"Wrote figure: {out_png2}")

    print("\nDone.")
    print(f"Input:  {in_csv}")
    print(f"Output: {out_png1}")
    print(f"Output: {out_png2}")


if __name__ == "__main__":
    main()
