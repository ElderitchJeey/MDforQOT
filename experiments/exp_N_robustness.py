from __future__ import print_function

import os
import sys
import time
import csv
import argparse
import numpy as np
from collections import defaultdict

# ===============================
# Non-interactive backend (HPC-safe)
# ===============================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ===============================
# Project imports
# ===============================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.instances import gen_H_random, gen_marginal
from src.SolverofEQOT import (
    potential_marginal_kl_descent,
    md_type_sinkhorn_potential,
)

# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser("Robustness w.r.t. number of marginals N (fixed eps=1)")

    p.add_argument("--d", type=int, required=True, help="dimension of each marginal")
    p.add_argument("--N_list", type=int, nargs="+", required=True)

    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])

    # Fixed protocol (still adjustable from CLI if needed)
    p.add_argument("--eps", type=float, default=1.0)
    p.add_argument("--tol_F", type=float, default=1e-8)
    p.add_argument("--max_iter", type=int, default=25000)

    p.add_argument("--M_list", type=int, nargs="+", default=[1, 2, 5])

    p.add_argument("--outdir", type=str, default="experiments/N_robustness")

    return p.parse_args()

# ============================================================
# Plot style (journal-ish)
# ============================================================

def setup_mpl():
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "figure.figsize": (6.6, 4.2),
        "savefig.bbox": "tight",
    })

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

# ============================================================
# Run one solver (aligned to SolverofEQOT_tolF_fixed.py)
#   - init potentials are zero INSIDE solver (no keyword needed)
#   - early stop is by tol_F, budget = max_iter
# ============================================================

def run_solver(algo, H, gammas, eps, dims, tol_F, max_iter, M_inner=None):
    t0 = time.time()

    if algo == "KL":
        res = potential_marginal_kl_descent(
            H=H,
            gammas=gammas,
            eps=float(eps),
            dims=dims,
            T=int(max_iter),
            tol_F=float(tol_F),
            tol_tr=None,          # we stop by F-marg
            store_hist=False,
            project_pi=True,
        )

    elif algo == "MD":
        # md_type_sinkhorn_potential requires T_outer and tol_tr (even if tol_F is used)
        res = md_type_sinkhorn_potential(
            H=H,
            gammas=gammas,
            eps=float(eps),
            dims=dims,
            T_outer=int(max_iter),
            tol_tr=0.0,           # disable trace stop; rely on tol_F
            tol_F=float(tol_F),
            M_inner=int(M_inner),
            keep_U_hist=False,
            keep_pi_hist=False,
            tol_inner=None,
            project_pi=True,
        )
    else:
        raise ValueError(f"Unknown algo={algo}")

    wall_total = time.time() - t0

    converged = int(bool(getattr(res, "converged", False)))
    gibbs_total = float(getattr(res, "gibbs_calls", np.nan))

    # penalized "to tol" metrics:
    # - if converged: use last recorded time (solver early-stops at tol_F)
    # - else: use full wall_total and gibbs_total
    times = getattr(res, "times", [])
    if converged and times:
        time_to_tol = float(times[-1])
        gibbs_to_tol = float(gibbs_total)
    else:
        time_to_tol = float(wall_total)
        gibbs_to_tol = float(gibbs_total)

    # seconds per gibbs call (to tol) (penalized)
    sec_per_gibbs = float(time_to_tol / gibbs_to_tol) if (gibbs_to_tol > 0 and np.isfinite(gibbs_to_tol)) else float("nan")

    return {
        "converged": converged,
        "gibbs_to_tol": gibbs_to_tol,
        "time_to_tol": time_to_tol,
        "sec_per_gibbs_to_tol": sec_per_gibbs,
        "gibbs_total": gibbs_total,
        "wall_sec_total": float(wall_total),
    }

# ============================================================
# Aggregation: median over ALL seeds (penalized), plus hit-rate
# ============================================================

def aggregate(per_run_rows, seeds):
    denom = len(seeds)
    grouped = defaultdict(list)

    for r in per_run_rows:
        key = (r["algo"], int(r["M_inner"]), int(r["N"]))
        grouped[key].append(r)

    agg_rows = []
    for (algo, M, N), rows in grouped.items():
        hit_k = sum(int(rr["converged"]) for rr in rows)

        gibbs_vals = [float(rr["gibbs_to_tol"]) for rr in rows]
        secpg_vals = [float(rr["sec_per_gibbs_to_tol"]) for rr in rows]

        agg_rows.append({
            "algo": algo,
            "M_inner": int(M),
            "N": int(N),
            "hit_k": int(hit_k),
            "hit_rate": float(hit_k / denom) if denom > 0 else 0.0,
            "gibbs_median": float(np.median(gibbs_vals)),
            "sec_per_gibbs_median": float(np.median(secpg_vals)),
        })

    # deterministic ordering: KL first, then MD by M
    def _ord(row):
        algo_rank = {"KL": 0, "MD": 1}.get(row["algo"], 99)
        return (algo_rank, row["M_inner"], row["N"])
    agg_rows.sort(key=_ord)
    return agg_rows

# ============================================================
# Plot helpers
# ============================================================

def plot_metric(agg_rows, N_list, denom_seeds, ykey, ylabel, title, outbase, logy=False):
    setup_mpl()
    fig, ax = plt.subplots()

    # curve key -> N -> row
    curves = defaultdict(dict)
    for r in agg_rows:
        curves[(r["algo"], int(r["M_inner"]))][int(r["N"])] = r

    # fixed legend order
    curve_order = [("KL", 0), ("MD", 1), ("MD", 2), ("MD", 5)]

    for algo, M in curve_order:
        if (algo, M) not in curves:
            continue
        data = curves[(algo, M)]

        xs, ys = [], []
        for N in N_list:
            rr = data.get(int(N), None)
            xs.append(int(N))
            ys.append(float(rr[ykey]) if rr is not None else float("nan"))

        label = "KL descent" if algo == "KL" else f"MD (M={M})"
        ax.plot(xs, ys, marker="o", label=label)

        # annotate hit-rate < 1
        for N in N_list:
            rr = data.get(int(N), None)
            if rr is None:
                continue
            if int(rr["hit_k"]) < denom_seeds:
                ax.annotate(
                    f"{int(rr['hit_k'])}/{denom_seeds}",
                    (int(N), float(rr[ykey])),
                    textcoords="offset points",
                    xytext=(6, 6),
                )

    ax.set_xlabel("Number of marginals N")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks([int(n) for n in N_list])  # integer ticks
    ax.grid(True, alpha=0.3)

    if logy:
        ax.set_yscale("log")

    ax.legend(frameon=True)

    suf = ".logy" if logy else ".linearY"
    fig.savefig(outbase + suf + ".png", dpi=220)
    fig.savefig(outbase + suf + ".pdf")
    plt.close(fig)

# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    ensure_dir(args.outdir)

    N_list = [int(x) for x in args.N_list]
    N_list_sorted = sorted(N_list)

    per_run_rows = []

    for seed in args.seeds:
        rng = np.random.default_rng(int(seed))

        for N in N_list_sorted:
            dims = [int(args.d)] * int(N)

            # new instance per (N,d,seed)
            H = gen_H_random(dims=dims, rng=rng)
            gammas = [gen_marginal(int(args.d), rng=rng) for _ in range(int(N))]

            # KL
            r_kl = run_solver(
                algo="KL",
                H=H,
                gammas=gammas,
                eps=args.eps,
                dims=dims,
                tol_F=args.tol_F,
                max_iter=args.max_iter,
                M_inner=None,
            )
            per_run_rows.append({
                "seed": int(seed),
                "N": int(N),
                "d": int(args.d),
                "eps": float(args.eps),
                "tol_F": float(args.tol_F),
                "max_iter": int(args.max_iter),
                "algo": "KL",
                "M_inner": 0,
                **r_kl,
            })

            # MD for each M
            for M in args.M_list:
                r_md = run_solver(
                    algo="MD",
                    H=H,
                    gammas=gammas,
                    eps=args.eps,
                    dims=dims,
                    tol_F=args.tol_F,
                    max_iter=args.max_iter,
                    M_inner=int(M),
                )
                per_run_rows.append({
                    "seed": int(seed),
                    "N": int(N),
                    "d": int(args.d),
                    "eps": float(args.eps),
                    "tol_F": float(args.tol_F),
                    "max_iter": int(args.max_iter),
                    "algo": "MD",
                    "M_inner": int(M),
                    **r_md,
                })

    # Save per-run CSV
    per_csv = os.path.join(args.outdir, f"per_run_N_robustness_d{args.d}_eps{args.eps}.csv")
    with open(per_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_run_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_run_rows)

    # Aggregate CSV
    agg_rows = aggregate(per_run_rows, args.seeds)
    agg_csv = os.path.join(args.outdir, f"aggregated_N_robustness_d{args.d}_eps{args.eps}.csv")
    with open(agg_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
        writer.writeheader()
        writer.writerows(agg_rows)

    # Plots
    title = f"N-robustness (non-commuting)  d={args.d}, eps={args.eps}, tol_F={args.tol_F}"
    outbase_gibbs = os.path.join(args.outdir, f"gibbs_vs_N_d{args.d}_eps{args.eps}")
    outbase_secpg = os.path.join(args.outdir, f"sec_per_gibbs_vs_N_d{args.d}_eps{args.eps}")

    plot_metric(
        agg_rows, N_list_sorted, len(args.seeds),
        ykey="gibbs_median",
        ylabel="Gibbs calls to reach tol",
        title=title,
        outbase=outbase_gibbs,
        logy=False,
    )
    plot_metric(
        agg_rows, N_list_sorted, len(args.seeds),
        ykey="gibbs_median",
        ylabel="Gibbs calls to reach tol",
        title=title,
        outbase=outbase_gibbs,
        logy=True,
    )

    plot_metric(
        agg_rows, N_list_sorted, len(args.seeds),
        ykey="sec_per_gibbs_median",
        ylabel="Seconds per Gibbs call (to tol)",
        title=title,
        outbase=outbase_secpg,
        logy=False,
    )
    plot_metric(
        agg_rows, N_list_sorted, len(args.seeds),
        ykey="sec_per_gibbs_median",
        ylabel="Seconds per Gibbs call (to tol)",
        title=title,
        outbase=outbase_secpg,
        logy=True,
    )

    print("Saved results to:", args.outdir)
    print("Per-run CSV:", per_csv)
    print("Aggregated CSV:", agg_csv)

if __name__ == "__main__":
    main()
