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
    dbga_algorithm_2_2,
    potential_marginal_kl_descent,
    md_type_sinkhorn_potential,
)

# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser("Epsilon robustness (non-commuting), journal-style")

    p.add_argument("--N", type=int, required=True)
    p.add_argument("--d", type=int, required=True)

    p.add_argument("--eps_min", type=float, default=0.5)
    p.add_argument("--eps_max", type=float, default=1.0)
    p.add_argument("--eps_num", type=int, default=5)

    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])

    # stopping by F-marg
    p.add_argument("--tol_F", type=float, default=1e-8)

    # unified iteration budget; solver will early-stop if tol_F reached
    p.add_argument("--max_iter", type=int, default=25000)

    # algorithms to run
    # "BDGA" means dbga_algorithm_2_2 (only N=2)
    p.add_argument("--algos", type=str, nargs="+", default=["BDGA", "KL", "MD"])

    # MD inner steps list
    p.add_argument("--M_list", type=int, nargs="+", default=[1])

    # marginal generation kind (matches your instances.py)
    p.add_argument("--marginal_kind", type=str, default="medium",
                   choices=["easy", "medium", "hard"])

    # Optional tol_tr for solvers (disabled by default; we use tol_F as stopping)
    p.add_argument("--tol_tr", type=float, default=0.0)

    # output directory
    p.add_argument("--outdir", type=str, default="experiments/eps_robustness")

    return p.parse_args()

# ============================================================
# Helpers
# ============================================================

def make_eps_list(eps_min, eps_max, eps_num):
    return np.logspace(np.log10(eps_min), np.log10(eps_max), int(eps_num)).tolist()

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def label_for_curve(algo, M_inner):
    if algo == "MD":
        return f"MD (M={int(M_inner)})"
    if algo == "KL":
        return "KL descent"
    if algo == "BDGA":
        return "BDGA"
    return str(algo)

def setup_mpl():
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "figure.figsize": (6.2, 4.2),
        "savefig.bbox": "tight",
    })

def _safe_last(lst, default=np.nan):
    return float(lst[-1]) if (lst is not None and len(lst) > 0) else float(default)

# ============================================================
# Run one solver (uses tol_F early stop inside solver)
# ============================================================

def run_solver(algo, H, gammas, eps, dims, tol_F, max_iter, tol_tr, M_inner):
    """
    Returns dict with:
      converged, gibbs_calls_to_tol, time_sec_to_tol, F_final, e_tr_final, n_iters
    """
    t0 = time.time()

    if algo == "KL":
        res = potential_marginal_kl_descent(
            H=H,
            gammas=gammas,
            eps=float(eps),
            dims=dims,
            T=int(max_iter),
            tol_tr=None if tol_tr <= 0 else float(tol_tr),
            tol_F=float(tol_F),
            store_hist=False,
            project_pi=True,
        )

    elif algo == "MD":
        res = md_type_sinkhorn_potential(
            H=H,
            gammas=gammas,
            eps=float(eps),
            dims=dims,
            T_outer=int(max_iter),
            tol_tr=float(tol_tr) if tol_tr > 0 else 0.0,  # keep but typically disabled
            tol_F=float(tol_F),
            M_inner=int(M_inner),
            keep_U_hist=False,
            keep_pi_hist=False,
            tol_inner=None,
            project_pi=True,
        )

    elif algo == "BDGA":
        if len(dims) != 2:
            raise ValueError("BDGA (dbga_algorithm_2_2) only supports N=2.")
        res = dbga_algorithm_2_2(
            H=H,
            gammas=gammas,
            eps=float(eps),
            dims=dims,
            T=int(max_iter),
            tol_tr=None if tol_tr <= 0 else float(tol_tr),
            tol_F=float(tol_F),
            # keep other defaults (delta etc.) as in SolverofEQOT
            store_hist=False,
            project_pi=True,
        )

    else:
        raise ValueError(f"Unknown algo={algo}")

    wall_total = time.time() - t0

    converged = bool(getattr(res, "converged", False))

    # "to tol" metrics: since solver early-stops by tol_F, last entries correspond to stopping time.
    gibbs_calls = getattr(res, "gibbs_calls", np.nan)
    time_to_tol = _safe_last(getattr(res, "times", None), default=wall_total)

    F_final = _safe_last(getattr(res, "F_list", None), default=np.nan)
    e_tr_final = _safe_last(getattr(res, "e_tr_list", None), default=np.nan)

    n_iters = len(getattr(res, "F_list", [])) - 1  # recorded steps (rough proxy)

    return {
        "converged": int(converged),
        "gibbs_calls_to_tol": float(gibbs_calls) if converged else float("nan"),
        "time_sec_to_tol": float(time_to_tol) if converged else float("nan"),
        "wall_sec_total": float(wall_total),
        "F_final": float(F_final),
        "e_tr_final": float(e_tr_final),
        "n_recorded_steps": int(max(n_iters, 0)),
        "gibbs_calls_total": float(gibbs_calls),
    }

# ============================================================
# Aggregation + plotting
# ============================================================

def aggregate(per_run_rows, eps_list, seeds):
    """
    Returns agg_rows list with keys:
      algo, M_inner, eps, hit_k, hit_rate, gibbs_median, time_median
    Median is over converged seeds only.
    """
    denom = len(seeds)
    grouped_all = defaultdict(list)
    for r in per_run_rows:
        key = (r["algo"], int(r["M_inner"]), float(r["eps"]))
        grouped_all[key].append(r)

    agg_rows = []
    for (algo, M_inner, eps), rows in grouped_all.items():
        conv = [rr for rr in rows if int(rr["converged"]) == 1]
        hit_k = len(conv)
        hit_rate = hit_k / float(denom) if denom > 0 else 0.0

        gibbs_vals = [rr["gibbs_calls_to_tol"] for rr in conv if np.isfinite(rr["gibbs_calls_to_tol"])]
        time_vals = [rr["time_sec_to_tol"] for rr in conv if np.isfinite(rr["time_sec_to_tol"])]

        agg_rows.append({
            "algo": algo,
            "M_inner": int(M_inner),
            "eps": float(eps),
            "hit_k": int(hit_k),
            "hit_rate": float(hit_rate),
            "gibbs_median": float(np.median(gibbs_vals)) if len(gibbs_vals) else float("nan"),
            "time_median": float(np.median(time_vals)) if len(time_vals) else float("nan"),
        })

    # ensure deterministic ordering
    def _order(row):
        order = {"BDGA": 0, "KL": 1, "MD": 2}
        return (order.get(row["algo"], 99), row["M_inner"], row["eps"])
    agg_rows.sort(key=_order)
    return agg_rows

def plot_metric(agg_rows, eps_list, denom_seeds, outpath_base, title, ykey, loglog=False):
    """
    agg_rows contains: algo, M_inner, eps, hit_k, <ykey>
    ykey in {"gibbs_median", "time_median"}
    """
    setup_mpl()
    fig, ax = plt.subplots()

    # curve -> {eps: (y, hit_k)}
    curves = defaultdict(dict)
    for r in agg_rows:
        curve = (r["algo"], int(r["M_inner"]))
        curves[curve][float(r["eps"])] = (float(r[ykey]), int(r["hit_k"]))

    def curve_sort_key(curve):
        algo, M = curve
        order = {"BDGA": 0, "KL": 1, "MD": 2}
        return (order.get(algo, 99), M)

    for (algo, M_inner) in sorted(curves.keys(), key=curve_sort_key):
        xs, ys, hits = [], [], []
        for eps in eps_list:
            v = curves[(algo, M_inner)].get(float(eps), (np.nan, 0))
            xs.append(float(eps))
            ys.append(float(v[0]))
            hits.append(int(v[1]))

        ax.plot(xs, ys, marker="o", label=label_for_curve(algo, M_inner))

        # annotate hit-rate only when < denom and point is finite
        for x, y, hk in zip(xs, ys, hits):
            if np.isfinite(y) and hk < denom_seeds:
                ax.annotate(f"{hk}/{denom_seeds}", (x, y),
                            textcoords="offset points", xytext=(6, 6))

    ax.set_title(title)
    ax.set_xlabel(r"$\varepsilon$")
    ax.set_ylabel("Gibbs calls to reach tol" if ykey == "gibbs_median" else "Time (s) to reach tol")

    ax.grid(True, which="both", alpha=0.3)
    ax.set_xscale("log")
    if loglog:
        ax.set_yscale("log")

    ax.legend(frameon=True)

    suf = ".loglog" if loglog else ".linearY"
    fig.savefig(outpath_base + suf + ".pdf")
    fig.savefig(outpath_base + suf + ".png", dpi=220)
    plt.close(fig)

# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    ensure_dir(args.outdir)

    eps_list = make_eps_list(args.eps_min, args.eps_max, args.eps_num)
    dims = [int(args.d)] * int(args.N)

    # enforce BDGA only for N=2
    algos = list(args.algos)
    if int(args.N) != 2 and "BDGA" in algos:
        algos = [a for a in algos if a != "BDGA"]

    per_run_rows = []

    # --------------------------------------------------------
    # Sweep seeds: fixed instance per (N,d,seed), reused across eps
    # --------------------------------------------------------
    for seed in args.seeds:
        rng = np.random.default_rng(int(seed))

        # non-commuting only: use gen_H_random
        H = gen_H_random(dims=dims, rng=rng)
        gammas = [gen_marginal(int(args.d), rng=rng, kind=args.marginal_kind) for _ in range(int(args.N))]

        for eps in eps_list:
            for algo in algos:
                M_candidates = args.M_list if algo == "MD" else [0]
                for M_inner in M_candidates:
                    result = run_solver(
                        algo=algo,
                        H=H,
                        gammas=gammas,
                        eps=float(eps),
                        dims=dims,
                        tol_F=float(args.tol_F),
                        max_iter=int(args.max_iter),
                        tol_tr=float(args.tol_tr),
                        M_inner=int(M_inner),
                    )
                    per_run_rows.append({
                        "N": int(args.N),
                        "d": int(args.d),
                        "seed": int(seed),
                        "eps": float(eps),
                        "algo": str(algo),
                        "M_inner": int(M_inner),
                        "tol_F": float(args.tol_F),
                        "max_iter": int(args.max_iter),
                        "tol_tr": float(args.tol_tr),
                        "marginal_kind": str(args.marginal_kind),
                        **result,
                    })

    if len(per_run_rows) == 0:
        raise RuntimeError("No runs executed. Check --algos / N / inputs.")

    # --------------------------------------------------------
    # Save per-run CSV
    # --------------------------------------------------------
    per_run_csv = os.path.join(args.outdir, f"per_run_eps_N{args.N}_d{args.d}.csv")
    with open(per_run_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_run_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_run_rows)

    # --------------------------------------------------------
    # Aggregate + save aggregated CSV
    # --------------------------------------------------------
    agg_rows = aggregate(per_run_rows, eps_list, args.seeds)
    agg_csv = os.path.join(args.outdir, f"aggregated_eps_N{args.N}_d{args.d}.csv")
    with open(agg_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
        writer.writeheader()
        writer.writerows(agg_rows)

    # --------------------------------------------------------
    # Plot 4 figures: (gibbs/time) x (linear-y/loglog)
    # --------------------------------------------------------
    denom = len(args.seeds)
    title = f"Epsilon robustness (non-commuting)  N={args.N}, d={args.d}, tol_F={args.tol_F}"
    base = os.path.join(args.outdir, f"eps_robustness_N{args.N}_d{args.d}")

    plot_metric(agg_rows, eps_list, denom, base + "_gibbs", title, "gibbs_median", loglog=False)
    plot_metric(agg_rows, eps_list, denom, base + "_gibbs", title, "gibbs_median", loglog=True)

    plot_metric(agg_rows, eps_list, denom, base + "_time", title, "time_median", loglog=False)
    plot_metric(agg_rows, eps_list, denom, base + "_time", title, "time_median", loglog=True)

    print("Saved results to:", args.outdir)
    print("Per-run CSV:", per_run_csv)
    print("Aggregated CSV:", agg_csv)
    print("Figures:")
    print(" ", base + "_gibbs.linearY.(pdf/png)")
    print(" ", base + "_gibbs.loglog.(pdf/png)")
    print(" ", base + "_time.linearY.(pdf/png)")
    print(" ", base + "_time.loglog.(pdf/png)")

if __name__ == "__main__":
    main()
