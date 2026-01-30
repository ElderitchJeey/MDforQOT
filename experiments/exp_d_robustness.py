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

def label_curve(algo, M):
    if algo == "KL":
        return "KL descent"
    if algo == "MD":
        return f"MD (M={int(M)})"
    return f"{algo}"

# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser("d-robustness experiments (non-commuting)")

    p.add_argument("--seeds", type=int, nargs="+", default=[0,1,2,3,4])

    p.add_argument("--eps", type=float, default=1.0)
    p.add_argument("--tol_F", type=float, default=1e-8)
    p.add_argument("--max_iter", type=int, default=25000)

    # default experiments as requested
    p.add_argument("--d_min_N2", type=int, default=2)
    p.add_argument("--d_max_N2", type=int, default=12)

    p.add_argument("--d_min_N3", type=int, default=2)
    p.add_argument("--d_max_N3", type=int, default=6)

    # MD inner steps list
    p.add_argument("--M_list", type=int, nargs="+", default=[1,2,5])

    p.add_argument("--outdir", type=str, default="experiments/d_robustness")

    # choose subset if needed
    # choices: E1, E2, E3
    p.add_argument("--which", type=str, nargs="+", default=["E1","E2","E3"])

    return p.parse_args()

# ============================================================
# One solver run (aligned to new SolverofEQOT)
# - init potentials are zero inside solver
# - early stop by tol_F inside solver
# - penalize non-converged by using budget (wall_total, gibbs_total)
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
            tol_tr=None,
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
            tol_tr=0.0,
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
    times = getattr(res, "times", [])

    if converged and times:
        time_to_tol = float(times[-1])
        gibbs_to_tol = float(gibbs_total)
    else:
        time_to_tol = float(wall_total)
        gibbs_to_tol = float(gibbs_total)

    sec_per_gibbs = (time_to_tol / gibbs_to_tol) if (np.isfinite(gibbs_to_tol) and gibbs_to_tol > 0) else np.nan

    return {
        "converged": converged,
        "gibbs_to_tol": gibbs_to_tol,
        "time_to_tol": time_to_tol,
        "sec_per_gibbs_to_tol": float(sec_per_gibbs) if np.isfinite(sec_per_gibbs) else float("nan"),
        "gibbs_total": gibbs_total,
        "wall_sec_total": float(wall_total),
    }

# ============================================================
# Aggregate: median over ALL seeds (penalized) + hit-rate
# ============================================================

def aggregate(per_run_rows, seeds):
    denom = len(seeds)
    grouped = defaultdict(list)  # (algo, M, d) -> list[r]
    for r in per_run_rows:
        key = (r["algo"], int(r["M_inner"]), int(r["d"]))
        grouped[key].append(r)

    agg_rows = []
    for (algo, M, d), rows in grouped.items():
        hit_k = sum(int(rr["converged"]) for rr in rows)

        gibbs_vals = [float(rr["gibbs_to_tol"]) for rr in rows]
        agg_rows.append({
            "algo": algo,
            "M_inner": int(M),
            "d": int(d),
            "hit_k": int(hit_k),
            "hit_rate": float(hit_k / denom) if denom > 0 else 0.0,
            "gibbs_median": float(np.median(gibbs_vals)),
        })

    # deterministic ordering
    def _ord(r):
        rank = {"KL": 0, "MD": 1}.get(r["algo"], 99)
        return (rank, r["M_inner"], r["d"])
    agg_rows.sort(key=_ord)
    return agg_rows

# ============================================================
# Plot: x=d integer ticks, y linear + logy
# ============================================================

def plot_gibbs_vs_d(agg_rows, d_list, denom_seeds, title, outbase, logy=False):
    setup_mpl()
    fig, ax = plt.subplots()

    curves = defaultdict(dict)  # (algo,M) -> d -> row
    for r in agg_rows:
        curves[(r["algo"], int(r["M_inner"]))][int(r["d"])] = r

    # legend order: KL then MD by M
    curve_keys = sorted(curves.keys(), key=lambda k: (0 if k[0]=="KL" else 1, k[1]))

    for (algo, M) in curve_keys:
        xs, ys = [], []
        for d in d_list:
            rr = curves[(algo, M)].get(int(d), None)
            xs.append(int(d))
            ys.append(float(rr["gibbs_median"]) if rr is not None else float("nan"))

        ax.plot(xs, ys, marker="o", label=label_curve(algo, M))

        # annotate hit-rate < 1
        for d in d_list:
            rr = curves[(algo, M)].get(int(d), None)
            if rr is None:
                continue
            if int(rr["hit_k"]) < denom_seeds:
                ax.annotate(
                    f"{int(rr['hit_k'])}/{denom_seeds}",
                    (int(d), float(rr["gibbs_median"])),
                    textcoords="offset points",
                    xytext=(6, 6),
                )

    ax.set_xlabel("Dimension d")
    ax.set_ylabel("Gibbs calls to reach tol")
    ax.set_title(title)

    ax.set_xticks([int(x) for x in d_list])
    ax.grid(True, which="both", alpha=0.3)

    if logy:
        ax.set_yscale("log")

    ax.legend(frameon=True)

    suf = ".logy" if logy else ".linearY"
    fig.savefig(outbase + suf + ".png", dpi=220)
    fig.savefig(outbase + suf + ".pdf")
    plt.close(fig)

# ============================================================
# One experiment runner
# ============================================================

def run_experiment(exp_name, N, d_list, algos, M_list, seeds, eps, tol_F, max_iter, outdir):
    """
    exp_name: string
    N: int
    d_list: list[int]
    algos: list of "KL"/"MD"
    M_list: list[int] for MD
    """
    exp_dir = os.path.join(outdir, exp_name)
    ensure_dir(exp_dir)

    per_run_rows = []
    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        for d in d_list:
            dims = [int(d)] * int(N)

            # instance fixed per (N,d,seed)
            H = gen_H_random(dims=dims, rng=rng)
            gammas = [gen_marginal(int(d), rng=rng) for _ in range(int(N))]

            for algo in algos:
                if algo == "KL":
                    out = run_solver("KL", H, gammas, eps, dims, tol_F, max_iter, M_inner=None)
                    per_run_rows.append({
                        "exp": exp_name,
                        "N": int(N),
                        "d": int(d),
                        "seed": int(seed),
                        "eps": float(eps),
                        "tol_F": float(tol_F),
                        "max_iter": int(max_iter),
                        "algo": "KL",
                        "M_inner": 0,
                        **out,
                    })
                elif algo == "MD":
                    for M in M_list:
                        out = run_solver("MD", H, gammas, eps, dims, tol_F, max_iter, M_inner=int(M))
                        per_run_rows.append({
                            "exp": exp_name,
                            "N": int(N),
                            "d": int(d),
                            "seed": int(seed),
                            "eps": float(eps),
                            "tol_F": float(tol_F),
                            "max_iter": int(max_iter),
                            "algo": "MD",
                            "M_inner": int(M),
                            **out,
                        })
                else:
                    raise ValueError(f"Unknown algo {algo}")

    # save per-run
    per_csv = os.path.join(exp_dir, f"per_run_{exp_name}.csv")
    with open(per_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_run_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_run_rows)

    # aggregate
    agg_rows = aggregate(per_run_rows, seeds)
    agg_csv = os.path.join(exp_dir, f"aggregated_{exp_name}.csv")
    with open(agg_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
        writer.writeheader()
        writer.writerows(agg_rows)

    # plots: gibbs vs d (linear + logy)
    title = f"d-robustness ({exp_name})  N={N}, eps={eps}, tol_F={tol_F}"
    outbase = os.path.join(exp_dir, f"gibbs_vs_d_{exp_name}")
    plot_gibbs_vs_d(agg_rows, d_list, len(seeds), title, outbase, logy=False)
    plot_gibbs_vs_d(agg_rows, d_list, len(seeds), title, outbase, logy=True)

    print(f"[{exp_name}] Saved to {exp_dir}")
    print("  per-run:", per_csv)
    print("  aggregated:", agg_csv)

# ============================================================
# Main: run E1/E2/E3
# ============================================================

def main():
    args = parse_args()
    ensure_dir(args.outdir)

    seeds = list(args.seeds)

    d_list_N2 = list(range(int(args.d_min_N2), int(args.d_max_N2) + 1))
    d_list_N3 = list(range(int(args.d_min_N3), int(args.d_max_N3) + 1))

    which = set([w.strip() for w in args.which])

    # E1: eps=1, N=2, d=2..12, compare KL vs MD(M=1)
    if "E1" in which:
        run_experiment(
            exp_name="E1_N2_KL_vs_MD_M1",
            N=2,
            d_list=d_list_N2,
            algos=["KL", "MD"],
            M_list=[1],
            seeds=seeds,
            eps=args.eps,
            tol_F=args.tol_F,
            max_iter=args.max_iter,
            outdir=args.outdir,
        )

    # E2: eps=1, N=2, d=2..12, compare MD(M=1,2,5)
    if "E2" in which:
        run_experiment(
            exp_name="E2_N2_MD_M125",
            N=2,
            d_list=d_list_N2,
            algos=["MD"],
            M_list=list(args.M_list),
            seeds=seeds,
            eps=args.eps,
            tol_F=args.tol_F,
            max_iter=args.max_iter,
            outdir=args.outdir,
        )

    # E3: eps=1, N=3, d=2..6, compare KL vs MD(M=1,2,5)
    if "E3" in which:
        run_experiment(
            exp_name="E3_N3_KL_vs_MD_M125",
            N=3,
            d_list=d_list_N3,
            algos=["KL", "MD"],
            M_list=list(args.M_list),
            seeds=seeds,
            eps=args.eps,
            tol_F=args.tol_F,
            max_iter=args.max_iter,
            outdir=args.outdir,
        )

if __name__ == "__main__":
    main()