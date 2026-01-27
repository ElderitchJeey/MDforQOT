# experiments/sweep_tol_complexity_multimarginal.py
"""
Sweep complexity at fixed tolerance tau=1e-8 for multi-marginal entropic QOT.

Goal:
  Compare KL descent vs MD-Sinkhorn in terms of Gibbs calls to reach F_marg <= tau.

Outputs:
  1) per-run CSV:    experiments/figures/sweep_tol/per_run.csv
  2) summary CSV:    experiments/figures/sweep_tol/summary.csv
  3) publication PDF/PNG figure (Option A):
       x-axis: log10(eps)
       y-axis: log10(gibbs_calls_to_tol)
       panels: N
       color: d
       linestyle: algorithm
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# ----------------------------
# Make imports robust (repo-root on sys.path)
# ----------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.instances import (
    gen_H_random,
    gen_H_commuting,
    gen_H_conjugated_from_diagonal,
    gen_marginal,
)
from src.SolverofEQOT import (
    potential_marginal_kl_descent,
    md_type_sinkhorn_potential,
)
from src.linalg import hermitianize


# ============================================================
# Style (match Figure 5.1)
# ============================================================

def paper_style():
    mpl.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "lines.linewidth": 2.0,
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "figure.constrained_layout.use": True,
    })


# ============================================================
# Instance generation (N-marginal)
# ============================================================

def normalize_cost_operator(H: np.ndarray) -> np.ndarray:
    """Shift + scale Hermitian H so spectrum is roughly in [0,1]."""
    H = hermitianize(H)
    evals = np.linalg.eigvalsh(H)
    lam_min = float(evals.min())
    lam_max = float(evals.max())
    H2 = hermitianize(H - lam_min * np.eye(H.shape[0], dtype=complex))
    denom = max(lam_max - lam_min, 1e-12)
    return hermitianize(H2 / denom)


def make_instance_N(
    d: int,
    N: int,
    seed: int,
    *,
    H_kind: str = "random",
    H_scale: float = 1.0,
    marginal_kind: str = "medium",
    hard_delta: float = 1e-4,
    normalize_cost: bool = False,
):
    rng = np.random.default_rng(seed)
    dims = [int(d)] * int(N)

    hk = H_kind.lower().strip()
    if hk == "random":
        H = gen_H_random(dims, rng, scale=H_scale)
    elif hk == "commuting":
        H = gen_H_commuting(dims, rng, scale=H_scale)
    elif hk in {"conjugated", "conjugated_from_diagonal", "conjugated_diagonal"}:
        H = gen_H_conjugated_from_diagonal(dims, rng, scale=H_scale)
        hk = "conjugated"
    else:
        raise ValueError(f"Unknown H_kind='{H_kind}'")

    gammas = [gen_marginal(d, rng, kind=marginal_kind, hard_delta=hard_delta) for _ in range(N)]

    if normalize_cost:
        H = normalize_cost_operator(H)

    return H, gammas, dims, hk


# ============================================================
# Utilities: Gibbs calls to tolerance
# ============================================================

def first_gibbs_to_tol(res, tau: float):
    """
    Compute first Gibbs call count where F_list <= tau.
    Returns (gibbs_to_tol, success, k_index).
    If never reaches tau: (nan, False, -1).
    """
    F = np.asarray(res.F_list, dtype=float)
    if F.size == 0:
        return (np.nan, False, -1)

    # Find first index meeting tolerance
    idx = np.where(F <= tau)[0]
    if idx.size == 0:
        return (np.nan, False, -1)
    k = int(idx[0])

    # Map iteration index to Gibbs calls
    if hasattr(res, "gibbs_calls_list") and res.gibbs_calls_list is not None and len(res.gibbs_calls_list) == len(F):
        g_list = np.asarray(res.gibbs_calls_list, dtype=float)
        g = float(g_list[k])
    else:
        # Fallback: linear interpolation between 0 and total gibbs calls
        total = float(getattr(res, "gibbs_calls", 0) or 0)
        if len(F) == 1:
            g = max(1.0, total)
        else:
            g = float(np.linspace(0.0, total, num=len(F))[k])
        g = max(1.0, g)

    return (g, True, k)


# ============================================================
# Run one configuration
# ============================================================

def run_one(args, d: int, N: int, eps: float, seed: int):
    H, gammas, dims, hk = make_instance_N(
        d=d, N=N, seed=seed,
        H_kind=args.H_kind,
        H_scale=args.H_scale,
        marginal_kind=args.marginal_kind,
        hard_delta=args.hard_delta,
        normalize_cost=args.normalize_cost,
    )

    rows = []

    # --- KL descent ---
    t0 = time.perf_counter()
    res_kl = potential_marginal_kl_descent(
        H, gammas, eps, dims,
        T=args.T_kl,
        eta=args.eta_kl,          # None -> default 1/N inside solver
        tol_tr=None,              # we are benchmarking by F_marg tolerance
        store_hist=False,
        project_pi=True,
    )
    t1 = time.perf_counter()
    g_tol, ok, k = first_gibbs_to_tol(res_kl, args.tau)
    rows.append({
        "algo": "KL",
        "d": d, "N": N, "eps": eps, "seed": seed,
        "tau": args.tau,
        "H_kind": hk,
        "normalize_cost": int(args.normalize_cost),
        "T_budget": args.T_kl,
        "gibbs_total": int(res_kl.gibbs_calls),
        "time_total_sec": float(t1 - t0),
        "success": int(ok),
        "gibbs_to_tol": float(g_tol),
        "iter_index_to_tol": int(k),
    })

    # --- MD-Sinkhorn (choose one M_inner for sweep; default M=5) ---
    t0 = time.perf_counter()
    res_md = md_type_sinkhorn_potential(
        H, gammas, eps, dims,
        T_outer=args.T_md,
        tol_tr=None,
        jitter=args.jitter,
        eta_inner=args.eta_inner,
        M_inner=args.M_inner,
        project_pi=True,
        keep_U_hist=False,
        keep_pi_hist=False,
    )
    t1 = time.perf_counter()
    g_tol, ok, k = first_gibbs_to_tol(res_md, args.tau)
    rows.append({
        "algo": f"MD(M={args.M_inner})",
        "d": d, "N": N, "eps": eps, "seed": seed,
        "tau": args.tau,
        "H_kind": hk,
        "normalize_cost": int(args.normalize_cost),
        "T_budget": args.T_md,
        "gibbs_total": int(res_md.gibbs_calls),
        "time_total_sec": float(t1 - t0),
        "success": int(ok),
        "gibbs_to_tol": float(g_tol),
        "iter_index_to_tol": int(k),
    })

    return rows


# ============================================================
# Summaries + plotting (Option A)
# ============================================================

def summarize(df: pd.DataFrame):
    """
    Summary table:
      median gibbs_to_tol (over successful runs)
      median time_total_sec
      success rate
      also include median total gibbs
    """
    def med_success(x):
        xs = x[np.isfinite(x)]
        if xs.size == 0:
            return np.nan
        return float(np.median(xs))

    grp_cols = ["algo", "d", "N", "eps", "tau", "H_kind", "normalize_cost"]
    g = df.groupby(grp_cols, dropna=False)

    out = g.apply(lambda sub: pd.Series({
        "n_runs": int(len(sub)),
        "success_rate": float(sub["success"].mean()),
        "median_gibbs_to_tol": med_success(sub.loc[sub["success"] == 1, "gibbs_to_tol"].to_numpy()),
        "median_time_sec": med_success(sub.loc[sub["success"] == 1, "time_total_sec"].to_numpy()),
        "median_gibbs_total": float(np.median(sub["gibbs_total"].to_numpy())),
    })).reset_index()

    return out


def plot_option_A(summary: pd.DataFrame, out_dir: str, title: str):
    """
    Option A:
      Panels by N.
      x = log10(eps)
      y = log10(median_gibbs_to_tol)
      color = d
      linestyle = algo
    """
    paper_style()
    os.makedirs(out_dir, exist_ok=True)

    # Filter successful only (otherwise log10 undefined)
    S = summary.copy()
    S = S[np.isfinite(S["median_gibbs_to_tol"])]

    Ns = sorted(S["N"].unique().tolist())
    ds = sorted(S["d"].unique().tolist())
    algos = S["algo"].unique().tolist()

    # Consistent palette across panels
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    d_to_color = {d: colors[i % len(colors)] for i, d in enumerate(ds)}

    # Linestyle map for algo
    # KL: solid, MD: dashed
    ls_map = {}
    for a in algos:
        if a == "KL":
            ls_map[a] = "-"
        else:
            ls_map[a] = (0, (5, 2))

    ncols = len(Ns)
    fig, axes = plt.subplots(1, ncols, figsize=(6.4 * ncols, 4.6), sharey=True)

    if ncols == 1:
        axes = [axes]

    for ax, N in zip(axes, Ns):
        SN = S[S["N"] == N]
        for d in ds:
            SNd = SN[SN["d"] == d]
            for a in algos:
                Sa = SNd[SNd["algo"] == a].sort_values("eps")
                if Sa.empty:
                    continue
                x = np.log10(Sa["eps"].to_numpy(dtype=float))
                y = np.log10(Sa["median_gibbs_to_tol"].to_numpy(dtype=float))
                ax.plot(
                    x, y,
                    color=d_to_color[d],
                    linestyle=ls_map[a],
                )

        ax.set_title(rf"$N={N}$")
        ax.set_xlabel(r"$\log_{10}\varepsilon$")
        ax.grid(False)

    axes[0].set_ylabel(r"$\log_{10}(\mathrm{Gibbs\ calls\ to\ reach}\ F_{\mathrm{marg}}\le \tau)$")
    fig.suptitle(title)

    # Two compact legends: colors for d, linestyles for algo
    # Legend 1: d
    d_handles = []
    for d in ds:
        h, = axes[0].plot([], [], color=d_to_color[d], linestyle="-")
        d_handles.append(h)
    leg1 = axes[0].legend(d_handles, [rf"$d={d}$" for d in ds], loc="lower left", frameon=True, framealpha=0.95)
    axes[0].add_artist(leg1)

    # Legend 2: algo
    a_handles = []
    for a in algos:
        h, = axes[0].plot([], [], color="0.2", linestyle=ls_map[a])
        a_handles.append(h)
    axes[0].legend(a_handles, algos, loc="lower right", frameon=True, framealpha=0.95)

    pdf_path = os.path.join(out_dir, "optionA_gibbs_to_tol.pdf")
    png_path = os.path.join(out_dir, "optionA_gibbs_to_tol.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[Saved] {pdf_path}")
    print(f"[Saved] {png_path}")


# ============================================================
# Main
# ============================================================

def parse_list_int(s: str):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_list_float(s: str):
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def main():
    p = argparse.ArgumentParser("Sweep Gibbs calls to reach F_marg tolerance for KL vs MD (multi-marginal).")

    # Sweep grids
    p.add_argument("--eps_list", type=str, default="0.1,0.2,0.5,1.0",
                   help="Comma-separated eps values.")
    p.add_argument("--d_list", type=str, default="2,3,4,6",
                   help="Comma-separated local dimensions d.")
    p.add_argument("--N_list", type=str, default="3,4,5,8",
                   help="Comma-separated marginal numbers N.")
    p.add_argument("--seed_list", type=str, default="0,1,2",
                   help="Comma-separated seeds.")

    # Fixed tolerance
    p.add_argument("--tau", type=float, default=1e-8)

    # Instance params
    p.add_argument("--H_kind", type=str, default="random", help="random/commuting/conjugated_from_diagonal")
    p.add_argument("--H_scale", type=float, default=1.0)
    p.add_argument("--marginal_kind", type=str, default="medium")
    p.add_argument("--hard_delta", type=float, default=1e-4)
    p.add_argument("--normalize_cost", action="store_true")

    # Budgets
    p.add_argument("--T_kl", type=int, default=4000)
    p.add_argument("--T_md", type=int, default=4000)

    # KL hyperparam (not shown in figure)
    p.add_argument("--eta_kl", type=float, default=None)

    # MD hyperparams (sweep uses a fixed inner count)
    p.add_argument("--M_inner", type=int, default=5)
    p.add_argument("--eta_inner", type=float, default=1.0)
    p.add_argument("--jitter", type=float, default=1e-12)

    # Output
    p.add_argument("--out_dir", type=str, default="experiments/figures/sweep_tol")
    args = p.parse_args()

    eps_list = parse_list_float(args.eps_list)
    d_list = parse_list_int(args.d_list)
    N_list = parse_list_int(args.N_list)
    seed_list = parse_list_int(args.seed_list)

    os.makedirs(args.out_dir, exist_ok=True)

    # Run sweep
    all_rows = []
    total_jobs = len(eps_list) * len(d_list) * len(N_list) * len(seed_list)
    job_id = 0

    for N in N_list:
        for d in d_list:
            for eps in eps_list:
                for seed in seed_list:
                    job_id += 1
                    print(f"[{job_id}/{total_jobs}] Run d={d}, N={N}, eps={eps:g}, seed={seed} ...")
                    rows = run_one(args, d=d, N=N, eps=eps, seed=seed)
                    all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    per_run_csv = os.path.join(args.out_dir, "per_run.csv")
    df.to_csv(per_run_csv, index=False)
    print(f"[Saved] {per_run_csv}")

    # Summary table
    summary = summarize(df)
    summary_csv = os.path.join(args.out_dir, "summary.csv")
    summary.to_csv(summary_csv, index=False)
    print(f"[Saved] {summary_csv}")

    # Also print a compact table to stdout (median over eps aggregated optional)
    # Here: show a readable pivot for each N, eps with d lines (kept minimal)
    print("\n[Summary head]")
    print(summary.head(12).to_string(index=False))

    # Plot Option A
    title = rf"Complexity to reach $F_{{\mathrm{{marg}}}}\le {args.tau:g}$ (KL vs MD, {args.H_kind}, norm={int(args.normalize_cost)})"
    plot_option_A(summary, args.out_dir, title)


if __name__ == "__main__":
    main()
