"""
experiments/fig5_2_multimarginal_gibbs_to_tol.py

Multi-marginal scaling experiment (journal-style):
  - Fix d=3, eps=1
  - Sweep N in {3,4,5,6}
  - Sweep seeds
  - Compare ONLY our solvers: KL-descent vs MD-Sinkhorn (fixed M_inner)
  - Metric for stopping: F_marg <= tau (tau=1e-8 by default)
  - Output per-run CSV + aggregated summary CSV
  - Generate journal-style line plots:
      (i) Gibbs calls to reach tolerance vs N
     (ii) Wall-clock time to reach tolerance vs N   (recommended)

Compatibility: Python 3.8+
"""

from __future__ import print_function

import os
import sys
import time
import csv
import argparse
from typing import List, Dict, Tuple

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


# ----------------------------
# Robust imports: add repo root to sys.path
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
# Plot style (journal-friendly)
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
# Helpers: parsing
# ============================================================

def parse_int_range(spec: str) -> List[int]:
    """
    Accept:
      "3-6"  -> [3,4,5,6]
      "3,5"  -> [3,5]
    """
    s = spec.strip()
    if "-" in s and "," not in s:
        a, b = s.split("-")
        a, b = int(a), int(b)
        if a <= b:
            return list(range(a, b + 1))
        return list(range(a, b - 1, -1))
    return [int(x.strip()) for x in s.split(",") if x.strip()]


# ============================================================
# Instance generation
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


def make_instance(d: int, N: int, seed: int, *,
                  H_kind: str,
                  H_scale: float,
                  marginal_kind: str,
                  hard_delta: float,
                  normalize_cost: bool):
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
        raise ValueError("Unknown H_kind='{}'".format(H_kind))

    gammas = [gen_marginal(d, rng, kind=marginal_kind, hard_delta=hard_delta) for _ in range(N)]

    if normalize_cost:
        H = normalize_cost_operator(H)

    return H, gammas, dims, hk


# ============================================================
# Tolerance extraction: gibbs_to_tol & time_to_tol using F_marg
# ============================================================

def first_hit_index(arr: np.ndarray, thresh: float) -> int:
    """Return first index i with arr[i] <= thresh; if none, return -1."""
    idx = np.where(arr <= thresh)[0]
    if idx.size == 0:
        return -1
    return int(idx[0])


def extract_to_tol(res, tau: float) -> Dict[str, float]:
    """
    Uses res.F_list trajectory and res.gibbs_calls_list/times to compute:
      - gibbs_to_tol
      - time_to_tol
    Falls back to final values if tau not reached.
    """
    F = np.asarray(getattr(res, "F_list", []), dtype=float)
    t = np.asarray(getattr(res, "times", []), dtype=float)
    g_list = getattr(res, "gibbs_calls_list", None)
    g_list = np.asarray(g_list, dtype=float) if g_list is not None else None

    final_g = float(getattr(res, "gibbs_calls", 0) or 0)
    final_t = float(t[-1]) if t.size > 0 else 0.0
    final_F = float(F[-1]) if F.size > 0 else float("inf")

    if F.size == 0:
        return {
            "hit": 0.0,
            "gibbs_to_tol": final_g,
            "time_to_tol": final_t,
            "final_F": final_F,
        }

    i = first_hit_index(F, float(tau))
    if i < 0:
        return {
            "hit": 0.0,
            "gibbs_to_tol": final_g,
            "time_to_tol": final_t,
            "final_F": final_F,
        }

    # hit
    if g_list is not None and g_list.size == F.size:
        g_to = float(g_list[i])
    else:
        # fallback: proportional proxy if list missing
        g_to = float(final_g * (i / max(1, F.size - 1)))

    if t.size == F.size:
        t_to = float(t[i])
    else:
        t_to = float(final_t * (i / max(1, F.size - 1)))

    return {
        "hit": 1.0,
        "gibbs_to_tol": g_to,
        "time_to_tol": t_to,
        "final_F": final_F,
    }


# ============================================================
# Aggregation: mean/std and median/IQR
# ============================================================

def mean_std(vals: np.ndarray) -> Tuple[float, float]:
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return (float("nan"), float("nan"))
    if vals.size == 1:
        return (float(vals[0]), 0.0)
    return (float(vals.mean()), float(vals.std(ddof=1)))


def median_iqr(vals: np.ndarray) -> Tuple[float, float, float]:
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    q1 = float(np.quantile(vals, 0.25))
    med = float(np.quantile(vals, 0.50))
    q3 = float(np.quantile(vals, 0.75))
    return (med, q1, q3)


def aggregate_by_N(per_run: List[Dict], metric_key: str) -> List[Dict]:
    """
    Group by (algo_label, N) and aggregate metric_key across seeds.
    """
    groups = {}
    for r in per_run:
        lab = r["label"]
        N = int(r["N"])
        groups.setdefault((lab, N), []).append(float(r[metric_key]))

    out = []
    for (lab, N), vals in groups.items():
        arr = np.asarray(vals, dtype=float)
        mu, sd = mean_std(arr)
        med, q1, q3 = median_iqr(arr)
        out.append({
            "label": lab,
            "N": N,
            "n": int(np.isfinite(arr).sum()),
            metric_key + "_mean": mu,
            metric_key + "_std": sd,
            metric_key + "_median": med,
            metric_key + "_q1": q1,
            metric_key + "_q3": q3,
        })

    out.sort(key=lambda rr: (rr["label"], rr["N"]))
    return out


# ============================================================
# CSV
# ============================================================

def write_csv(path: str, rows: List[Dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ============================================================
# Plotting (journal-style lines)
# ============================================================

def plot_line(summary: List[Dict], *, y_mean_key: str, y_std_key: str,
              out_pdf: str, out_png: str,
              title: str, ylabel: str,
              N_list: List[int], ylog: bool = False):
    paper_style()
    fig, ax = plt.subplots(figsize=(6.6, 4.6))

    labels = sorted({r["label"] for r in summary})
    for lab in labels:
        pts = [r for r in summary if r["label"] == lab]
        pts.sort(key=lambda rr: rr["N"])
        x = np.array([rr["N"] for rr in pts], dtype=float)
        y = np.array([rr[y_mean_key] for rr in pts], dtype=float)
        e = np.array([rr[y_std_key] for rr in pts], dtype=float)

        ax.plot(x, y, marker="o", label=lab)
        ax.fill_between(x, y - e, y + e, alpha=0.18)

    ax.set_xlabel(r"Number of marginals $N$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(N_list)
    if ylog:
        ax.set_yscale("log")
    ax.legend(loc="best", frameon=True, framealpha=0.95)

    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[Saved] {}".format(out_pdf))
    print("[Saved] {}".format(out_png))


# ============================================================
# Run solvers
# ============================================================

def run_KL(H, gammas, eps, dims, args):
    # tol_tr is trace-norm stopping; we disable it and use F_marg tolerance externally
    tol_tr_disabled = -1.0
    t0 = time.perf_counter()
    res = potential_marginal_kl_descent(
        H, gammas, eps, dims,
        T=args.T_kl,
        eta=args.eta_kl,
        tol_tr=tol_tr_disabled,
        store_hist=False,
        project_pi=True,
    )
    t1 = time.perf_counter()
    # prefer solver times; but keep wall clock too (debug)
    wall = float(t1 - t0)
    return res, wall


def run_MD(H, gammas, eps, dims, args):
    tol_tr_disabled = -1.0
    t0 = time.perf_counter()
    res = md_type_sinkhorn_potential(
        H, gammas, eps, dims,
        T_outer=args.T_md,
        tol_tr=tol_tr_disabled,
        jitter=args.jitter,
        eta_inner=args.eta_inner,
        M_inner=int(args.M_inner),
        project_pi=True,
        keep_U_hist=False,
        keep_pi_hist=False,
    )
    t1 = time.perf_counter()
    wall = float(t1 - t0)
    return res, wall


# ============================================================
# Main
# ============================================================

def main():
    p = argparse.ArgumentParser("Multi-marginal scaling: KL vs MD, gibbs_to_tol vs N (journal style).")

    # Fixed params (as requested)
    p.add_argument("--d", type=int, default=3)
    p.add_argument("--eps", type=float, default=1.0)
    p.add_argument("--N_list", type=str, default="3-6")
    p.add_argument("--seed_list", type=str, default="0,1,2")

    # Tolerance (on F_marg)
    p.add_argument("--tau", type=float, default=1e-8)

    # Instance params
    p.add_argument("--H_kind", type=str, default="random", help="random/commuting/conjugated_from_diagonal")
    p.add_argument("--H_scale", type=float, default=1.0)
    p.add_argument("--marginal_kind", type=str, default="medium")
    p.add_argument("--hard_delta", type=float, default=1e-4)
    p.add_argument("--normalize_cost", action="store_true")

    # Algorithm budgets
    p.add_argument("--T_kl", type=int, default=3000)
    p.add_argument("--T_md", type=int, default=3000)

    # KL hyperparam (None means internal default)
    p.add_argument("--eta_kl", type=float, default=None)

    # MD hyperparams
    p.add_argument("--M_inner", type=int, default=2)
    p.add_argument("--eta_inner", type=float, default=1.0)
    p.add_argument("--jitter", type=float, default=1e-12)

    # Plot options
    p.add_argument("--ylog", action="store_true", help="Use log scale for y-axis in plots")

    # Output
    p.add_argument("--out_dir", type=str, default="experiments/figures/multimarginal_scaling_N")

    args = p.parse_args()

    d = int(args.d)
    eps = float(args.eps)
    tau = float(args.tau)
    N_list = parse_int_range(args.N_list)
    seed_list = parse_int_range(args.seed_list)

    os.makedirs(args.out_dir, exist_ok=True)

    per_run: List[Dict] = []

    for N in N_list:
        for seed in seed_list:
            print("[Run] N={}, d={}, eps={}, seed={} ...".format(N, d, eps, seed), flush=True)

            H, gammas, dims, hk = make_instance(
                d=d, N=int(N), seed=int(seed),
                H_kind=args.H_kind, H_scale=args.H_scale,
                marginal_kind=args.marginal_kind, hard_delta=args.hard_delta,
                normalize_cost=args.normalize_cost,
            )

            # KL
            res_kl, wall_kl = run_KL(H, gammas, eps, dims, args)
            to_kl = extract_to_tol(res_kl, tau=tau)
            per_run.append({
                "label": "KL descent",
                "algo": "KL",
                "M_inner": "",
                "N": int(N),
                "d": d,
                "eps": eps,
                "seed": int(seed),
                "tau": tau,
                "H_kind": hk,
                "normalize_cost": int(args.normalize_cost),
                "hit": int(to_kl["hit"]),
                "gibbs_to_tol": float(to_kl["gibbs_to_tol"]),
                "time_to_tol": float(to_kl["time_to_tol"]),
                "final_F": float(to_kl["final_F"]),
                "final_gibbs": int(getattr(res_kl, "gibbs_calls", 0) or 0),
                "final_time": float(res_kl.times[-1]) if getattr(res_kl, "times", None) else 0.0,
                "wall_clock_sec": float(wall_kl),
            })

            # MD
            res_md, wall_md = run_MD(H, gammas, eps, dims, args)
            to_md = extract_to_tol(res_md, tau=tau)
            per_run.append({
                "label": "MD-Sinkhorn (M={})".format(int(args.M_inner)),
                "algo": "MD",
                "M_inner": int(args.M_inner),
                "N": int(N),
                "d": d,
                "eps": eps,
                "seed": int(seed),
                "tau": tau,
                "H_kind": hk,
                "normalize_cost": int(args.normalize_cost),
                "hit": int(to_md["hit"]),
                "gibbs_to_tol": float(to_md["gibbs_to_tol"]),
                "time_to_tol": float(to_md["time_to_tol"]),
                "final_F": float(to_md["final_F"]),
                "final_gibbs": int(getattr(res_md, "gibbs_calls", 0) or 0),
                "final_time": float(res_md.times[-1]) if getattr(res_md, "times", None) else 0.0,
                "wall_clock_sec": float(wall_md),
            })

    # Save per-run CSV
    per_run_csv = os.path.join(args.out_dir, "per_run.csv")
    write_csv(per_run_csv, per_run)
    print("[Saved] {}".format(per_run_csv))

    # Aggregate summaries
    summary_g = aggregate_by_N(per_run, metric_key="gibbs_to_tol")
    summary_t = aggregate_by_N(per_run, metric_key="time_to_tol")

    summary_csv = os.path.join(args.out_dir, "summary_by_N.csv")
    # merge rows by (label,N) for both metrics
    merged = {}
    for r in summary_g:
        merged[(r["label"], r["N"])] = dict(r)
    for r in summary_t:
        key = (r["label"], r["N"])
        base = merged.get(key, {"label": r["label"], "N": r["N"], "n": r["n"]})
        base.update(r)
        merged[key] = base
    merged_rows = list(merged.values())
    merged_rows.sort(key=lambda rr: (rr["label"], rr["N"]))

    write_csv(summary_csv, merged_rows)
    print("[Saved] {}".format(summary_csv))

    # Plot: Gibbs to tol vs N (mean±std)
    out_g_pdf = os.path.join(args.out_dir, "fig_multimarginal_gibbs_to_tol.pdf")
    out_g_png = os.path.join(args.out_dir, "fig_multimarginal_gibbs_to_tol.png")
    title = r"Multi-marginal scaling ($d={}$, $\varepsilon={}$, $\tau=10^{{{}}}$)".format(
        d, eps, int(np.log10(tau)) if tau > 0 else -8
    )
    plot_line(
        summary=merged_rows,
        y_mean_key="gibbs_to_tol_mean",
        y_std_key="gibbs_to_tol_std",
        out_pdf=out_g_pdf,
        out_png=out_g_png,
        title=title,
        ylabel=r"Gibbs calls to reach $F_{\mathrm{marg}}\leq \tau$",
        N_list=N_list,
        ylog=args.ylog,
    )

    # Plot: time to tol vs N (mean±std) — recommended
    out_t_pdf = os.path.join(args.out_dir, "fig_multimarginal_time_to_tol.pdf")
    out_t_png = os.path.join(args.out_dir, "fig_multimarginal_time_to_tol.png")
    plot_line(
        summary=merged_rows,
        y_mean_key="time_to_tol_mean",
        y_std_key="time_to_tol_std",
        out_pdf=out_t_pdf,
        out_png=out_t_png,
        title=title,
        ylabel=r"Wall-clock time to reach $F_{\mathrm{marg}}\leq \tau$ (sec)",
        N_list=N_list,
        ylog=args.ylog,
    )


if __name__ == "__main__":
    main()
