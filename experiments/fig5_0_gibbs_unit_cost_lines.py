"""
experiments/fig5_0_gibbs_unit_cost_lines.py

Journal-style Paragraph 1 figures:
Validate "Gibbs calls" as a solver-independent unit of computational cost by comparing
wall-clock time per Gibbs call across solvers, while sweeping:
  (A) d (at fixed N=2, eps=eps_fixed)
  (B) eps (at fixed N=2, d=d_fixed)

For each run:
    time_per_gibbs = runtime_sec / max(gibbs_calls, 1)

We aggregate over seeds and plot mean ± std.

Compatibility:
  - Python 3.8+
  - Avoid tol_tr=None (use tol_tr=-1 to disable stopping)

Outputs (default out_dir = experiments/figures/gibbs_unit_cost):
  per_run_lines.csv
  summary_vs_d.csv
  summary_vs_eps.csv
  fig5_0_time_per_gibbs_vs_d_N2_eps{eps_fixed}.pdf/png
  fig5_0_time_per_gibbs_vs_eps_N2_d{d_fixed}.pdf/png
  fig5_0_time_per_gibbs_panels.pdf/png   (two-panel figure)
"""

from __future__ import print_function

import os
import sys
import time
import csv
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple


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
    dbga_algorithm_2_2,
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
# Parsing helpers
# ============================================================

def parse_int_range(spec: str) -> List[int]:
    """
    Accept:
      "3-10"  -> [3,4,...,10]
      "3,5,7" -> [3,5,7]
    """
    s = spec.strip()
    if "-" in s and "," not in s:
        a, b = s.split("-")
        a, b = int(a), int(b)
        if a <= b:
            return list(range(a, b + 1))
        return list(range(a, b - 1, -1))
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_float_list(spec: str) -> List[float]:
    """
    Accept:
      "0.1,0.3,1.0" -> [0.1,0.3,1.0]
    """
    return [float(x.strip()) for x in spec.split(",") if x.strip()]


def build_eps_list(eps_min: float, eps_max: float, eps_num: int, eps_space: str) -> List[float]:
    if eps_num <= 1:
        return [float(eps_min)]
    if eps_space.lower() == "log":
        return np.logspace(np.log10(eps_min), np.log10(eps_max), eps_num).tolist()
    return np.linspace(eps_min, eps_max, eps_num).tolist()


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
# Run one algorithm and measure time per Gibbs call
# ============================================================

def run_and_measure(algo: str, H, gammas, eps: float, dims, args, *, M_inner: int = 5):
    """
    Return dict:
      runtime_sec, gibbs_total, time_per_gibbs_sec
    """
    t0 = time.perf_counter()

    # Disable stopping-by-trace criterion (solver expects numeric tol_tr)
    tol_tr_disabled = -1.0

    if algo == "BGDA":
        res = dbga_algorithm_2_2(
            H, gammas, eps, dims,
            T=args.T_bgda,
            delta=args.delta_paper,
            gauge_trace0=args.gauge_trace0,
            store_hist=False,
            project_pi=True,
        )
        label = "BGDA (Alg. 2.2)"

    elif algo == "KL":
        res = potential_marginal_kl_descent(
            H, gammas, eps, dims,
            T=args.T_kl,
            eta=args.eta_kl,
            tol_tr=tol_tr_disabled,
            store_hist=False,
            project_pi=True,
        )
        label = "KL descent"

    elif algo == "MD":
        res = md_type_sinkhorn_potential(
            H, gammas, eps, dims,
            T_outer=args.T_md,
            tol_tr=tol_tr_disabled,
            jitter=args.jitter,
            eta_inner=args.eta_inner,
            M_inner=int(M_inner),
            project_pi=True,
            keep_U_hist=False,
            keep_pi_hist=False,
        )
        label = "MD-Sinkhorn (M={})".format(int(M_inner))
    else:
        raise ValueError("algo must be one of {BGDA, KL, MD}")

    t1 = time.perf_counter()

    gibbs_total = int(getattr(res, "gibbs_calls", 0) or 0)
    runtime_sec = float(t1 - t0)
    time_per_gibbs = float(runtime_sec / max(gibbs_total, 1))

    return {
        "algo": algo,
        "label": label,
        "gibbs_total": gibbs_total,
        "runtime_sec": runtime_sec,
        "time_per_gibbs_sec": time_per_gibbs,
    }


# ============================================================
# CSV helpers
# ============================================================

def write_csv(path: str, rows: List[Dict], fieldnames: List[str]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ============================================================
# Aggregation helpers
# ============================================================

def mean_std(vals: np.ndarray) -> Tuple[float, float]:
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return (float("nan"), float("nan"))
    if vals.size == 1:
        return (float(vals[0]), 0.0)
    return (float(vals.mean()), float(vals.std(ddof=1)))


def aggregate_by_x(per_run: List[Dict], *, x_key: str) -> List[Dict]:
    """
    Aggregate time_per_gibbs_sec by (label, x_key) over seeds.
    Returns list of rows with mean/std and n.
    """
    groups = {}
    for r in per_run:
        lab = r["label"]
        x = r[x_key]
        groups.setdefault((lab, x), []).append(r["time_per_gibbs_sec"])

    out = []
    for (lab, x), vals in groups.items():
        arr = np.array(vals, dtype=float)
        mu, sd = mean_std(arr)
        out.append({
            "label": lab,
            x_key: x,
            "mean_time_per_gibbs_sec": mu,
            "std_time_per_gibbs_sec": sd,
            "n": int(np.isfinite(arr).sum()),
        })

    # sort by label then x
    out.sort(key=lambda rr: (rr["label"], rr[x_key]))
    return out


# ============================================================
# Plotting
# ============================================================

def plot_vs_d(summary: List[Dict], *, d_list: List[int], out_pdf: str, out_png: str, title: str):
    paper_style()
    fig, ax = plt.subplots(figsize=(6.4, 4.6))

    labels = sorted({r["label"] for r in summary})
    for lab in labels:
        pts = [r for r in summary if r["label"] == lab]
        pts.sort(key=lambda rr: rr["d"])
        x = np.array([rr["d"] for rr in pts], dtype=float)
        y = np.array([rr["mean_time_per_gibbs_sec"] for rr in pts], dtype=float)
        e = np.array([rr["std_time_per_gibbs_sec"] for rr in pts], dtype=float)

        ax.plot(x, y, marker="o", label=lab)
        ax.fill_between(x, y - e, y + e, alpha=0.18)

    ax.set_xlabel(r"Local dimension $d$")
    ax.set_ylabel("Wall-clock time per Gibbs call (sec)")
    ax.set_title(title)
    ax.set_xticks(d_list)
    ax.legend(loc="best", frameon=True, framealpha=0.95)

    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[Saved] {}".format(out_pdf))
    print("[Saved] {}".format(out_png))


def plot_vs_eps(summary: List[Dict], *, eps_list: List[float], out_pdf: str, out_png: str, title: str):
    paper_style()
    fig, ax = plt.subplots(figsize=(6.4, 4.6))

    labels = sorted({r["label"] for r in summary})
    for lab in labels:
        pts = [r for r in summary if r["label"] == lab]
        pts.sort(key=lambda rr: rr["eps"])
        x = np.array([rr["eps"] for rr in pts], dtype=float)
        y = np.array([rr["mean_time_per_gibbs_sec"] for rr in pts], dtype=float)
        e = np.array([rr["std_time_per_gibbs_sec"] for rr in pts], dtype=float)

        ax.plot(x, y, marker="o", label=lab)
        ax.fill_between(x, y - e, y + e, alpha=0.18)

    ax.set_xscale("log")
    ax.set_xlabel(r"Regularization parameter $\varepsilon$ (log scale)")
    ax.set_ylabel("Wall-clock time per Gibbs call (sec)")
    ax.set_title(title)
    ax.legend(loc="best", frameon=True, framealpha=0.95)

    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[Saved] {}".format(out_pdf))
    print("[Saved] {}".format(out_png))


def plot_panels(figA_path: str, figB_path: str, out_pdf: str, out_png: str,
                titleA: str, titleB: str):
    """
    Create a 1x2 panel figure by re-running plot code is cleaner,
    but here we just rebuild panels from summaries in memory in main().
    This placeholder is unused (kept for clarity).
    """
    pass


# ============================================================
# Main
# ============================================================

def main():
    p = argparse.ArgumentParser("Paragraph-1 journal figures: time per Gibbs call vs d and vs eps (N=2).")

    # Fixed N=2 always here
    p.add_argument("--N", type=int, default=2)

    # Sweep (A): d sweep at fixed eps
    p.add_argument("--d_list", type=str, default="3-10")
    p.add_argument("--eps_fixed", type=float, default=1.0)

    # Sweep (B): eps sweep at fixed d
    p.add_argument("--d_fixed", type=int, default=10)
    p.add_argument("--eps_min", type=float, default=0.1)
    p.add_argument("--eps_max", type=float, default=1.0)
    p.add_argument("--eps_num", type=int, default=5)
    p.add_argument("--eps_space", type=str, default="log", choices=["log", "lin"])

    # Seeds
    p.add_argument("--seed_list", type=str, default="0,1,2")

    # Algorithms
    # For journal clarity, default to one MD curve (M=5). You can pass multiple M's, e.g. "1,5,10".
    p.add_argument("--md_M", type=str, default="5")
    p.add_argument("--include_bgda", action="store_true", help="Include BGDA (Alg. 2.2) in N=2 sweeps")

    # Instance params
    p.add_argument("--H_kind", type=str, default="random", help="random/commuting/conjugated_from_diagonal")
    p.add_argument("--H_scale", type=float, default=1.0)
    p.add_argument("--marginal_kind", type=str, default="medium")
    p.add_argument("--hard_delta", type=float, default=1e-4)
    p.add_argument("--normalize_cost", action="store_true")

    # Budgets (not a convergence experiment; keep moderate but stable)
    p.add_argument("--T_bgda", type=int, default=1500)
    p.add_argument("--T_kl", type=int, default=1500)
    p.add_argument("--T_md", type=int, default=1500)

    # BGDA params
    p.add_argument("--delta_paper", type=float, default=1e-6)
    p.add_argument("--gauge_trace0", action="store_true")

    # KL params
    p.add_argument("--eta_kl", type=float, default=None)

    # MD params
    p.add_argument("--eta_inner", type=float, default=1.0)
    p.add_argument("--jitter", type=float, default=1e-12)

    # Output
    p.add_argument("--out_dir", type=str, default="experiments/figures/gibbs_unit_cost")

    args = p.parse_args()

    N = int(args.N)
    if N != 2:
        raise ValueError("This script is designed for N=2 only (Paragraph 1 figures).")

    d_list = parse_int_range(args.d_list)
    seed_list = parse_int_range(args.seed_list)
    eps_list = build_eps_list(args.eps_min, args.eps_max, args.eps_num, args.eps_space)
    md_M_list = parse_int_range(args.md_M)

    os.makedirs(args.out_dir, exist_ok=True)

    # Decide which solver curves to run
    algos = ["KL"] + ["MD"] * len(md_M_list)
    # We'll label MD by M separately.
    run_bgda = bool(args.include_bgda)

    per_run_rows: List[Dict] = []

    # ----------------------------
    # (A) Sweep d at fixed eps
    # ----------------------------
    epsA = float(args.eps_fixed)
    for d in d_list:
        for seed in seed_list:
            H, gammas, dims, hk = make_instance(
                d=int(d), N=N, seed=int(seed),
                H_kind=args.H_kind, H_scale=args.H_scale,
                marginal_kind=args.marginal_kind, hard_delta=args.hard_delta,
                normalize_cost=args.normalize_cost,
            )

            if run_bgda:
                meas = run_and_measure("BGDA", H, gammas, epsA, dims, args)
                per_run_rows.append({
                    "panel": "vs_d",
                    "label": meas["label"],
                    "algo": meas["algo"],
                    "M_inner": "",
                    "N": N, "d": int(d), "eps": epsA, "seed": int(seed),
                    "H_kind": hk, "normalize_cost": int(args.normalize_cost),
                    "gibbs_total": meas["gibbs_total"],
                    "runtime_sec": meas["runtime_sec"],
                    "time_per_gibbs_sec": meas["time_per_gibbs_sec"],
                })

            meas = run_and_measure("KL", H, gammas, epsA, dims, args)
            per_run_rows.append({
                "panel": "vs_d",
                "label": meas["label"],
                "algo": meas["algo"],
                "M_inner": "",
                "N": N, "d": int(d), "eps": epsA, "seed": int(seed),
                "H_kind": hk, "normalize_cost": int(args.normalize_cost),
                "gibbs_total": meas["gibbs_total"],
                "runtime_sec": meas["runtime_sec"],
                "time_per_gibbs_sec": meas["time_per_gibbs_sec"],
            })

            for M in md_M_list:
                meas = run_and_measure("MD", H, gammas, epsA, dims, args, M_inner=int(M))
                per_run_rows.append({
                    "panel": "vs_d",
                    "label": meas["label"],
                    "algo": meas["algo"],
                    "M_inner": int(M),
                    "N": N, "d": int(d), "eps": epsA, "seed": int(seed),
                    "H_kind": hk, "normalize_cost": int(args.normalize_cost),
                    "gibbs_total": meas["gibbs_total"],
                    "runtime_sec": meas["runtime_sec"],
                    "time_per_gibbs_sec": meas["time_per_gibbs_sec"],
                })

    # ----------------------------
    # (B) Sweep eps at fixed d
    # ----------------------------
    dB = int(args.d_fixed)
    for eps in eps_list:
        eps = float(eps)
        for seed in seed_list:
            H, gammas, dims, hk = make_instance(
                d=dB, N=N, seed=int(seed),
                H_kind=args.H_kind, H_scale=args.H_scale,
                marginal_kind=args.marginal_kind, hard_delta=args.hard_delta,
                normalize_cost=args.normalize_cost,
            )

            if run_bgda:
                meas = run_and_measure("BGDA", H, gammas, eps, dims, args)
                per_run_rows.append({
                    "panel": "vs_eps",
                    "label": meas["label"],
                    "algo": meas["algo"],
                    "M_inner": "",
                    "N": N, "d": dB, "eps": eps, "seed": int(seed),
                    "H_kind": hk, "normalize_cost": int(args.normalize_cost),
                    "gibbs_total": meas["gibbs_total"],
                    "runtime_sec": meas["runtime_sec"],
                    "time_per_gibbs_sec": meas["time_per_gibbs_sec"],
                })

            meas = run_and_measure("KL", H, gammas, eps, dims, args)
            per_run_rows.append({
                "panel": "vs_eps",
                "label": meas["label"],
                "algo": meas["algo"],
                "M_inner": "",
                "N": N, "d": dB, "eps": eps, "seed": int(seed),
                "H_kind": hk, "normalize_cost": int(args.normalize_cost),
                "gibbs_total": meas["gibbs_total"],
                "runtime_sec": meas["runtime_sec"],
                "time_per_gibbs_sec": meas["time_per_gibbs_sec"],
            })

            for M in md_M_list:
                meas = run_and_measure("MD", H, gammas, eps, dims, args, M_inner=int(M))
                per_run_rows.append({
                    "panel": "vs_eps",
                    "label": meas["label"],
                    "algo": meas["algo"],
                    "M_inner": int(M),
                    "N": N, "d": dB, "eps": eps, "seed": int(seed),
                    "H_kind": hk, "normalize_cost": int(args.normalize_cost),
                    "gibbs_total": meas["gibbs_total"],
                    "runtime_sec": meas["runtime_sec"],
                    "time_per_gibbs_sec": meas["time_per_gibbs_sec"],
                })

    # ----------------------------
    # Save per-run CSV
    # ----------------------------
    per_run_csv = os.path.join(args.out_dir, "per_run_lines.csv")
    write_csv(per_run_csv, per_run_rows, fieldnames=list(per_run_rows[0].keys()))
    print("[Saved] {}".format(per_run_csv))

    # ----------------------------
    # Aggregate + plot vs d
    # ----------------------------
    per_run_A = [r for r in per_run_rows if r["panel"] == "vs_d"]
    summary_d = aggregate_by_x(per_run_A, x_key="d")
    summary_d_csv = os.path.join(args.out_dir, "summary_vs_d.csv")
    write_csv(summary_d_csv, summary_d, fieldnames=list(summary_d[0].keys()))
    print("[Saved] {}".format(summary_d_csv))

    outA_pdf = os.path.join(args.out_dir, "fig5_0_time_per_gibbs_vs_d_N2_eps{}.pdf".format(args.eps_fixed))
    outA_png = os.path.join(args.out_dir, "fig5_0_time_per_gibbs_vs_d_N2_eps{}.png".format(args.eps_fixed))
    titleA = r"Time per Gibbs call vs $d$ (N=2, $\varepsilon={}$)".format(args.eps_fixed)
    plot_vs_d(summary_d, d_list=d_list, out_pdf=outA_pdf, out_png=outA_png, title=titleA)

    # ----------------------------
    # Aggregate + plot vs eps
    # ----------------------------
    per_run_B = [r for r in per_run_rows if r["panel"] == "vs_eps"]
    summary_eps = aggregate_by_x(per_run_B, x_key="eps")
    summary_eps_csv = os.path.join(args.out_dir, "summary_vs_eps.csv")
    write_csv(summary_eps_csv, summary_eps, fieldnames=list(summary_eps[0].keys()))
    print("[Saved] {}".format(summary_eps_csv))

    outB_pdf = os.path.join(args.out_dir, "fig5_0_time_per_gibbs_vs_eps_N2_d{}.pdf".format(args.d_fixed))
    outB_png = os.path.join(args.out_dir, "fig5_0_time_per_gibbs_vs_eps_N2_d{}.png".format(args.d_fixed))
    titleB = r"Time per Gibbs call vs $\varepsilon$ (N=2, $d={}$)".format(args.d_fixed)
    plot_vs_eps(summary_eps, eps_list=eps_list, out_pdf=outB_pdf, out_png=outB_png, title=titleB)

    # ----------------------------
    # Two-panel figure (journal-friendly)
    # ----------------------------
    paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))

    # Panel (a): vs d
    ax = axes[0]
    labelsA = sorted({r["label"] for r in summary_d})
    for lab in labelsA:
        pts = [r for r in summary_d if r["label"] == lab]
        pts.sort(key=lambda rr: rr["d"])
        x = np.array([rr["d"] for rr in pts], dtype=float)
        y = np.array([rr["mean_time_per_gibbs_sec"] for rr in pts], dtype=float)
        e = np.array([rr["std_time_per_gibbs_sec"] for rr in pts], dtype=float)
        ax.plot(x, y, marker="o", label=lab)
        ax.fill_between(x, y - e, y + e, alpha=0.18)
    ax.set_xlabel(r"Local dimension $d$")
    ax.set_ylabel("Time per Gibbs call (sec)")
    ax.set_title(r"(a) $N=2$, $\varepsilon={}$".format(args.eps_fixed))
    ax.set_xticks(d_list)

    # Panel (b): vs eps
    ax = axes[1]
    labelsB = sorted({r["label"] for r in summary_eps})
    for lab in labelsB:
        pts = [r for r in summary_eps if r["label"] == lab]
        pts.sort(key=lambda rr: rr["eps"])
        x = np.array([rr["eps"] for rr in pts], dtype=float)
        y = np.array([rr["mean_time_per_gibbs_sec"] for rr in pts], dtype=float)
        e = np.array([rr["std_time_per_gibbs_sec"] for rr in pts], dtype=float)
        ax.plot(x, y, marker="o", label=lab)
        ax.fill_between(x, y - e, y + e, alpha=0.18)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\varepsilon$ (log scale)")
    ax.set_ylabel("Time per Gibbs call (sec)")
    ax.set_title(r"(b) $N=2$, $d={}$".format(args.d_fixed))

    # Shared legend (right panel)
    axes[1].legend(loc="best", frameon=True, framealpha=0.95)

    panels_pdf = os.path.join(args.out_dir, "fig5_0_time_per_gibbs_panels.pdf")
    panels_png = os.path.join(args.out_dir, "fig5_0_time_per_gibbs_panels.png")
    fig.savefig(panels_pdf, bbox_inches="tight")
    fig.savefig(panels_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[Saved] {}".format(panels_pdf))
    print("[Saved] {}".format(panels_png))


if __name__ == "__main__":
    main()
