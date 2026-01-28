"""
experiments/fig5_2_multimarginal_gibbs_to_tol.py

Multi-marginal scaling experiment (journal-style):
  - Fix d=3, eps=1
  - Sweep N in {3,4,5,6}
  - Sweep seeds
  - Compare ONLY our solvers: KL-descent vs MD-Sinkhorn (fixed M_inner)
  - Metric for stopping: F_marg <= tau (tau=1e-8 by default)
  - Output per-run CSV + aggregated summary CSV
  - Generate journal-style line plots (linear + log y):
      (i) Gibbs calls to reach tolerance vs N
     (ii) Wall-clock time to reach tolerance vs N
  - Annotate hit-rate per point (hit/total across seeds)

Compatibility: Python 3.8+
"""

from __future__ import print_function

import os
import sys
import csv
import argparse
from typing import List, Dict, Tuple

import numpy as np

# ===============================
# IMPORTANT: non-interactive backend for HPC
# ===============================
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt

# ----------------------------
# Robust imports: add repo root
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
        step = 1 if a <= b else -1
        return list(range(a, b + step, step))
    return [int(x.strip()) for x in s.split(",") if x.strip()]


# ============================================================
# Instance generation
# ============================================================

def normalize_cost_operator(H: np.ndarray) -> np.ndarray:
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
    else:
        H = gen_H_conjugated_from_diagonal(dims, rng, scale=H_scale)
        hk = "conjugated"

    gammas = [gen_marginal(d, rng, kind=marginal_kind, hard_delta=hard_delta) for _ in range(N)]

    if normalize_cost:
        H = normalize_cost_operator(H)

    return H, gammas, dims, hk


# ============================================================
# Tolerance extraction
# ============================================================

def first_hit_index(arr: np.ndarray, thresh: float) -> int:
    idx = np.where(arr <= thresh)[0]
    return int(idx[0]) if idx.size > 0 else -1


def extract_to_tol(res, tau: float) -> Dict[str, float]:
    """
    Uses res.F_list trajectory and res.gibbs_calls_list/times (if present).
    Returns:
      hit in {0,1}, gibbs_to_tol, time_to_tol, final_F
    """
    F = np.asarray(getattr(res, "F_list", []), dtype=float)
    t = np.asarray(getattr(res, "times", []), dtype=float)
    g = np.asarray(getattr(res, "gibbs_calls_list", []), dtype=float)

    final_g = float(getattr(res, "gibbs_calls", 0) or 0)
    final_t = float(t[-1]) if t.size > 0 else 0.0
    final_F = float(F[-1]) if F.size > 0 else float("inf")

    if F.size == 0:
        return dict(hit=0, gibbs_to_tol=final_g, time_to_tol=final_t, final_F=final_F)

    i = first_hit_index(F, tau)
    if i < 0:
        return dict(hit=0, gibbs_to_tol=final_g, time_to_tol=final_t, final_F=final_F)

    g_to = float(g[i]) if g.size == F.size else final_g * (i / max(1, F.size - 1))
    t_to = float(t[i]) if t.size == F.size else final_t * (i / max(1, F.size - 1))

    return dict(hit=1, gibbs_to_tol=g_to, time_to_tol=t_to, final_F=final_F)


# ============================================================
# CSV utilities
# ============================================================

def write_csv(path: str, rows: List[Dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# ============================================================
# Aggregation (mean/std + hit-rate)
# ============================================================

def mean_std(vals: np.ndarray) -> Tuple[float, float]:
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan")
    if vals.size == 1:
        return float(vals[0]), 0.0
    return float(vals.mean()), float(vals.std(ddof=1))


def aggregate_by_N(per_run: List[Dict], key: str) -> List[Dict]:
    """
    Group by (label, N) and aggregate:
      - key_mean, key_std
      - n_runs
      - hit_count, hit_rate
    """
    groups: Dict[Tuple[str, int], List[Dict]] = {}
    for r in per_run:
        groups.setdefault((r["label"], int(r["N"])), []).append(r)

    out: List[Dict] = []
    for (lab, N), rows in groups.items():
        vals = np.asarray([float(rr[key]) for rr in rows], dtype=float)
        mu, sd = mean_std(vals)

        hits = np.asarray([int(rr.get("hit", 0)) for rr in rows], dtype=int)
        n_runs = int(len(rows))
        hit_count = int(hits.sum())
        hit_rate = float(hit_count) / float(n_runs) if n_runs > 0 else float("nan")

        out.append({
            "label": lab,
            "N": int(N),
            "n_runs": n_runs,
            "hit_count": hit_count,
            "hit_rate": hit_rate,
            f"{key}_mean": mu,
            f"{key}_std": sd,
        })

    out.sort(key=lambda rr: (rr["label"], rr["N"]))
    return out


# ============================================================
# Plotting (lines + error band + hit-rate annotation)
# ============================================================

def plot_scaling(summary: List[Dict], *, key: str, out_pdf: str, ylog: bool,
                 ylabel: str, title: str):
    """
    key: "gibbs_to_tol" or "time_to_tol"
    summary rows contain: key_mean, key_std, n_runs, hit_count
    """
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    labels = sorted({r["label"] for r in summary})

    for lab in labels:
        pts = [r for r in summary if r["label"] == lab]
        pts.sort(key=lambda rr: rr["N"])

        x = np.array([r["N"] for r in pts], dtype=float)
        y = np.array([r[f"{key}_mean"] for r in pts], dtype=float)
        e = np.array([r[f"{key}_std"] for r in pts], dtype=float)

        ax.plot(x, y, marker="o", label=lab)
        ax.fill_between(x, y - e, y + e, alpha=0.18)

        # annotate hit-rate as "hit/total" above each marker
        for r, xi, yi in zip(pts, x, y):
            hc = int(r.get("hit_count", 0))
            nr = int(r.get("n_runs", 0))

            # Only annotate non-100% hit rate
            if nr == 0 or hc == nr:
                continue

            txt = f"{hc}/{nr}"

            if ylog and np.isfinite(yi) and yi > 0:
                y_text = yi * 1.12
            else:
                y_text = yi + (0.02 * yi if np.isfinite(yi) else 0.0)

            ax.text(xi, y_text, txt, ha="center", va="bottom", fontsize=8)
    ax.set_xlabel(r"Number of marginals $N$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(sorted({int(r["N"]) for r in summary}))
    ax.legend(loc="best", frameon=True, framealpha=0.95)

    if ylog:
        ax.set_yscale("log")

    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {out_pdf}", flush=True)


# ============================================================
# Main
# ============================================================

def main():
    p = argparse.ArgumentParser("Multi-marginal scaling: KL vs MD")

    p.add_argument("--d", type=int, default=3)
    p.add_argument("--eps", type=float, default=1.0)
    p.add_argument("--N_list", type=str, default="3-6")
    p.add_argument("--seed_list", type=str, default="0,1,2")
    p.add_argument("--tau", type=float, default=1e-8)

    p.add_argument("--H_kind", type=str, default="random")
    p.add_argument("--H_scale", type=float, default=1.0)
    p.add_argument("--marginal_kind", type=str, default="medium")
    p.add_argument("--hard_delta", type=float, default=1e-4)
    p.add_argument("--normalize_cost", action="store_true")

    p.add_argument("--T_kl", type=int, default=3000)
    p.add_argument("--T_md", type=int, default=3000)
    p.add_argument("--eta_kl", type=float, default=None)
    p.add_argument("--M_inner", type=int, default=2)
    p.add_argument("--eta_inner", type=float, default=1.0)
    p.add_argument("--jitter", type=float, default=1e-12)

    p.add_argument("--out_dir", type=str, default="experiments/figures/multimarginal_scaling_N")

    args = p.parse_args()
    
    tag = f"d{args.d}"

    N_list = parse_int_range(args.N_list)
    seed_list = parse_int_range(args.seed_list)

    os.makedirs(args.out_dir, exist_ok=True)

    per_run: List[Dict] = []

    for N in N_list:
        for seed in seed_list:
            print(f"[Run] N={N}, seed={seed}", flush=True)

            H, gammas, dims, hk = make_instance(
                d=args.d, N=N, seed=seed,
                H_kind=args.H_kind, H_scale=args.H_scale,
                marginal_kind=args.marginal_kind,
                hard_delta=args.hard_delta,
                normalize_cost=args.normalize_cost,
            )

            # KL
            res_kl = potential_marginal_kl_descent(
                H, gammas, args.eps, dims,
                T=args.T_kl, eta=args.eta_kl,
                tol_tr=-1.0, store_hist=False, project_pi=True,
            )
            to_kl = extract_to_tol(res_kl, args.tau)

            # MD
            res_md = md_type_sinkhorn_potential(
                H, gammas, args.eps, dims,
                T_outer=args.T_md, tol_tr=-1.0,
                jitter=args.jitter, eta_inner=args.eta_inner,
                M_inner=args.M_inner, project_pi=True,
            )
            to_md = extract_to_tol(res_md, args.tau)

            # store per-run (include seed + metadata for reproducibility)
            per_run.append({
                "label": "KL descent",
                "algo": "KL",
                "N": int(N),
                "d": int(args.d),
                "eps": float(args.eps),
                "seed": int(seed),
                "tau": float(args.tau),
                "H_kind": hk,
                "normalize_cost": int(args.normalize_cost),
                **to_kl,
            })
            per_run.append({
                "label": f"MD-Sinkhorn (M={args.M_inner})",
                "algo": "MD",
                "M_inner": int(args.M_inner),
                "N": int(N),
                "d": int(args.d),
                "eps": float(args.eps),
                "seed": int(seed),
                "tau": float(args.tau),
                "H_kind": hk,
                "normalize_cost": int(args.normalize_cost),
                **to_md,
            })

    # -------------------------
    # Outputs
    # -------------------------
    write_csv(
        os.path.join(args.out_dir, f"per_run_{tag}.csv"),
        per_run
    )
    print(f"[Saved] {os.path.join(args.out_dir, 'per_run.csv')}", flush=True)

    paper_style()

    # summaries (include hit_rate)
    summary_g = aggregate_by_N(per_run, "gibbs_to_tol")
    summary_t = aggregate_by_N(per_run, "time_to_tol")

    write_csv(
        os.path.join(args.out_dir, f"summary_gibbs_to_tol_{tag}.csv"),
        summary_g
    )
    write_csv(
        os.path.join(args.out_dir, f"summary_time_to_tol_{tag}.csv"),
        summary_t
    )

    print("[Saved] summary CSVs", flush=True)

    title = (rf"Multi-marginal scaling ($d={args.d}$, $\varepsilon={args.eps}$, "
             rf"$\tau={args.tau:g}$); annotations: hit/total")

    # ---- Gibbs-to-tol: linear + log
    plot_scaling(
        summary_g, key="gibbs_to_tol",
        out_pdf=os.path.join(args.out_dir, f"fig_multimarginal_gibbs_to_tol_linear_{tag}.pdf")
        ylog=False,
        ylabel=r"Gibbs calls to reach $F_{\mathrm{marg}}\leq \tau$",
        title=title,
    )
    plot_scaling(
        summary_g, key="gibbs_to_tol",
        out_pdf=os.path.join(args.out_dir, "fig_multimarginal_gibbs_to_tol_log_{tag}.pdf"),
        ylog=True,
        ylabel=r"Gibbs calls to reach $F_{\mathrm{marg}}\leq \tau$",
        title=title,
    )

    # ---- Time-to-tol: linear + log
    plot_scaling(
        summary_t, key="time_to_tol",
        out_pdf=os.path.join(args.out_dir, "fig_multimarginal_time_to_tol_linear_{tag}.pdf"),
        ylog=False,
        ylabel=r"Wall-clock time to reach $F_{\mathrm{marg}}\leq \tau$ (sec)",
        title=title,
    )
    plot_scaling(
        summary_t, key="time_to_tol",
        out_pdf=os.path.join(args.out_dir, "fig_multimarginal_time_to_tol_log_{tag}.pdf"),
        ylog=True,
        ylabel=r"Wall-clock time to reach $F_{\mathrm{marg}}\leq \tau$ (sec)",
        title=title,
    )

    print("[Done]", flush=True)


if __name__ == "__main__":
    main()
