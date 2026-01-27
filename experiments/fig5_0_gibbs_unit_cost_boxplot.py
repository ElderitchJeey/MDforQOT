"""
experiments/fig5_0_gibbs_unit_cost_boxplot.py

Paragraph 1 figure:
Justify using "Gibbs calls" as a solver-independent complexity unit by showing that
wall-clock time per Gibbs call is comparable across solvers (and MD inner-step counts)
across a range of (N, d, eps).

For each run:
    time_per_gibbs = runtime_sec / max(gibbs_calls, 1)

For each config group (case_id, N, d, eps, seed), normalize:
    rel_time_per_gibbs = time_per_gibbs / median_over_algorithms(time_per_gibbs)

We plot boxplots of rel_time_per_gibbs for each algorithm label.

Cases:
  Case 1: N=2, compare BGDA(Alg 2.2) vs KL vs MD(M in M_list)
  Case 2: N in [3..8], compare KL vs MD(M in M_list)  (no BGDA)

Outputs (default out_dir = experiments/figures/gibbs_unit_cost):
  per_run.csv
  per_config_normalized.csv
  summary.csv
  boxplot_case1_N2.pdf/png
  boxplot_case2_N3to8.pdf/png

Python compatibility:
  - Compatible with Python 3.8+ (NO "int | None" syntax)
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
from typing import Optional, List


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
# Plot style (match your Figure 5.1 style)
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
        "lines.linewidth": 1.8,
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "figure.constrained_layout.use": True,
    })


# ============================================================
# Helpers: parse ranges
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

def run_and_measure(algo: str,
                    H,
                    gammas,
                    eps: float,
                    dims,
                    args,
                    M_inner: Optional[int] = None):
    """
    algo in {"BGDA", "KL", "MD"}.
    Return dict with runtime, gibbs_calls, time_per_gibbs.
    """
    t0 = time.perf_counter()

    # IMPORTANT: tol_tr MUST be numeric for your solver (cannot be None).
    # We set tol_tr = -1.0 to effectively disable early stopping by trace tolerance.
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
            eta=args.eta_kl,          # None -> default 1/N in solver
            tol_tr=tol_tr_disabled,   # disable stopping
            store_hist=False,
            project_pi=True,
        )
        label = "KL descent"

    elif algo == "MD":
        M = int(M_inner) if M_inner is not None else int(args.M_default)
        res = md_type_sinkhorn_potential(
            H, gammas, eps, dims,
            T_outer=args.T_md,
            tol_tr=tol_tr_disabled,   # disable stopping (must be numeric!)
            jitter=args.jitter,
            eta_inner=args.eta_inner,
            M_inner=M,
            project_pi=True,
            keep_U_hist=False,
            keep_pi_hist=False,
        )
        label = "MD-Sinkhorn (M={})".format(M)

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
# CSV writers
# ============================================================

def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ============================================================
# Normalization + summary
# ============================================================

def normalize_within_config(per_run_rows):
    """
    For each group (case_id, N, d, eps, seed), compute median time_per_gibbs across algorithms,
    and return rows with rel_time_per_gibbs = time_per_gibbs / median.
    """
    def key(r):
        return (r["case_id"], r["N"], r["d"], r["eps"], r["seed"])

    groups = {}
    for r in per_run_rows:
        groups.setdefault(key(r), []).append(r)

    out = []
    for _, rows in groups.items():
        vals = np.array([rr["time_per_gibbs_sec"] for rr in rows], dtype=float)
        med = float(np.median(vals)) if vals.size > 0 else np.nan
        if (not np.isfinite(med)) or med <= 0:
            med = np.nan

        for rr in rows:
            rel = float(rr["time_per_gibbs_sec"] / med) if np.isfinite(med) else np.nan
            r2 = dict(rr)
            r2["median_time_per_gibbs_in_group"] = med
            r2["rel_time_per_gibbs"] = rel
            out.append(r2)
    return out


def summarize_by_label(norm_rows):
    """
    Summarize rel_time_per_gibbs by label.
    """
    labels = sorted({r["label"] for r in norm_rows})
    summary = []
    for lab in labels:
        vals = np.array([r["rel_time_per_gibbs"] for r in norm_rows if r["label"] == lab], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        q25, q50, q75 = np.percentile(vals, [25, 50, 75])
        summary.append({
            "label": lab,
            "n": int(vals.size),
            "median": float(q50),
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1) if vals.size > 1 else 0.0),
            "q25": float(q25),
            "q75": float(q75),
            "iqr": float(q75 - q25),
        })
    return summary


# ============================================================
# Plot: boxplot + jitter
# ============================================================

def boxplot_rel_cost(norm_rows, case_id: str, out_dir: str, title: str):
    paper_style()
    os.makedirs(out_dir, exist_ok=True)

    rows = [r for r in norm_rows if r["case_id"] == case_id]
    if not rows:
        print("[Skip] No data for {}".format(case_id))
        return

    labels = sorted({r["label"] for r in rows})

    def label_key(lab: str):
        if lab.startswith("BGDA"):
            return (0, 0)
        if lab.startswith("KL"):
            return (1, 0)
        if lab.startswith("MD-Sinkhorn"):
            try:
                m = int(lab.split("M=")[1].split(")")[0])
            except Exception:
                m = 999
            return (2, m)
        return (9, 0)

    labels = sorted(labels, key=label_key)

    data = []
    for lab in labels:
        v = np.array([r["rel_time_per_gibbs"] for r in rows if r["label"] == lab], dtype=float)
        v = v[np.isfinite(v)]
        data.append(v)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    ax.boxplot(
        data,
        labels=labels,
        showfliers=False,
        whis=(10, 90),
    )

    rng = np.random.default_rng(0)
    for i, v in enumerate(data, start=1):
        if v.size == 0:
            continue
        x = i + 0.08 * rng.standard_normal(size=v.size)
        ax.scatter(x, v, s=10, alpha=0.25)

    ax.axhline(1.0, linewidth=1.2, linestyle="--")
    ax.set_ylabel(r"Relative time per Gibbs call (normalized within each $(N,d,\varepsilon,\mathrm{seed})$)")
    ax.set_title(title)

    if len(labels) >= 5:
        for tick in ax.get_xticklabels():
            tick.set_rotation(20)
            tick.set_ha("right")

    pdf_path = os.path.join(out_dir, "boxplot_{}.pdf".format(case_id))
    png_path = os.path.join(out_dir, "boxplot_{}.png".format(case_id))
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("[Saved] {}".format(pdf_path))
    print("[Saved] {}".format(png_path))


# ============================================================
# Main
# ============================================================

def main():
    p = argparse.ArgumentParser("Sweep (N,d,eps,seed) and make boxplots for relative time per Gibbs call.")

    p.add_argument("--case", type=str, default="both", choices=["both", "case1", "case2"])

    # d range
    p.add_argument("--d_list", type=str, default="3-10")

    # eps grid
    p.add_argument("--eps_min", type=float, default=0.1)
    p.add_argument("--eps_max", type=float, default=1.0)
    p.add_argument("--eps_num", type=int, default=5)
    p.add_argument("--eps_space", type=str, default="log", choices=["log", "lin"])

    # seeds
    p.add_argument("--seed_list", type=str, default="0,1,2")

    # Case2 N range
    p.add_argument("--N_list_case2", type=str, default="3-8")

    # MD inner steps
    p.add_argument("--M_list", type=str, default="1-5")
    p.add_argument("--M_default", type=int, default=5)

    # Instance params
    p.add_argument("--H_kind", type=str, default="random", help="random/commuting/conjugated_from_diagonal")
    p.add_argument("--H_scale", type=float, default=1.0)
    p.add_argument("--marginal_kind", type=str, default="medium")
    p.add_argument("--hard_delta", type=float, default=1e-4)
    p.add_argument("--normalize_cost", action="store_true")

    # Budgets (this experiment is about per-call cost)
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

    d_list = parse_int_range(args.d_list)
    eps_list = build_eps_list(args.eps_min, args.eps_max, args.eps_num, args.eps_space)
    seed_list = parse_int_range(args.seed_list)
    M_list = parse_int_range(args.M_list)
    N_list_case2 = parse_int_range(args.N_list_case2)

    os.makedirs(args.out_dir, exist_ok=True)

    per_run = []

    # ---------------- Case 1: N=2
    if args.case in {"both", "case1"}:
        case_id = "case1_N2"
        N = 2
        for d in d_list:
            for eps in eps_list:
                for seed in seed_list:
                    print("[{}] d={}, eps={}, seed={}".format(case_id, d, eps, seed))
                    H, gammas, dims, hk = make_instance(
                        d=d, N=N, seed=seed,
                        H_kind=args.H_kind, H_scale=args.H_scale,
                        marginal_kind=args.marginal_kind, hard_delta=args.hard_delta,
                        normalize_cost=args.normalize_cost,
                    )

                    # BGDA
                    meas = run_and_measure("BGDA", H, gammas, float(eps), dims, args)
                    per_run.append({
                        "case_id": case_id,
                        "label": meas["label"],
                        "algo": meas["algo"],
                        "M_inner": "",
                        "d": d, "N": N, "eps": float(eps), "seed": seed,
                        "H_kind": hk,
                        "normalize_cost": int(args.normalize_cost),
                        "gibbs_total": meas["gibbs_total"],
                        "runtime_sec": meas["runtime_sec"],
                        "time_per_gibbs_sec": meas["time_per_gibbs_sec"],
                    })

                    # KL
                    meas = run_and_measure("KL", H, gammas, float(eps), dims, args)
                    per_run.append({
                        "case_id": case_id,
                        "label": meas["label"],
                        "algo": meas["algo"],
                        "M_inner": "",
                        "d": d, "N": N, "eps": float(eps), "seed": seed,
                        "H_kind": hk,
                        "normalize_cost": int(args.normalize_cost),
                        "gibbs_total": meas["gibbs_total"],
                        "runtime_sec": meas["runtime_sec"],
                        "time_per_gibbs_sec": meas["time_per_gibbs_sec"],
                    })

                    # MD for each M
                    for M in M_list:
                        meas = run_and_measure("MD", H, gammas, float(eps), dims, args, M_inner=int(M))
                        per_run.append({
                            "case_id": case_id,
                            "label": meas["label"],
                            "algo": meas["algo"],
                            "M_inner": int(M),
                            "d": d, "N": N, "eps": float(eps), "seed": seed,
                            "H_kind": hk,
                            "normalize_cost": int(args.normalize_cost),
                            "gibbs_total": meas["gibbs_total"],
                            "runtime_sec": meas["runtime_sec"],
                            "time_per_gibbs_sec": meas["time_per_gibbs_sec"],
                        })

    # ---------------- Case 2: N=3..8 (only our algorithms)
    if args.case in {"both", "case2"}:
        case_id = "case2_N3to8"
        for N in N_list_case2:
            for d in d_list:
                for eps in eps_list:
                    for seed in seed_list:
                        print("[{}] N={}, d={}, eps={}, seed={}".format(case_id, N, d, eps, seed))
                        H, gammas, dims, hk = make_instance(
                            d=d, N=N, seed=seed,
                            H_kind=args.H_kind, H_scale=args.H_scale,
                            marginal_kind=args.marginal_kind, hard_delta=args.hard_delta,
                            normalize_cost=args.normalize_cost,
                        )

                        # KL
                        meas = run_and_measure("KL", H, gammas, float(eps), dims, args)
                        per_run.append({
                            "case_id": case_id,
                            "label": meas["label"],
                            "algo": meas["algo"],
                            "M_inner": "",
                            "d": d, "N": N, "eps": float(eps), "seed": seed,
                            "H_kind": hk,
                            "normalize_cost": int(args.normalize_cost),
                            "gibbs_total": meas["gibbs_total"],
                            "runtime_sec": meas["runtime_sec"],
                            "time_per_gibbs_sec": meas["time_per_gibbs_sec"],
                        })

                        # MD for each M
                        for M in M_list:
                            meas = run_and_measure("MD", H, gammas, float(eps), dims, args, M_inner=int(M))
                            per_run.append({
                                "case_id": case_id,
                                "label": meas["label"],
                                "algo": meas["algo"],
                                "M_inner": int(M),
                                "d": d, "N": N, "eps": float(eps), "seed": seed,
                                "H_kind": hk,
                                "normalize_cost": int(args.normalize_cost),
                                "gibbs_total": meas["gibbs_total"],
                                "runtime_sec": meas["runtime_sec"],
                                "time_per_gibbs_sec": meas["time_per_gibbs_sec"],
                            })

    if not per_run:
        print("[Done] No runs executed.")
        return

    # Save per_run.csv
    per_run_csv = os.path.join(args.out_dir, "per_run.csv")
    write_csv(per_run_csv, per_run, fieldnames=list(per_run[0].keys()))
    print("[Saved] {}".format(per_run_csv))

    # Normalize within config groups
    norm_rows = normalize_within_config(per_run)

    norm_csv = os.path.join(args.out_dir, "per_config_normalized.csv")
    write_csv(norm_csv, norm_rows, fieldnames=list(norm_rows[0].keys()))
    print("[Saved] {}".format(norm_csv))

    # Summary by label
    summary = summarize_by_label(norm_rows)
    if summary:
        summary_csv = os.path.join(args.out_dir, "summary.csv")
        write_csv(summary_csv, summary, fieldnames=list(summary[0].keys()))
        print("[Saved] {}".format(summary_csv))

    # Boxplots
    if args.case in {"both", "case1"}:
        title = ("Relative time per Gibbs call (N=2; d={}; eps in [{},{}]; {} norm={})"
                 .format(args.d_list, args.eps_min, args.eps_max, args.H_kind, int(args.normalize_cost)))
        boxplot_rel_cost(norm_rows, case_id="case1_N2", out_dir=args.out_dir, title=title)

    if args.case in {"both", "case2"}:
        title = ("Relative time per Gibbs call (N={}; d={}; eps in [{},{}]; {} norm={})"
                 .format(args.N_list_case2, args.d_list, args.eps_min, args.eps_max, args.H_kind, int(args.normalize_cost)))
        boxplot_rel_cost(norm_rows, case_id="case2_N3to8", out_dir=args.out_dir, title=title)


if __name__ == "__main__":
    main()

