"""
experiments/figure5_2_multimarginal_convergence.py

Chapter 5 (next experiment):
Multi-marginal convergence comparison for N in {3,4,5} (default),
fixed d (default 3) and eps (default 0.5).

Metric (main):
  y-axis: log10(F_marg)  where F_marg is stored as res.F_list

x-axis:
  log10(# Gibbs calls)   from res.gibbs_calls_list

Curves (for each N):
  - KL descent
  - MD-Sinkhorn (M_inner = 1)
  - MD-Sinkhorn (M_inner = 5)

All curves are plotted in a single figure.

Outputs (png + pdf):
  experiments/figures/fig5_2_multimarginal_Fmarg_d{d}_eps{eps}_Ns{Ns}_H{Hkind}_norm{0/1}.png
  experiments/figures/fig5_2_multimarginal_Fmarg_d{d}_eps{eps}_Ns{Ns}_H{Hkind}_norm{0/1}.pdf
"""

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from src.instances import (
    gen_H_random,
    gen_H_commuting,
    gen_H_conjugated_from_diagonal,
    gen_marginal,
)

from src.SolverofEQOT import (
    potential_marginal_kl_descent,   # our KL-descent (supports N marginals)
    md_type_sinkhorn_potential,      # our MD-type Sinkhorn (supports N marginals)
)

from src.linalg import hermitianize


# ============================================================
# CLI parsing
# ============================================================

def parse_int_csv(s: str) -> List[int]:
    parts = [p.strip() for p in str(s).split(",") if p.strip() != ""]
    out = [int(p) for p in parts]
    if len(out) == 0:
        raise ValueError("Empty Ns list.")
    return out


# ============================================================
# Instance utilities
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


def make_instance(
    N: int,
    d: int,
    seed: int,
    *,
    H_kind: str = "random",
    H_scale: float = 1.0,
    marginal_kind: str = "medium",
    hard_delta: float = 1e-4,
    normalize_cost: bool = False,
) -> Tuple[np.ndarray, List[np.ndarray], List[int], str]:
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
        raise ValueError(f"Unknown H_kind='{H_kind}' (use random/commuting/conjugated_from_diagonal).")

    gammas = [gen_marginal(d, rng, kind=marginal_kind, hard_delta=hard_delta) for _ in range(N)]

    if normalize_cost:
        H = normalize_cost_operator(H)

    return H, gammas, dims, hk


# ============================================================
# Result extraction
# ============================================================

def safe_get_x(res) -> np.ndarray:
    """Use gibbs_calls_list if present; fallback is a linear interpolation to total gibbs_calls."""
    if hasattr(res, "gibbs_calls_list") and res.gibbs_calls_list is not None and len(res.gibbs_calls_list) > 0:
        x = np.asarray(res.gibbs_calls_list, dtype=float)
    else:
        total = float(getattr(res, "gibbs_calls", 0) or 0)
        n = len(getattr(res, "F_list", []) or [])
        if n <= 1:
            x = np.asarray([max(1.0, total)], dtype=float)
        else:
            x = np.linspace(0.0, total, num=n, dtype=float)
    return np.maximum(x, 1.0)


def safe_get_Fmarg(res) -> np.ndarray:
    if not hasattr(res, "F_list") or res.F_list is None or len(res.F_list) == 0:
        raise AttributeError("Result object has no non-empty F_list (needed for F_marg plot).")
    y = np.asarray(res.F_list, dtype=float)
    return np.maximum(y, 1e-300)


# ============================================================
# Run solvers for a given N
# ============================================================

def run_for_N(args, N: int) -> Dict[str, object]:
    H, gammas, dims, hk = make_instance(
        N=N,
        d=args.d,
        seed=args.seed,
        H_kind=args.H_kind,
        H_scale=args.H_scale,
        marginal_kind=args.marginal_kind,
        hard_delta=args.hard_delta,
        normalize_cost=args.normalize_cost,
    )

    results: Dict[str, object] = {}

    # KL
    print(f"[Run] N={N}: KL descent ...")
    res_kl = potential_marginal_kl_descent(
        H, gammas, args.eps, dims,
        T=args.T_kl,
        eta=args.eta_kl,               # None -> internal default 1/N
        jitter_log=args.jitter,
        tol_tr=args.tol_tr,
        store_hist=False,
        project_pi=True,
    )
    results[f"N={N} | KL"] = res_kl

    # MD-Sinkhorn with M_inner in {1,5}
    for M in args.M_list:
        print(f"[Run] N={N}: MD-Sinkhorn (M_inner={M}) ...")
        res_md = md_type_sinkhorn_potential(
            H, gammas, args.eps, dims,
            T_outer=args.T_md,
            tol_tr=args.tol_tr,
            jitter=args.jitter,
            eta_inner=args.eta_inner,
            M_inner=int(M),
            name="MD-Sinkhorn",
            keep_U_hist=False,
            keep_pi_hist=False,
            tol_inner=None,
            project_pi=True,
        )
        results[f"N={N} | MD (M={M})"] = res_md

    return results, hk


# ============================================================
# Publication-style plotting
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


def plot_all(args, all_results: Dict[str, object], hk: str, out_prefix: str):
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    paper_style()

    # color per N, linestyle per algorithm
    # Use tab10: N=3,4,5 distinct colors
    cmap = plt.get_cmap("tab10")
    Ns_sorted = sorted(set(int(k.split("|")[0].split("=")[1].strip()) for k in all_results.keys()))

    color_of_N = {N: cmap(i % 10) for i, N in enumerate(Ns_sorted)}

    # Linestyle mapping
    # KL: solid, MD(M=1): dashed, MD(M=5): dotted
    def style_for(label: str):
        # label format: "N={N} | KL" or "N={N} | MD (M={M})"
        N = int(label.split("|")[0].split("=")[1].strip())
        c = color_of_N[N]
        if "| KL" in label:
            return dict(color=c, linestyle="-", zorder=4)
        if "MD (M=1)" in label:
            return dict(color=c, linestyle=(0, (5, 2)), zorder=3)
        if "MD (M=5)" in label:
            return dict(color=c, linestyle=":", zorder=3)
        # fallback
        return dict(color=c, linestyle="-", zorder=2)

    # Legend order: grouped by N, then KL/MD1/MD5
    ordered_labels: List[str] = []
    for N in Ns_sorted:
        ordered_labels.append(f"N={N} | KL")
        ordered_labels.append(f"N={N} | MD (M=1)")
        ordered_labels.append(f"N={N} | MD (M=5)")

    fig, ax = plt.subplots(figsize=(7.0, 4.8))

    for lab in ordered_labels:
        if lab not in all_results:
            continue
        res = all_results[lab]
        x = safe_get_x(res)
        y = safe_get_Fmarg(res)
        ax.plot(np.log10(x), np.log10(y), label=lab, **style_for(lab))

    ax.set_xlabel(r"$\log_{10}(\#\ \mathrm{Gibbs\ calls})$")
    ax.set_ylabel(r"$\log_{10}\,F_{\mathrm{marg}}$")
    Ns_str = ",".join(str(n) for n in Ns_sorted)
    ax.set_title(rf"Multi-marginal convergence ($d={args.d},\ \varepsilon={args.eps:g},\ N\in\{{{Ns_str}\}}$)")

    # Compact legend; many curves => 2 columns is usually better
    ax.legend(
        loc="upper right",
        frameon=True,
        framealpha=0.95,
        borderpad=0.6,
        handlelength=2.6,
        handletextpad=0.8,
        labelspacing=0.35,
        ncol=2,
    )

    ax.margins(x=0.02, y=0.06)

    png_path = out_prefix + ".png"
    pdf_path = out_prefix + ".pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[Saved] {png_path}")
    print(f"[Saved] {pdf_path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser("Multi-marginal convergence plot: KL vs MD-Sinkhorn for N in {3,4,5}")
    parser.add_argument("--Ns", type=str, default="3,4,5", help="Comma-separated list of marginal counts, e.g. 3,4,5")
    parser.add_argument("--d", type=int, default=3)
    parser.add_argument("--eps", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--H_kind", type=str, default="random",
                        help="random / commuting / conjugated_from_diagonal")
    parser.add_argument("--H_scale", type=float, default=1.0)
    parser.add_argument("--marginal_kind", type=str, default="medium")
    parser.add_argument("--hard_delta", type=float, default=1e-4)
    parser.add_argument("--normalize_cost", action="store_true")

    # Main metric in this experiment is Fmarg only; tol_tr is still used for optional early-stopping inside solvers
    parser.add_argument("--tol_tr", type=float, default=1e-6)

    # Budgets (can be tuned; kept modest to stay fast for N=5)
    parser.add_argument("--T_kl", type=int, default=5000)
    parser.add_argument("--T_md", type=int, default=5000)

    # KL hyperparam (not shown in figure)
    parser.add_argument("--eta_kl", type=float, default=None)

    # MD hyperparams
    parser.add_argument("--eta_inner", type=float, default=1.0)
    parser.add_argument("--M_list", type=str, default="1,5", help="MD inner steps list (fixed to 2 values for this figure). Default: 1,5")

    parser.add_argument("--jitter", type=float, default=1e-12)

    parser.add_argument("--out_dir", type=str, default="experiments/figures")
    args = parser.parse_args()

    Ns = parse_int_csv(args.Ns)
    # enforce "small to large" ordering in figure narrative
    Ns = sorted(Ns)

    M_list = parse_int_csv(args.M_list)
    # enforce exactly {1,5} by default; allow user override but keep <=2 for clarity
    if len(M_list) > 2:
        raise ValueError(f"For this figure, please use at most 2 MD inner-step values (got {M_list}).")
    args.M_list = M_list

    all_results: Dict[str, object] = {}
    hk_used = None

    for N in Ns:
        results_N, hk = run_for_N(args, N)
        hk_used = hk if hk_used is None else hk_used
        all_results.update(results_N)

    Ns_str = ",".join(str(n) for n in Ns)
    out_prefix = os.path.join(
        args.out_dir,
        f"fig5_2_multimarginal_Fmarg_d{args.d}_eps{args.eps:g}_Ns{Ns_str}_H{hk_used}_norm{int(args.normalize_cost)}"
    )
    plot_all(args, all_results, hk_used, out_prefix)


if __name__ == "__main__":
    main()
