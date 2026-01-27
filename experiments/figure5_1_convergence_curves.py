"""
experiments/figure5_1_convergence_curves.py

Chapter 5 Figure 5.1 (main):
Convergence curves (publication style) for:
  (i) Paper Algorithm 2.2 (2-marginal baseline)
  (ii) Potential marginal KL-descent (our KL descent)
  (iii) Mirror-Descent-Type Sinkhorn (MD-Sinkhorn) with inner steps M in a user-specified list (<=3)

MAIN PLOT (default):
  x-axis: log10(# Gibbs calls)
  y-axis: log10(F_marg) where F_marg = sum_i KL(Tr_{≠i} pi || gamma_i)

OPTIONAL (appendix / robustness):
  --metric trace uses y-axis: log10(max_i ||Tr_{≠i} pi - gamma_i||_1)

Outputs (png + pdf):
  experiments/figures/fig5_1_convergence_{metric}_d{d}_eps{eps}_H{Hkind}_norm{0/1}.png
  experiments/figures/fig5_1_convergence_{metric}_d{d}_eps{eps}_H{Hkind}_norm{0/1}.pdf

Extra summary table (CSV + LaTeX):
  experiments/figures/fig5_1_convergence_{metric}_d{d}_eps{eps}_H{Hkind}_norm{0/1}_summary.csv
  experiments/figures/fig5_1_convergence_{metric}_d{d}_eps{eps}_H{Hkind}_norm{0/1}_summary.tex
"""

import argparse
import os
import time
import csv
from typing import Dict, List, Union, Any

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
    dbga_algorithm_2_2,              # Algorithm 2.2 (paper)
    potential_marginal_kl_descent,   # our KL-descent
    md_type_sinkhorn_potential,      # our MD-type Sinkhorn
)

from src.linalg import hermitianize


# ============================================================
# CLI parsing utilities
# ============================================================

def parse_inner_steps(raw: Union[str, List[int], List[str], None]) -> List[int]:
    """
    Accept:
      --inner_steps 1 2 5     (nargs="*")
      --inner_steps 1,2,5     (string)
      None -> default [1,5,10]
    Enforce: 1 <= len(list) <= 3, all positive ints, unique, preserve order.
    """
    if raw is None:
        steps = [1, 5, 10]
    elif isinstance(raw, list):
        steps = [int(x) for x in raw]
    else:
        s = str(raw).strip()
        if s == "":
            steps = [1, 5, 10]
        else:
            parts = [p.strip() for p in s.split(",") if p.strip() != ""]
            steps = [int(p) for p in parts]

    out: List[int] = []
    seen = set()
    for m in steps:
        if m <= 0:
            raise ValueError(f"inner_steps must be positive integers, got {m}.")
        if m not in seen:
            out.append(m)
            seen.add(m)

    if len(out) == 0:
        out = [1, 5, 10]
    if len(out) > 3:
        raise ValueError(f"At most 3 inner steps are allowed (got {out}).")
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


def make_two_marginal_instance(
    d: int,
    seed: int,
    *,
    H_kind: str = "random",
    H_scale: float = 1.0,
    marginal_kind: str = "medium",
    hard_delta: float = 1e-4,
    normalize_cost: bool = False,
):
    rng = np.random.default_rng(seed)
    dims = [int(d), int(d)]

    hk = H_kind.lower().strip()
    if hk == "random":
        H = gen_H_random(dims, rng, scale=H_scale)
    elif hk == "commuting":
        H = gen_H_commuting(dims, rng, scale=H_scale)
    elif hk in {"conjugated", "conjugated_from_diagonal", "conjugated_diagonal"}:
        H = gen_H_conjugated_from_diagonal(dims, rng, scale=H_scale)
        hk = "conjugated"
    else:
        raise ValueError(f"Unknown H_kind='{H_kind}' (use random/commuting/conjugated_from_diagonal)")

    rho = gen_marginal(d, rng, kind=marginal_kind, hard_delta=hard_delta)
    sigma = gen_marginal(d, rng, kind=marginal_kind, hard_delta=hard_delta)
    gammas = [rho, sigma]

    if normalize_cost:
        H = normalize_cost_operator(H)

    return H, gammas, dims, hk


# ============================================================
# Result extraction
# ============================================================

def safe_get_x(res) -> np.ndarray:
    """
    x-axis: gibbs_calls_list if available; else fallback to total gibbs_calls.
    """
    if hasattr(res, "gibbs_calls_list") and res.gibbs_calls_list is not None and len(res.gibbs_calls_list) > 0:
        x = np.asarray(res.gibbs_calls_list, dtype=float)
    else:
        total = float(getattr(res, "gibbs_calls", 0) or 0)
        n = len(getattr(res, "F_list", []) or getattr(res, "e_tr_list", []))
        if n <= 1:
            x = np.asarray([max(1.0, total)], dtype=float)
        else:
            x = np.linspace(0.0, total, num=n, dtype=float)

    return np.maximum(x, 1.0)  # avoid log10(0)


def safe_get_y(res, metric: str) -> np.ndarray:
    """
    metric:
      - 'Fmarg' : use res.F_list (stores F_marg(pi_k, gammas))
      - 'trace' : use res.e_tr_list (max marginal trace norm mismatch)
    """
    m = metric.lower().strip()

    if m in {"fmarg", "f_marg", "f"}:
        if not hasattr(res, "F_list") or res.F_list is None or len(res.F_list) == 0:
            raise AttributeError("Result object has no non-empty F_list.")
        y = np.asarray(res.F_list, dtype=float)
        return np.maximum(y, 1e-300)

    if m in {"trace", "tr", "e_tr"}:
        if not hasattr(res, "e_tr_list") or res.e_tr_list is None or len(res.e_tr_list) == 0:
            raise AttributeError("Result object has no non-empty e_tr_list.")
        y = np.asarray(res.e_tr_list, dtype=float)
        return np.maximum(y, 1e-300)

    raise ValueError("metric must be 'Fmarg' or 'trace'.")


def get_total_gibbs_calls(res) -> int:
    """
    Prefer res.gibbs_calls; fallback to last entry of gibbs_calls_list.
    """
    if hasattr(res, "gibbs_calls") and res.gibbs_calls is not None:
        return int(res.gibbs_calls)
    if hasattr(res, "gibbs_calls_list") and res.gibbs_calls_list is not None and len(res.gibbs_calls_list) > 0:
        return int(res.gibbs_calls_list[-1])
    return 0


def get_final_value(res, attr: str) -> float:
    """
    attr in {"F_list","e_tr_list"}; returns last value or NaN.
    """
    if hasattr(res, attr):
        lst = getattr(res, attr)
        if lst is not None and len(lst) > 0:
            return float(lst[-1])
    return float("nan")


# ============================================================
# Run solvers + timing
# ============================================================

def run_algorithms(args, H, gammas, dims) -> (Dict[str, object], List[Dict[str, Any]]):
    results: Dict[str, object] = {}
    summary_rows: List[Dict[str, Any]] = []

    def record_row(method: str, res_obj, wall_time: float):
        gibbs = get_total_gibbs_calls(res_obj)
        summary_rows.append({
            "method": method,
            "wall_time_sec": wall_time,
            "gibbs_calls": gibbs,
            "sec_per_1k_gibbs": (wall_time / max(gibbs, 1)) * 1000.0,
            "final_Fmarg": get_final_value(res_obj, "F_list"),
            "final_trace_err": get_final_value(res_obj, "e_tr_list"),
        })

    # (i) Algorithm 2.2
    print("[Run] Algorithm 2.2 ...")
    t0 = time.perf_counter()
    res_alg22 = dbga_algorithm_2_2(
        H, gammas, args.eps, dims,
        T=args.T_alg22,
        tol_tr=args.tol_tr,
        delta=args.delta_paper,
        gauge_trace0=args.gauge_trace0,
        store_hist=False,
        project_pi=True,
        jitter=args.jitter,
        U0=None,
        V0=None,
    )
    t1 = time.perf_counter()
    results["Alg. 2.2"] = res_alg22
    record_row("Alg. 2.2", res_alg22, t1 - t0)

    # (ii) KL descent
    print("[Run] KL descent ...")
    t0 = time.perf_counter()
    res_kl = potential_marginal_kl_descent(
        H, gammas, args.eps, dims,
        T=args.T_kl,
        eta=args.eta_kl,               # None -> internal default 1/N
        jitter_log=args.jitter,
        tol_tr=args.tol_tr,
        store_hist=False,
        project_pi=True,
    )
    t1 = time.perf_counter()
    results["KL descent"] = res_kl
    record_row("KL descent", res_kl, t1 - t0)

    # (iii) MD-Sinkhorn family
    for M in args.inner_steps:
        label = f"MD-Sinkhorn (inner steps={M})"
        print(f"[Run] {label} ...")
        t0 = time.perf_counter()
        res_md = md_type_sinkhorn_potential(
            H, gammas, args.eps, dims,
            T_outer=args.T_md,
            tol_tr=args.tol_tr,
            jitter=args.jitter,
            eta_inner=args.eta_inner,
            M_inner=M,
            name="MD-Sinkhorn",
            keep_U_hist=False,
            keep_pi_hist=False,
            tol_inner=None,
            project_pi=True,
        )
        t1 = time.perf_counter()
        results[label] = res_md
        record_row(label, res_md, t1 - t0)

    return results, summary_rows


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


def plot_curves(args, results: Dict[str, object], out_prefix: str):
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    paper_style()

    metric_key = args.metric.lower().strip()

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    c_kl = colors[1]  # orange-ish
    c_md = colors[2]  # green-ish

    style_map = {
        "Alg. 2.2": dict(color="0.35", linestyle=(0, (5, 3)), zorder=1),
        "KL descent": dict(color=c_kl, linestyle="-", zorder=4),
    }

    md_linestyles = [
        (0, (5, 2)),  # dashed
        "-",          # solid
        ":",          # dotted
    ]
    for j, M in enumerate(args.inner_steps):
        ls = md_linestyles[j]
        style_map[f"MD-Sinkhorn (inner steps={M})"] = dict(color=c_md, linestyle=ls, zorder=3)

    ordered_labels = ["Alg. 2.2", "KL descent"] + [
        f"MD-Sinkhorn (inner steps={M})" for M in args.inner_steps
    ]

    fig, ax = plt.subplots(figsize=(6.4, 4.8))

    for lab in ordered_labels:
        if lab not in results:
            continue
        res = results[lab]
        x = safe_get_x(res)
        y = safe_get_y(res, metric=args.metric)
        ax.plot(np.log10(x), np.log10(y), label=lab, **style_map.get(lab, {}))

    ax.set_xlabel(r"$\log_{10}(\#\ \mathrm{Gibbs\ calls})$")

    if metric_key in {"fmarg", "f_marg", "f"}:
        ax.set_ylabel(r"$\log_{10}\,F_{\mathrm{marg}}$")
        ax.set_title(rf"Convergence for entropic QOT ($d={args.d},\ \varepsilon={args.eps:g}$)")
    else:
        ax.set_ylabel(r"$\log_{10}(\mathrm{marginal\ trace\ error})$")
        ax.set_title(rf"Convergence (trace-norm feasibility) ($d={args.d},\ \varepsilon={args.eps:g}$)")

    ax.legend(
        loc="lower right",
        frameon=True,
        framealpha=0.95,
        borderpad=0.6,
        handlelength=2.6,
        handletextpad=0.8,
        labelspacing=0.4,
    )

    ax.margins(x=0.02, y=0.05)

    png_path = out_prefix + ".png"
    pdf_path = out_prefix + ".pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[Saved] {png_path}")
    print(f"[Saved] {pdf_path}")


# ============================================================
# Summary table output
# ============================================================

def save_summary_csv(rows: List[Dict[str, Any]], out_csv_path: str):
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    fields = ["method", "wall_time_sec", "gibbs_calls", "sec_per_1k_gibbs", "final_Fmarg", "final_trace_err"]
    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def save_summary_tex(rows: List[Dict[str, Any]], out_tex_path: str):
    os.makedirs(os.path.dirname(out_tex_path), exist_ok=True)
    lines: List[str] = []
    lines.append(r"\begin{tabular}{l r r r r r}")
    lines.append(r"\hline")
    lines.append(r"Method & time (s) & Gibbs calls & s/1k Gibbs & final $F_{\mathrm{marg}}$ & final trace err \\")
    lines.append(r"\hline")
    for r in rows:
        lines.append(
            f"{r['method']} & "
            f"{float(r['wall_time_sec']):.3f} & "
            f"{int(r['gibbs_calls'])} & "
            f"{float(r['sec_per_1k_gibbs']):.3f} & "
            f"{float(r['final_Fmarg']):.3e} & "
            f"{float(r['final_trace_err']):.3e} \\\\"
        )
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")

    with open(out_tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser("Generate Figure 5.1 convergence curves (Alg2.2 vs KL vs MD-Sinkhorn)")
    parser.add_argument("--d", type=int, default=3)
    parser.add_argument("--eps", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--H_kind", type=str, default="random",
                        help="random / commuting / conjugated_from_diagonal")
    parser.add_argument("--H_scale", type=float, default=1.0)
    parser.add_argument("--marginal_kind", type=str, default="medium")
    parser.add_argument("--hard_delta", type=float, default=1e-4)
    parser.add_argument("--normalize_cost", action="store_true")

    parser.add_argument("--tol_tr", type=float, default=1e-6)

    parser.add_argument("--T_alg22", type=int, default=5000)
    parser.add_argument("--T_kl", type=int, default=2000)
    parser.add_argument("--T_md", type=int, default=2000)

    parser.add_argument("--delta_paper", type=float, default=1e-6)
    parser.add_argument("--gauge_trace0", action="store_true")

    parser.add_argument("--eta_kl", type=float, default=None)
    parser.add_argument("--eta_inner", type=float, default=1.0)
    parser.add_argument("--jitter", type=float, default=1e-12)

    parser.add_argument("--metric", type=str, default="Fmarg", choices=["Fmarg", "trace"])

    parser.add_argument(
        "--inner_steps",
        nargs="*",
        default=None,
        help="MD-Sinkhorn inner steps (<=3). Examples: --inner_steps 1 2 5  OR  --inner_steps 1,2,5. Default: 1 5 10",
    )

    parser.add_argument("--out_dir", type=str, default="experiments/figures")
    args = parser.parse_args()

    # parse inner steps with constraints (<=3), default [1,5,10]
    if args.inner_steps is None or len(args.inner_steps) == 0:
        args.inner_steps = [1, 5, 10]
    elif len(args.inner_steps) == 1 and isinstance(args.inner_steps[0], str) and ("," in args.inner_steps[0]):
        args.inner_steps = parse_inner_steps(args.inner_steps[0])
    else:
        args.inner_steps = parse_inner_steps(args.inner_steps)

    H, gammas, dims, hk = make_two_marginal_instance(
        d=args.d,
        seed=args.seed,
        H_kind=args.H_kind,
        H_scale=args.H_scale,
        marginal_kind=args.marginal_kind,
        hard_delta=args.hard_delta,
        normalize_cost=args.normalize_cost,
    )

    results, summary_rows = run_algorithms(args, H, gammas, dims)

    out_prefix = os.path.join(
        args.out_dir,
        f"fig5_1_convergence_{args.metric}_d{args.d}_eps{args.eps:g}_H{hk}_norm{int(args.normalize_cost)}"
    )

    plot_curves(args, results, out_prefix)

    # save summary table
    csv_path = out_prefix + "_summary.csv"
    tex_path = out_prefix + "_summary.tex"
    save_summary_csv(summary_rows, csv_path)
    save_summary_tex(summary_rows, tex_path)
    print(f"[Saved] {csv_path}")
    print(f"[Saved] {tex_path}")


if __name__ == "__main__":
    main()

