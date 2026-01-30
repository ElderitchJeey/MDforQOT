"""
experiments/fig_inner_update_convergence.py

Convergence test for standalone inner iteration md_inner_update_i
(ONLY inner loop, no outer MD-Sinkhorn).

This version records BOTH:
  - e_i_tr = || Tr_{!=i}(pi) - gamma_i ||_1
  - F_i    = KL( Tr_{!=i}(pi) || gamma_i )

and produces 4 plots:
  (1) e_i_tr vs inner step  (log y)
  (2) F_i    vs inner step  (log y)
  (3) e_i_tr vs Gibbs calls (log y)
  (4) F_i    vs Gibbs calls (log y)

Compatibility: Python 3.8+
"""

from __future__ import print_function

import os
import sys
import csv
import argparse
from typing import List, Dict

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------
# Project imports
# ----------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.instances import gen_H_random, gen_H_ising_qubits, gen_marginal
from src.SolverofEQOT import md_inner_update_i


def parse_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def save_csv(path: str, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _safe_log10(x: float) -> float:
    # avoid -inf in csv/plots
    if not np.isfinite(x) or x <= 0:
        return float("nan")
    return float(np.log10(x))


def plot_curves(
    rows: List[Dict[str, object]],
    eps_list: List[float],
    eta_list: List[float],
    x_key: str,
    y_key: str,
    title: str,
    xlabel: str,
    ylabel: str,
    out_png: str,
    out_pdf: str,
) -> None:
    plt.figure(figsize=(6.6, 4.4))

    # group by (eps, eta)
    for eps in eps_list:
        for eta in eta_list:
            sub = [
                r for r in rows
                if abs(float(r["eps"]) - eps) < 1e-15
                and abs(float(r["eta_inner"]) - eta) < 1e-15
            ]
            sub.sort(key=lambda r: int(r["inner_step"]))
            if not sub:
                continue

            xs = [float(r[x_key]) for r in sub]
            ys = [float(r[y_key]) for r in sub]

            lbl = r"$\varepsilon={}$, $\eta={}$".format(eps, eta)
            plt.plot(xs, ys, marker="o", linewidth=1.6, markersize=3.6, label=lbl)

    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(frameon=True, fontsize=8)
    plt.tight_layout()

    ensure_dir(os.path.dirname(out_png))
    plt.savefig(out_png, dpi=250)
    plt.savefig(out_pdf)
    plt.close()


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--N", type=int, default=4)
    ap.add_argument("--d", type=int, default=3)
    ap.add_argument("--i", type=int, default=0, help="block index to update")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--eps_list", type=str, default="1,0.5,0.3")
    ap.add_argument("--eta_list", type=str, default="1.0,0.7,0.4")

    ap.add_argument("--M_inner", type=int, default=80)
    ap.add_argument("--tol_inner", type=float, default=0.0, help="<=0 disables trace-norm early stop")
    ap.add_argument("--tol_F_inner", type=float, default=0.0, help="<=0 disables KL early stop")
    ap.add_argument("--project_pi", action="store_true")

    ap.add_argument("--init_U", type=str, default="zero", choices=["zero", "random"])
    ap.add_argument("--U_scale", type=float, default=0.2)

    ap.add_argument("--H_kind", type=str, default="random", choices=["random", "ising_qubits"])
    ap.add_argument("--H_scale", type=float, default=1.0)
    ap.add_argument("--gamma_kind", type=str, default="medium", choices=["easy", "medium", "hard"])
    ap.add_argument("--hard_delta", type=float, default=1e-4)

    ap.add_argument("--outdir", type=str, default="experiments/figures/inner_update")
    ap.add_argument("--tag", type=str, default="")

    args = ap.parse_args()

    N, d, i = int(args.N), int(args.d), int(args.i)
    dims = [d] * N
    eps_list = parse_floats(args.eps_list)
    eta_list = parse_floats(args.eta_list)

    rng = np.random.default_rng(int(args.seed))

    # ---------- instance ----------
    if args.H_kind == "random":
        H = gen_H_random(dims, rng, scale=float(args.H_scale))
    else:
        if d != 2:
            raise ValueError("H_kind=ising_qubits requires d=2 (qubits).")
        H = gen_H_ising_qubits(N, rng, scale=float(args.H_scale), noncommuting=True)

    gammas = [
        gen_marginal(d, rng, kind=args.gamma_kind, hard_delta=float(args.hard_delta))
        for _ in range(N)
    ]
    gamma_i = gammas[i]

    # ---------- init potentials ----------
    if args.init_U == "zero":
        U_list0 = [np.zeros((d, d), dtype=complex) for _ in range(N)]
    else:
        U_list0 = []
        for _ in range(N):
            A = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
            U = 0.5 * (A + A.conj().T)
            U_list0.append(float(args.U_scale) * U)

    # ---------- run sweeps ----------
    traj_rows: List[Dict[str, object]] = []

    for eps in eps_list:
        for eta in eta_list:
            res = md_inner_update_i(
                i=i,
                U_list=U_list0,
                H=H,
                gamma_i=gamma_i,
                eps=float(eps),
                dims=dims,
                pi0=None,
                eta_inner=float(eta),
                M_inner=int(args.M_inner),
                tol_inner=float(args.tol_inner),
                tol_F_inner=float(args.tol_F_inner),
                project_pi=bool(args.project_pi),
                reset_counter=True,
                keep_history=True,
            )

            # Require the modified solver that records F_i_list
            if not hasattr(res, "F_i_list"):
                raise RuntimeError(
                    "md_inner_update_i result has no F_i_list. "
                    "Please modify SolverofEQOT.md_inner_update_i to record and return F_i_list."
                )

            e_list = list(res.e_i_tr_list)
            F_list = list(res.F_i_list)

            # Align lengths robustly
            L = min(len(e_list), len(F_list), len(res.times), len(res.gibbs_calls_list))
            for k in range(L):
                e = float(e_list[k])
                F = float(F_list[k])
                t = float(res.times[k])
                gc = int(res.gibbs_calls_list[k])

                traj_rows.append({
                    "seed": int(args.seed),
                    "N": N,
                    "d": d,
                    "i": i,
                    "eps": float(eps),
                    "eta_inner": float(eta),
                    "M_inner": int(args.M_inner),
                    "tol_inner": float(args.tol_inner),
                    "tol_F_inner": float(args.tol_F_inner),
                    "inner_step": int(k),
                    "time_sec": t,
                    "gibbs_calls": gc,
                    "e_i_tr": e,
                    "F_i": F,
                    "log10_e_i_tr": _safe_log10(e),
                    "log10_F_i": _safe_log10(F),
                    "converged": int(bool(res.converged)),
                })

    # ---------- save ----------
    ensure_dir(args.outdir)
    tag = ("_" + args.tag) if args.tag.strip() else ""
    stem = f"inner_update_conv_N{N}_d{d}_i{i}_seed{int(args.seed)}{tag}"

    csv_path = os.path.join(args.outdir, stem + ".csv")
    save_csv(
        csv_path,
        traj_rows,
        fieldnames=[
            "seed","N","d","i","eps","eta_inner","M_inner","tol_inner","tol_F_inner",
            "inner_step","time_sec","gibbs_calls",
            "e_i_tr","F_i","log10_e_i_tr","log10_F_i",
            "converged"
        ],
    )

    # ---------- plots ----------
    # (1) e vs step
    plot_curves(
        rows=traj_rows,
        eps_list=eps_list,
        eta_list=eta_list,
        x_key="inner_step",
        y_key="e_i_tr",
        title=r"Inner update convergence: $e_i^{\mathrm{tr}}$ vs step",
        xlabel=r"Inner step $k$",
        ylabel=r"$e_i^{\mathrm{tr}}(k)=\|\mathrm{Tr}_{\neq i}\pi_k-\gamma_i\|_1$",
        out_png=os.path.join(args.outdir, stem + "_e_vs_step.png"),
        out_pdf=os.path.join(args.outdir, stem + "_e_vs_step.pdf"),
    )

    # (2) F vs step
    plot_curves(
        rows=traj_rows,
        eps_list=eps_list,
        eta_list=eta_list,
        x_key="inner_step",
        y_key="F_i",
        title=r"Inner update convergence: $F_i=\mathrm{KL}(\rho_i\|\gamma_i)$ vs step",
        xlabel=r"Inner step $k$",
        ylabel=r"$F_i(k)$",
        out_png=os.path.join(args.outdir, stem + "_F_vs_step.png"),
        out_pdf=os.path.join(args.outdir, stem + "_F_vs_step.pdf"),
    )

    # (3) e vs Gibbs calls
    plot_curves(
        rows=traj_rows,
        eps_list=eps_list,
        eta_list=eta_list,
        x_key="gibbs_calls",
        y_key="e_i_tr",
        title=r"Inner update convergence: $e_i^{\mathrm{tr}}$ vs Gibbs calls",
        xlabel=r"Gibbs calls",
        ylabel=r"$e_i^{\mathrm{tr}}$",
        out_png=os.path.join(args.outdir, stem + "_e_vs_gibbs.png"),
        out_pdf=os.path.join(args.outdir, stem + "_e_vs_gibbs.pdf"),
    )

    # (4) F vs Gibbs calls
    plot_curves(
        rows=traj_rows,
        eps_list=eps_list,
        eta_list=eta_list,
        x_key="gibbs_calls",
        y_key="F_i",
        title=r"Inner update convergence: $F_i$ vs Gibbs calls",
        xlabel=r"Gibbs calls",
        ylabel=r"$F_i$",
        out_png=os.path.join(args.outdir, stem + "_F_vs_gibbs.png"),
        out_pdf=os.path.join(args.outdir, stem + "_F_vs_gibbs.pdf"),
    )

    print("Saved:")
    print("  CSV:", csv_path)
    print("  FIGS:", os.path.join(args.outdir, stem + "_*.png"))


if __name__ == "__main__":
    main()
