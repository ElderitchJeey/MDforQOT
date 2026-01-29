# experiments/sweep_eta_bgda.py
#
# BGDA eta sweep driver (delegates tuning logic to src.SolverofEQOT.tune_bgda_eta).
#
# Run (from repo root):
#   python -m experiments.sweep_eta_bgda
#   python -m experiments.sweep_eta_bgda --dims 2 2 --eps 0.5 --tol 1e-8 --T 300 --eta_grid 0.25 0.5 1 2 4 8
#   python -m experiments.sweep_eta_bgda --kind medium --seed 0 --save_csv results/bgda_eta_sweep.csv
#
# Notes:
# - This script assumes you already have src/instances.py (or src/instance.py) defining:
#     gen_H_random(dims, rng, scale)
#     gen_marginal(d, rng, kind, hard_delta)
# - Tuning logic lives in src/SolverofEQOT.py: tune_bgda_eta(...)

import argparse
import csv
import os
from typing import List

import numpy as np

# ---- adjust if your file name is src/instance.py instead of src/instances.py ----
from src.instances import gen_H_random, gen_marginal

from src.SolverofEQOT import tune_bgda_eta


def _save_csv(path: str, records) -> None:
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "eta",
                "converged",
                "stop_iter",
                "final_e_tr",
                "final_F",
                "time_sec",
                "gibbs_calls",
                "message",
            ]
        )
        for r in records:
            w.writerow(
                [
                    r.eta,
                    int(bool(r.converged)),
                    int(r.stop_iter),
                    float(r.final_e_tr),
                    float(r.final_F),
                    float(r.time_sec),
                    int(r.gibbs_calls),
                    str(r.message),
                ]
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", nargs="+", type=int, default=[2, 2], help="Local dimensions, e.g. --dims 2 2 or --dims 2 2 2")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for instance generation")
    ap.add_argument("--eps", type=float, default=0.5, help="Entropic regularization eps")
    ap.add_argument("--tol", type=float, default=1e-8, help="BGDA stopping tolerance on max marginal trace-norm error")
    ap.add_argument("--T", type=int, default=300, help="BGDA max iterations per eta")

    ap.add_argument(
        "--eta_grid",
        nargs="+",
        type=float,
        default=[0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
        help="Candidate eta list",
    )
    ap.add_argument(
        "--prefer",
        type=str,
        default="gibbs_calls",
        choices=["gibbs_calls", "time_sec"],
        help="Tuning preference (delegated to tune_bgda_eta)",
    )

    # instance controls
    ap.add_argument("--H_scale", type=float, default=1.0, help="Scale for gen_H_random")
    ap.add_argument("--kind", type=str, default="medium", choices=["easy", "medium", "hard"], help="Marginal generator kind")
    ap.add_argument("--hard_delta", type=float, default=1e-4, help="hard kind delta (requires 0<delta<1/(d-1))")

    # solver controls
    ap.add_argument("--gauge_trace0", action="store_true", help="Enable gauge fixing Tr(U_i)=0 in BGDA")
    ap.add_argument("--no_project", action="store_true", help="Disable proj_to_density inside Gibbs mapping")
    ap.add_argument("--store_hist", action="store_true", help="Store BGDA trajectories (large)")

    # output
    ap.add_argument("--save_csv", type=str, default="", help="Optional CSV path, e.g. results/bgda_eta_sweep.csv")

    args = ap.parse_args()

    dims = [int(x) for x in args.dims]
    rng = np.random.default_rng(int(args.seed))

    # Fixed instance
    H = gen_H_random(dims=dims, rng=rng, scale=float(args.H_scale))
    gammas = [
        gen_marginal(
            d=int(d),
            rng=rng,
            kind=str(args.kind),
            hard_delta=float(args.hard_delta),
        )
        for d in dims
    ]

    print("=" * 78)
    print("BGDA eta sweep (via tune_bgda_eta)")
    print(f"dims={dims}, seed={args.seed}, eps={args.eps}, tol={args.tol}, T={args.T}")
    print(f"eta_grid={args.eta_grid}")
    print(f"instance: H_scale={args.H_scale}, kind={args.kind}, hard_delta={args.hard_delta}")
    print(f"prefer={args.prefer}, gauge_trace0={bool(args.gauge_trace0)}, project_pi={not args.no_project}, store_hist={bool(args.store_hist)}")
    print("=" * 78)

    tune = tune_bgda_eta(
        H=H,
        gammas=gammas,
        eps=float(args.eps),
        dims=dims,
        eta_grid=[float(x) for x in args.eta_grid],
        T=int(args.T),
        tol_tr=float(args.tol),
        gauge_trace0=bool(args.gauge_trace0),
        project_pi=(not args.no_project),
        store_hist=bool(args.store_hist),
        prefer=str(args.prefer),
    )

    # Print table
    for r in tune.records:
        print(
            f"eta={r.eta:>8.4g} | conv={str(bool(r.converged)):5s} | "
            f"iter={int(r.stop_iter):4d} | e_tr={float(r.final_e_tr):.3e} | "
            f"F={float(r.final_F):.3e} | calls={int(r.gibbs_calls):4d} | "
            f"time={float(r.time_sec):.3f}s | {r.message}"
        )

    best_eta = float(tune.best_eta)
    best_run = tune.best_run

    # best_run is a DBGAResult, containing full lists + pi + U_list, etc.
    best_final_e = float(best_run.e_tr_list[-1]) if best_run.e_tr_list else float("inf")
    best_final_F = float(best_run.F_list[-1]) if best_run.F_list else float("inf")
    best_calls = int(best_run.gibbs_calls)

    print("-" * 78)
    print(
        f"BEST eta = {best_eta} (prefer={args.prefer})\n"
        f"  converged={bool(best_run.converged)}\n"
        f"  final_e_tr={best_final_e:.3e}\n"
        f"  final_F={best_final_F:.3e}\n"
        f"  gibbs_calls={best_calls}\n"
        f"  stop_iter={len(best_run.e_tr_list) - 1 if best_run.e_tr_list else -1}"
    )

    if args.save_csv:
        _save_csv(args.save_csv, tune.records)
        print(f"\nSaved CSV to: {args.save_csv}")

    print("\nDone.")


if __name__ == "__main__":
    main()

