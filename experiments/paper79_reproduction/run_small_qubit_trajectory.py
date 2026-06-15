"""Trajectory data for small paper79-like qubit examples."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from src.experiment_utils import parse_csv_ints
from src.SolverofEQOT import md_type_sinkhorn_potential, potential_marginal_kl_descent

from .run_small_qubit_trend import make_small_instance


def rows_from_result(*, method: str, res: Any, small_kind: str, case: str, N: int, eps: float, seed: int) -> List[Dict[str, Any]]:
    F = list(getattr(res, "F_list", []) or [])
    e = list(getattr(res, "e_tr_list", []) or [])
    times = list(getattr(res, "times", []) or [])
    gibbs = list(getattr(res, "gibbs_calls_list", []) or [])
    n = max(len(F), len(e), len(times), len(gibbs))

    rows: List[Dict[str, Any]] = []
    for k in range(n):
        rows.append(
            {
                "small_kind": small_kind,
                "small_case": case,
                "N": int(N),
                "seed": int(seed),
                "eps": float(eps),
                "method": method,
                "iter": k,
                "gibbs_calls": int(gibbs[k]) if k < len(gibbs) else k,
                "time_sec": float(times[k]) if k < len(times) else "",
                "F_marg": float(F[k]) if k < len(F) else "",
                "e_tr": float(e[k]) if k < len(e) else "",
                "converged_final": bool(getattr(res, "converged", False)),
            }
        )
    return rows


def run_case(args: argparse.Namespace) -> List[Dict[str, Any]]:
    H, gammas, dims, _ = make_small_instance(
        kind=args.kind,
        N=args.N,
        seed=args.seed,
        case=args.case,
    )

    rows: List[Dict[str, Any]] = []
    stop_tol_f = None if args.force_full_budget else args.tol_f
    stop_tol_tr = None if args.force_full_budget else args.tol_tr

    if "kl" in args.methods:
        res_kl = potential_marginal_kl_descent(
            H=H,
            gammas=gammas,
            eps=args.eps,
            dims=dims,
            T=args.T,
            eta=args.eta_kl,
            eta_rule=args.eta_kl_rule,
            jitter_log=args.jitter,
            tol_tr=stop_tol_tr,
            tol_F=stop_tol_f,
            store_hist=False,
            project_pi=True,
        )
        rows.extend(
            rows_from_result(
                method="KL descent",
                res=res_kl,
                small_kind=args.kind,
                case=args.case,
                N=args.N,
                eps=args.eps,
                seed=args.seed,
            )
        )

    if "md" in args.methods:
        for M in args.M_list:
            res_md = md_type_sinkhorn_potential(
                H=H,
                gammas=gammas,
                eps=args.eps,
                dims=dims,
                T_outer=args.T,
                tol_tr=stop_tol_tr if stop_tol_tr is not None else -1.0,
                tol_F=stop_tol_f,
                jitter=args.jitter,
                eta_inner=1.0,
                M_inner=int(M),
                keep_U_hist=False,
                keep_pi_hist=False,
                tol_inner=None if args.force_full_budget else args.tol_inner,
                project_pi=True,
            )
            rows.extend(
                rows_from_result(
                    method=f"MD-Sinkhorn (M={int(M)})",
                    res=res_md,
                    small_kind=args.kind,
                    case=args.case,
                    N=args.N,
                    eps=args.eps,
                    seed=args.seed,
                )
            )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Small qubit trajectory data")
    parser.add_argument("--kind", choices=["ising", "random"], default="ising")
    parser.add_argument("--N", type=int, default=4)
    parser.add_argument("--case", choices=["ground", "excited", "mixed"], default="ground")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--T", type=int, default=300)
    parser.add_argument("--methods", default="kl,md")
    parser.add_argument("--M_list", default="1,2,5")
    parser.add_argument("--eta_kl", type=float, default=None)
    parser.add_argument("--eta_kl_rule", choices=["eps_over_N", "eps"], default="eps_over_N")
    parser.add_argument("--tol_f", type=float, default=1e-8)
    parser.add_argument("--tol_tr", type=float, default=1e-8)
    parser.add_argument("--tol_inner", type=float, default=1e-4)
    parser.add_argument("--jitter", type=float, default=1e-10)
    parser.add_argument("--force_full_budget", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results") / "small_qubit_trajectory_eps1e-3.csv",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.methods = [x.strip().lower() for x in args.methods.split(",") if x.strip()]
    args.M_list = parse_csv_ints(args.M_list)

    rows = run_case(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
