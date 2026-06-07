"""Warm-start demo for small-epsilon EQOT on a 3-qubit mixed instance."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np

from src.SolverofEQOT import F_marg, md_type_sinkhorn_potential, marginal_trace_errors, potential_marginal_kl_descent
from src.epsilon_continuation import annealed_quantum_kl_descent

from .run_small_qubit_trend import make_small_instance


def summarize_result(
    *,
    method: str,
    mode: str,
    res: Any,
    H: np.ndarray,
    gammas: List[np.ndarray],
    dims: List[int],
    eps_final: float,
    warmup: Any = None,
) -> Dict[str, Any]:
    pi = res.pi
    per_i = marginal_trace_errors(pi, gammas, dims)
    warmup_gibbs = int(getattr(warmup, "gibbs_calls", 0) or 0) if warmup is not None else 0
    warmup_time = float((getattr(warmup, "times", []) or [0.0])[-1]) if warmup is not None else 0.0
    return {
        "method": method,
        "mode": mode,
        "eps_final": float(eps_final),
        "converged": bool(getattr(res, "converged", False)),
        "iters": max(0, len(getattr(res, "e_tr_list", []) or []) - 1),
        "gibbs_calls": int(getattr(res, "gibbs_calls", 0) or 0),
        "warmup_gibbs_calls": warmup_gibbs,
        "total_with_warm_gibbs": warmup_gibbs + int(getattr(res, "gibbs_calls", 0) or 0),
        "time_sec": float((getattr(res, "times", []) or [0.0])[-1]),
        "warmup_time_sec": warmup_time,
        "total_with_warm_time_sec": warmup_time + float((getattr(res, "times", []) or [0.0])[-1]),
        "final_F_marg": float(F_marg(pi, gammas, dims)),
        "final_e_tr": float(np.max(per_i)),
        "final_trace_sum": float(np.sum(per_i)),
        "final_cost": float(np.real(np.trace(H @ pi))),
        "stage_gibbs_calls": ";".join(str(x) for x in (getattr(res, "stage_gibbs_calls_list", None) or [])),
        "stage_iters": ";".join(str(x) for x in (getattr(res, "stage_iters_list", None) or [])),
        "eps_schedule": ";".join(str(x) for x in (getattr(res, "eps_schedule", None) or [])),
    }


def stage_tol(*, eps: float, is_final: bool, inner_tol: float, final_tol: float) -> float:
    return float(final_tol if is_final else inner_tol)


def run_annealed_md_warm(
    *,
    H: np.ndarray,
    gammas: List[np.ndarray],
    dims: List[int],
    eps_schedule: List[float],
    M_inner: int,
    max_inner: int,
    inner_tol: float,
    final_tol: float,
    jitter: float,
) -> Any:
    U0 = None
    stage_gibbs: List[int] = []
    stage_iters: List[int] = []
    total_gibbs = 0
    total_time = 0.0
    final_res = None
    for stage, eps in enumerate(eps_schedule):
        tol = stage_tol(eps=eps, is_final=(stage == len(eps_schedule) - 1), inner_tol=inner_tol, final_tol=final_tol)
        res = md_type_sinkhorn_potential(
            H=H,
            gammas=gammas,
            eps=eps,
            dims=dims,
            T_outer=max_inner,
            tol_tr=tol,
            tol_F=None,
            jitter=jitter,
            M_inner=M_inner,
            tol_inner=1e-4,
            project_pi=True,
            U0=U0,
        )
        stage_gibbs.append(int(res.gibbs_calls))
        stage_iters.append(max(0, len(res.e_tr_list) - 1))
        total_gibbs += int(res.gibbs_calls)
        total_time += float((res.times or [0.0])[-1])
        U0 = res.U_list
        final_res = res
    if final_res is None:
        raise RuntimeError("empty epsilon schedule")
    return SimpleNamespace(
        pi=final_res.pi,
        U_list=final_res.U_list,
        F_list=final_res.F_list,
        e_tr_list=final_res.e_tr_list,
        per_i_tr_list=final_res.per_i_tr_list,
        times=[total_time],
        converged=final_res.converged,
        gibbs_calls=total_gibbs,
        gibbs_calls_list=final_res.gibbs_calls_list,
        stage_gibbs_calls_list=stage_gibbs,
        stage_iters_list=stage_iters,
        eps_schedule=eps_schedule,
    )


def run_warm_path(args: argparse.Namespace, *, H: np.ndarray, gammas: List[np.ndarray], dims: List[int], eps_schedule: List[float], n_outer: int) -> Any:
    if args.warm_method == "kl":
        warm_eta = None if args.warm_kl_eta_rule == "eps_over_N" else 1.0
        return annealed_quantum_kl_descent(
            H=H,
            target_marginals=gammas,
            dims=dims,
            eps0=args.eps0,
            q=args.q,
            n_outer=n_outer,
            eta=warm_eta,
            inner_tol=args.inner_tol,
            final_tol=args.final_tol,
            max_inner=args.max_inner,
            rho_reg=args.jitter,
            log_floor=args.jitter,
            return_history=False,
            residual="trace_max",
        )
    return run_annealed_md_warm(
        H=H,
        gammas=gammas,
        dims=dims,
        eps_schedule=eps_schedule,
        M_inner=args.warm_M,
        max_inner=args.max_inner,
        inner_tol=args.inner_tol,
        final_tol=args.final_tol,
        jitter=args.jitter,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Warm-start demo: 3-qubit mixed eps=1e-3")
    parser.add_argument("--kind", choices=["ising", "random"], default="random")
    parser.add_argument("--N", type=int, default=3)
    parser.add_argument("--case", choices=["ground", "excited", "mixed"], default="mixed")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eps_final", type=float, default=1e-3)
    parser.add_argument("--eps0", type=float, default=1.0)
    parser.add_argument("--q", type=float, default=10.0)
    parser.add_argument("--max_inner", type=int, default=2000)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--inner_tol", type=float, default=1e-3)
    parser.add_argument("--final_tol", type=float, default=1e-8)
    parser.add_argument("--warm_method", choices=["kl", "md"], default="kl")
    parser.add_argument("--warm_kl_eta_rule", choices=["eps_over_N", "eps"], default="eps_over_N")
    parser.add_argument("--warm_M", type=int, default=1)
    parser.add_argument("--jitter", type=float, default=1e-10)
    parser.add_argument("--tol_tr", type=float, default=1e-8)
    parser.add_argument("--tol_F", type=float, default=1e-8)
    parser.add_argument("--out", type=Path, default=Path("results") / "warm_start_demo_N3_mixed_eps1e-3.csv")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    H, gammas, dims, label = make_small_instance(kind=args.kind, N=args.N, seed=args.seed, case=args.case)
    n_outer = int(round(np.log(args.eps0 / args.eps_final) / np.log(args.q)))
    eps_schedule = [args.eps0 / (args.q ** t) for t in range(n_outer + 1)]
    if not np.isclose(eps_schedule[-1], args.eps_final):
        raise ValueError("eps0/q**n_outer does not match eps_final; adjust eps0, q, eps_final.")

    rows: List[Dict[str, Any]] = []

    kl_cold = potential_marginal_kl_descent(
        H=H,
        gammas=gammas,
        eps=args.eps_final,
        dims=dims,
        T=args.max_inner,
        eta=args.eta * args.eps_final,
        jitter_log=args.jitter,
        tol_tr=args.tol_tr,
        tol_F=args.tol_F,
        project_pi=True,
    )
    rows.append(summarize_result(method="KL descent", mode="cold", res=kl_cold, H=H, gammas=gammas, dims=dims, eps_final=args.eps_final))

    warm_path = run_warm_path(args, H=H, gammas=gammas, dims=dims, eps_schedule=eps_schedule, n_outer=n_outer)

    kl_warm = potential_marginal_kl_descent(
        H=H,
        gammas=gammas,
        eps=args.eps_final,
        dims=dims,
        T=args.max_inner,
        eta=args.eta * args.eps_final,
        jitter_log=args.jitter,
        tol_tr=args.tol_tr,
        tol_F=args.tol_F,
        project_pi=True,
        U0=warm_path.U_list,
    )
    rows.append(
        summarize_result(
            method="KL descent",
            mode=f"warm_by_{args.warm_method}",
            res=kl_warm,
            warmup=warm_path,
            H=H,
            gammas=gammas,
            dims=dims,
            eps_final=args.eps_final,
        )
    )

    for M in (1, 2):
        md_cold = md_type_sinkhorn_potential(
            H=H,
            gammas=gammas,
            eps=args.eps_final,
            dims=dims,
            T_outer=args.max_inner,
            tol_tr=args.tol_tr,
            tol_F=args.tol_F,
            jitter=args.jitter,
            M_inner=M,
            tol_inner=1e-4,
            project_pi=True,
        )
        rows.append(
            summarize_result(method=f"MD-Sinkhorn (M={M})", mode="cold", res=md_cold, H=H, gammas=gammas, dims=dims, eps_final=args.eps_final)
        )

        md_warm = md_type_sinkhorn_potential(
            H=H,
            gammas=gammas,
            eps=args.eps_final,
            dims=dims,
            T_outer=args.max_inner,
            tol_tr=args.tol_tr,
            tol_F=args.tol_F,
            jitter=args.jitter,
            M_inner=M,
            tol_inner=1e-4,
            project_pi=True,
            U0=warm_path.U_list,
        )
        rows.append(
            summarize_result(
                method=f"MD-Sinkhorn (M={M})",
                mode=f"warm_by_{args.warm_method}",
                res=md_warm,
                warmup=warm_path,
                H=H,
                gammas=gammas,
                dims=dims,
                eps_final=args.eps_final,
            )
        )

    for row in rows:
        row["instance"] = label
        row["N"] = int(args.N)
        row["kind"] = args.kind
        row["case"] = args.case
        row["seed"] = int(args.seed)
        row["max_inner"] = int(args.max_inner)
        row["warm_method"] = args.warm_method
        row["warm_kl_eta_rule"] = args.warm_kl_eta_rule
        row["warm_M"] = int(args.warm_M)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(row)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
