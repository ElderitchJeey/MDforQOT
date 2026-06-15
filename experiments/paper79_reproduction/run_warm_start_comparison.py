"""Warm-start comparison runner for paper79 reproduction instances."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.annealed_solvers import annealed_eqot_solver
from src.experiment_utils import first_hit_index, parse_csv_ints, save_final_state
from src.SolverofEQOT import F_marg, md_type_sinkhorn_potential, marginal_trace_errors, potential_marginal_kl_descent

from .metrics import entropic_dual_value, entropic_primal_cost, primal_cost, von_neumann_entropy_term
from .run_small_qubit_trend import make_small_instance
from .run_wasserstein_trend import make_wasserstein_instance


def summarize_result(
    *,
    method: str,
    mode: str,
    res: Any,
    H: np.ndarray,
    gammas: List[np.ndarray],
    dims: List[int],
    eps: float,
    warmup: Any = None,
) -> Dict[str, Any]:
    pi = res.pi
    per_i = marginal_trace_errors(pi, gammas, dims)
    warmup_gibbs = int(getattr(warmup, "gibbs_calls", 0) or 0) if warmup is not None else 0
    warmup_time = float((getattr(warmup, "times", []) or [0.0])[-1]) if warmup is not None else 0.0
    run_gibbs = int(getattr(res, "gibbs_calls", 0) or 0)
    run_time = float((getattr(res, "times", []) or [0.0])[-1])
    e_list = list(getattr(res, "e_tr_list", []) or [])
    gibbs_list = list(getattr(res, "gibbs_calls_list", []) or [])
    try:
        dual_value = entropic_dual_value(
            H=H,
            U_list=getattr(res, "U_list"),
            gammas=gammas,
            dims=dims,
            eps=float(eps),
        )
    except Exception:
        dual_value = ""
    linear_cost = primal_cost(H, pi)
    entropy_term = von_neumann_entropy_term(pi)
    entropic_primal = entropic_primal_cost(H, pi, eps)
    primal_dual_gap = ""
    try:
        if dual_value not in (None, ""):
            primal_dual_gap = float(entropic_primal) - float(dual_value)
    except Exception:
        primal_dual_gap = ""
    row = {
        "method": method,
        "mode": mode,
        "converged": bool(getattr(res, "converged", False)),
        "iters": max(0, len(getattr(res, "e_tr_list", []) or []) - 1),
        "gibbs_calls": run_gibbs,
        "warmup_gibbs_calls": warmup_gibbs,
        "total_with_warm_gibbs": warmup_gibbs + run_gibbs,
        "time_sec": run_time,
        "warmup_time_sec": warmup_time,
        "total_with_warm_time_sec": warmup_time + run_time,
        "final_F_marg": float(F_marg(pi, gammas, dims)),
        "final_e_tr": float(np.max(per_i)),
        "final_trace_sum": float(np.sum(per_i)),
        "final_cost": entropic_primal,
        "final_entropic_primal": entropic_primal,
        "final_linear_cost": linear_cost,
        "final_entropy": entropy_term,
        "final_dual_value": dual_value,
        "final_primal_dual_gap": primal_dual_gap,
        "stage_gibbs_calls": ";".join(str(x) for x in (getattr(warmup, "stage_gibbs_calls_list", None) or [])),
        "stage_iters": ";".join(str(x) for x in (getattr(warmup, "stage_iters_list", None) or [])),
        "eps_schedule": ";".join(str(x) for x in (getattr(warmup, "stage_eps_list", None) or getattr(warmup, "eps_schedule", None) or [])),
    }
    for tol, label in [(1e-3, "1em03"), (1e-4, "1em04")]:
        idx = first_hit_index(e_list, tol) if e_list else -1
        hit_gibbs = int(gibbs_list[idx]) if idx >= 0 and len(gibbs_list) == len(e_list) else -1
        row[f"hit_tr_le_{label}"] = bool(idx >= 0) if e_list else ""
        row[f"hit_tr_iter_le_{label}"] = idx if e_list else ""
        row[f"hit_tr_gibbs_le_{label}"] = hit_gibbs
        row[f"total_hit_tr_gibbs_le_{label}"] = warmup_gibbs + hit_gibbs if hit_gibbs >= 0 else -1
    return row


def make_instance(args: argparse.Namespace):
    if args.instance == "qubit":
        H, gammas, dims, label = make_small_instance(kind=args.kind, N=args.N, seed=args.seed, case=args.case)
        metadata = {"small_kind": args.kind, "small_case": args.case, "N": int(args.N), "seed": int(args.seed)}
    elif args.instance == "wasserstein":
        H, gammas, dims, label = make_wasserstein_instance(args.d, args.wasserstein_case, args.jitter)
        metadata = {"wasserstein_case": args.wasserstein_case.upper(), "d": int(args.d)}
    else:
        raise ValueError("--instance must be qubit or wasserstein")
    return H, list(gammas), list(dims), label, metadata


def build_eps_schedule(eps0: float, eps_final: float, q: float) -> List[float]:
    n_outer = int(round(np.log(eps0 / eps_final) / np.log(q)))
    eps_schedule = [eps0 / (q ** t) for t in range(n_outer + 1)]
    if not np.isclose(eps_schedule[-1], eps_final):
        raise ValueError("eps0/q**n_outer does not match eps_final; adjust eps0, q, eps_final.")
    return [float(eps) for eps in eps_schedule]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Warm-start comparison for cold vs matched annealed warm starts")
    parser.add_argument("--instance", choices=["qubit", "wasserstein"], default="qubit")
    parser.add_argument("--kind", choices=["ising", "random"], default="random")
    parser.add_argument("--N", type=int, default=3)
    parser.add_argument("--case", choices=["ground", "excited", "mixed"], default="mixed")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--d", type=int, default=20)
    parser.add_argument("--wasserstein_case", choices=["WP", "WM", "WC", "WG"], default="WP")
    parser.add_argument("--eps_final", type=float, default=1e-3)
    parser.add_argument("--eps0", type=float, default=1.0)
    parser.add_argument("--q", type=float, default=10.0)
    parser.add_argument("--max_inner", type=int, default=2000)
    parser.add_argument("--max_gibbs_calls", type=int, default=None)
    parser.add_argument("--M_list", default="1,2,5")
    parser.add_argument("--inner_tol", type=float, default=1e-2)
    parser.add_argument("--final_tol", type=float, default=1e-8)
    parser.add_argument("--jitter", type=float, default=1e-10)
    parser.add_argument("--tol_tr", type=float, default=1e-8)
    parser.add_argument("--tol_F", type=float, default=1e-8)
    parser.add_argument("--skip_cold", action="store_true")
    parser.add_argument("--save_final_state", action="store_true")
    parser.add_argument("--state_dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("results") / "warm_start_comparison.csv")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    H, gammas, dims, label, metadata = make_instance(args)
    M_list = parse_csv_ints(args.M_list)
    eps_schedule = build_eps_schedule(args.eps0, args.eps_final, args.q)
    warm_eps_schedule = eps_schedule[:-1]

    def warm_budget() -> Optional[int]:
        if args.max_gibbs_calls is None:
            return None
        return max(1, int(args.max_gibbs_calls) - 1)

    def remaining_budget(warmup: Any) -> Optional[int]:
        if args.max_gibbs_calls is None:
            return None
        used = int(getattr(warmup, "gibbs_calls", 0) or 0) if warmup is not None else 0
        return max(1, int(args.max_gibbs_calls) - used)

    rows: List[Dict[str, Any]] = []
    row_results: List[Any] = []

    def add_row(row: Dict[str, Any], res: Any) -> None:
        rows.append(row)
        row_results.append(res)

    kl_specs = [("KL descent (eta=eps/N)", "eps_over_N"), ("KL descent (eta=eps)", "eps")]
    for label_method, eta_rule in kl_specs:
        if not args.skip_cold:
            cold = potential_marginal_kl_descent(
                H=H,
                gammas=gammas,
                eps=args.eps_final,
                dims=dims,
                T=args.max_inner,
                eta=None,
                eta_rule=eta_rule,
                jitter_log=args.jitter,
                tol_tr=args.tol_tr,
                tol_F=args.tol_F,
                project_pi=True,
                max_gibbs_calls=args.max_gibbs_calls,
            )
            add_row(summarize_result(method=label_method, mode="cold", res=cold, H=H, gammas=gammas, dims=dims, eps=args.eps_final), cold)
        warmup = None
        if warm_eps_schedule:
            warmup = annealed_eqot_solver(
                H=H,
                target_marginals=gammas,
                dims=dims,
                eps_schedule=warm_eps_schedule,
                method="kl",
                kl_eta_rule=eta_rule,
                max_inner=args.max_inner,
                inner_tol=args.inner_tol,
                final_tol=args.inner_tol,
                jitter=args.jitter,
                tol_F=None,
                project_pi=True,
                max_gibbs_calls=warm_budget(),
            )
        warm = potential_marginal_kl_descent(
            H=H,
            gammas=gammas,
            eps=args.eps_final,
            dims=dims,
            T=args.max_inner,
            eta=None,
            eta_rule=eta_rule,
            jitter_log=args.jitter,
            tol_tr=args.tol_tr,
            tol_F=args.tol_F,
            project_pi=True,
            U0=warmup.U_list if warmup is not None else None,
            max_gibbs_calls=remaining_budget(warmup),
        )
        add_row(summarize_result(method=label_method, mode="warm_matched", res=warm, warmup=warmup, H=H, gammas=gammas, dims=dims, eps=args.eps_final), warm)

    for M in M_list:
        if not args.skip_cold:
            cold = md_type_sinkhorn_potential(
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
                max_gibbs_calls=args.max_gibbs_calls,
            )
            add_row(summarize_result(method=f"MD-Sinkhorn (M={M})", mode="cold", res=cold, H=H, gammas=gammas, dims=dims, eps=args.eps_final), cold)
        warmup = None
        if warm_eps_schedule:
            warmup = annealed_eqot_solver(
                H=H,
                target_marginals=gammas,
                dims=dims,
                eps_schedule=warm_eps_schedule,
                method="md_sinkhorn",
                M_inner=M,
                max_inner=args.max_inner,
                inner_tol=args.inner_tol,
                final_tol=args.inner_tol,
                jitter=args.jitter,
                tol_F=None,
                project_pi=True,
                max_gibbs_calls=warm_budget(),
            )
        warm = md_type_sinkhorn_potential(
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
            U0=warmup.U_list if warmup is not None else None,
            max_gibbs_calls=remaining_budget(warmup),
        )
        add_row(summarize_result(method=f"MD-Sinkhorn (M={M})", mode="warm_matched", res=warm, warmup=warmup, H=H, gammas=gammas, dims=dims, eps=args.eps_final), warm)

    for row, res in zip(rows, row_results):
        row.update(metadata)
        row["paper79_label"] = label
        row["eps_final"] = float(args.eps_final)
        row["eps0"] = float(args.eps0)
        row["q"] = float(args.q)
        row["max_inner"] = int(args.max_inner)
        row["max_gibbs_calls"] = "" if args.max_gibbs_calls is None else int(args.max_gibbs_calls)
        row["warm_method"] = "matched"
        row["warm_variant"] = row["method"]
        state_path = save_final_state(
            args=args,
            row=row,
            res=res,
            H=H,
            gammas=gammas,
            dims=dims,
            eps=args.eps_final,
        )
        if state_path:
            row["final_state_path"] = state_path

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
