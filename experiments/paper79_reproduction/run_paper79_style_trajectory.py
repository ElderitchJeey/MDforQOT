"""Paper79-style trajectory data for one representative hard qubit instance.

The default instance is the Ising mixed qubit benchmark with N=7, eps=1e-3.
Rows are indexed by total Gibbs calls.  For warm-start methods, warmup Gibbs
calls are included in the x-axis budget.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.annealed_solvers import annealed_eqot_solver
from src.experiment_utils import parse_csv_ints
from src.SolverofEQOT import md_type_sinkhorn_potential, potential_marginal_kl_descent

from .metrics import entropic_dual_value
from .run_small_qubit_trend import make_small_instance


def build_eps_schedule(eps0: float, eps_final: float, q: float) -> List[float]:
    n_outer = int(round(np.log(float(eps0) / float(eps_final)) / np.log(float(q))))
    eps_schedule = [float(eps0) / (float(q) ** t) for t in range(n_outer + 1)]
    if not np.isclose(eps_schedule[-1], float(eps_final)):
        raise ValueError("eps0/q**n_outer does not match eps_final; adjust eps0, q, eps_final.")
    return eps_schedule


def rows_from_result(
    *,
    res: Any,
    method: str,
    mode: str,
    phase: str,
    small_kind: str,
    case: str,
    N: int,
    seed: int,
    eps: float,
    H: np.ndarray,
    gammas: List[np.ndarray],
    dims: List[int],
    gibbs_offset: int = 0,
    warmup_gibbs_calls: int = 0,
    record_every: int = 1,
) -> List[Dict[str, Any]]:
    F = list(getattr(res, "F_list", []) or [])
    e = list(getattr(res, "e_tr_list", []) or [])
    times = list(getattr(res, "times", []) or [])
    gibbs = list(getattr(res, "gibbs_calls_list", []) or [])
    U_hist = list(getattr(res, "U_hist", []) or [])
    n = max(len(F), len(e), len(times), len(gibbs))

    rows: List[Dict[str, Any]] = []
    for k in range(n):
        local_gibbs = int(gibbs[k]) if k < len(gibbs) else k
        if int(record_every) > 1 and local_gibbs % int(record_every) != 0 and k != n - 1:
            continue
        dual_value = ""
        if k < len(U_hist):
            try:
                dual_value = entropic_dual_value(H=H, U_list=U_hist[k], gammas=gammas, dims=dims, eps=eps)
            except Exception:
                dual_value = ""
        rows.append(
            {
                "small_kind": small_kind,
                "small_case": case,
                "N": int(N),
                "seed": int(seed),
                "eps": float(eps),
                "method": method,
                "mode": mode,
                "phase": phase,
                "iter": int(k),
                "local_gibbs_calls": local_gibbs,
                "warmup_gibbs_calls": int(warmup_gibbs_calls),
                "total_gibbs_calls": int(gibbs_offset) + local_gibbs,
                "time_sec": float(times[k]) if k < len(times) else "",
                "F_marg": float(F[k]) if k < len(F) else "",
                "e_tr": float(e[k]) if k < len(e) else "",
                "dual_value": dual_value,
                "converged_final": bool(getattr(res, "converged", False)),
            }
        )
    return rows


def rows_from_warmup_stages(
    *,
    warmup: Any,
    method: str,
    small_kind: str,
    case: str,
    N: int,
    seed: int,
    H: np.ndarray,
    gammas: List[np.ndarray],
    dims: List[int],
) -> List[Dict[str, Any]]:
    eps_list = list(getattr(warmup, "stage_eps_list", []) or [])
    gibbs = list(getattr(warmup, "gibbs_calls_list", []) or [])
    e = list(getattr(warmup, "e_tr_list", []) or [])
    F = list(getattr(warmup, "F_list", []) or [])
    times = list(getattr(warmup, "times", []) or [])
    U_hist = list(getattr(warmup, "U_hist", []) or [])

    rows: List[Dict[str, Any]] = []
    for k, eps in enumerate(eps_list):
        dual_value = ""
        if k < len(U_hist):
            try:
                dual_value = entropic_dual_value(H=H, U_list=U_hist[k], gammas=gammas, dims=dims, eps=float(eps))
            except Exception:
                dual_value = ""
        rows.append(
            {
                "small_kind": small_kind,
                "small_case": case,
                "N": int(N),
                "seed": int(seed),
                "eps": float(eps),
                "method": method,
                "mode": "warm_matched",
                "phase": "warmup_stage",
                "iter": int(k),
                "local_gibbs_calls": int(gibbs[k]) if k < len(gibbs) else "",
                "warmup_gibbs_calls": int(getattr(warmup, "gibbs_calls", 0) or 0),
                "total_gibbs_calls": int(gibbs[k]) if k < len(gibbs) else "",
                "time_sec": float(times[k]) if k < len(times) else "",
                "F_marg": float(F[k]) if k < len(F) else "",
                "e_tr": float(e[k]) if k < len(e) else "",
                "dual_value": dual_value,
                "converged_final": bool(getattr(warmup, "converged", False)),
            }
        )
    return rows


def remaining_budget(max_gibbs_calls: Optional[int], warmup: Any) -> Optional[int]:
    if max_gibbs_calls is None:
        return None
    used = int(getattr(warmup, "gibbs_calls", 0) or 0) if warmup is not None else 0
    return max(1, int(max_gibbs_calls) - used)


def run_kl(
    *,
    H: np.ndarray,
    gammas: List[np.ndarray],
    dims: List[int],
    eps: float,
    T: int,
    eta_rule: str,
    tol_tr: float,
    tol_F: float,
    jitter: float,
    max_gibbs_calls: Optional[int],
    track_dual: bool = True,
    U0: Optional[List[np.ndarray]] = None,
):
    return potential_marginal_kl_descent(
        H=H,
        gammas=gammas,
        eps=eps,
        dims=dims,
        T=T,
        eta=None,
        eta_rule=eta_rule,
        jitter_log=jitter,
        tol_tr=tol_tr,
        tol_F=tol_F,
        store_hist=track_dual,
        project_pi=True,
        U0=U0,
        max_gibbs_calls=max_gibbs_calls,
    )


def run_md(
    *,
    H: np.ndarray,
    gammas: List[np.ndarray],
    dims: List[int],
    eps: float,
    T: int,
    M: int,
    tol_tr: float,
    tol_F: float,
    tol_inner: float,
    jitter: float,
    max_gibbs_calls: Optional[int],
    track_dual: bool = True,
    U0: Optional[List[np.ndarray]] = None,
):
    return md_type_sinkhorn_potential(
        H=H,
        gammas=gammas,
        eps=eps,
        dims=dims,
        T_outer=T,
        tol_tr=tol_tr,
        tol_F=tol_F,
        jitter=jitter,
        M_inner=int(M),
        tol_inner=tol_inner,
        keep_U_hist=track_dual,
        project_pi=True,
        U0=U0,
        max_gibbs_calls=max_gibbs_calls,
    )


def run_case(args: argparse.Namespace) -> List[Dict[str, Any]]:
    H, gammas, dims, _ = make_small_instance(kind=args.kind, N=args.N, seed=args.seed, case=args.case)
    gammas = list(gammas)
    dims = list(dims)
    rows: List[Dict[str, Any]] = []
    eps_schedule = build_eps_schedule(args.eps0, args.eps, args.q)
    warm_eps_schedule = eps_schedule[:-1]

    kl_specs = [("KL eta=eps/N", "eps_over_N"), ("KL eta=eps", "eps")]
    for method, eta_rule in kl_specs:
        if "cold" in args.modes:
            cold = run_kl(
                H=H,
                gammas=gammas,
                dims=dims,
                eps=args.eps,
                T=args.T,
                eta_rule=eta_rule,
                tol_tr=args.tol_tr,
                tol_F=args.tol_F,
                jitter=args.jitter,
                max_gibbs_calls=args.max_gibbs_calls,
                track_dual=not args.no_dual,
            )
            cold.U_hist = cold.U_hist or []
            rows.extend(
                rows_from_result(
                    res=cold,
                    method=method,
                    mode="cold",
                    phase="target_eps",
                    small_kind=args.kind,
                    case=args.case,
                    N=args.N,
                    seed=args.seed,
                    eps=args.eps,
                    H=H,
                    gammas=gammas,
                    dims=dims,
                    record_every=args.record_every,
                )
            )

        if "warm" in args.modes:
            warmup = annealed_eqot_solver(
                H=H,
                target_marginals=gammas,
                dims=dims,
                eps_schedule=warm_eps_schedule,
                method="kl",
                kl_eta_rule=eta_rule,
                max_inner=args.T,
                max_gibbs_calls=max(1, int(args.max_gibbs_calls) - 1) if args.max_gibbs_calls else None,
                inner_tol=args.warm_tol,
                final_tol=args.warm_tol,
                tol_F=None,
                jitter=args.jitter,
                project_pi=True,
                return_history=True,
            )
            rows.extend(
                rows_from_warmup_stages(
                    warmup=warmup,
                    method=method,
                    small_kind=args.kind,
                    case=args.case,
                    N=args.N,
                    seed=args.seed,
                    H=H,
                    gammas=gammas,
                    dims=dims,
                )
            )
            warm = run_kl(
                H=H,
                gammas=gammas,
                dims=dims,
                eps=args.eps,
                T=args.T,
                eta_rule=eta_rule,
                tol_tr=args.tol_tr,
                tol_F=args.tol_F,
                jitter=args.jitter,
                max_gibbs_calls=remaining_budget(args.max_gibbs_calls, warmup),
                track_dual=not args.no_dual,
                U0=warmup.U_list,
            )
            warm.U_hist = warm.U_hist or []
            rows.extend(
                rows_from_result(
                    res=warm,
                    method=method,
                    mode="warm_matched",
                    phase="target_eps",
                    small_kind=args.kind,
                    case=args.case,
                    N=args.N,
                    seed=args.seed,
                    eps=args.eps,
                    H=H,
                    gammas=gammas,
                    dims=dims,
                    gibbs_offset=int(warmup.gibbs_calls),
                    warmup_gibbs_calls=int(warmup.gibbs_calls),
                    record_every=args.record_every,
                )
            )

    for M in args.M_list:
        method = f"MD M={int(M)}"
        if "cold" in args.modes:
            cold = run_md(
                H=H,
                gammas=gammas,
                dims=dims,
                eps=args.eps,
                T=args.T,
                M=int(M),
                tol_tr=args.tol_tr,
                tol_F=args.tol_F,
                tol_inner=args.tol_inner,
                jitter=args.jitter,
                max_gibbs_calls=args.max_gibbs_calls,
                track_dual=not args.no_dual,
            )
            rows.extend(
                rows_from_result(
                    res=cold,
                    method=method,
                    mode="cold",
                    phase="target_eps",
                    small_kind=args.kind,
                    case=args.case,
                    N=args.N,
                    seed=args.seed,
                    eps=args.eps,
                    H=H,
                    gammas=gammas,
                    dims=dims,
                    record_every=args.record_every,
                )
            )

        if "warm" in args.modes:
            warmup = annealed_eqot_solver(
                H=H,
                target_marginals=gammas,
                dims=dims,
                eps_schedule=warm_eps_schedule,
                method="md_sinkhorn",
                M_inner=int(M),
                max_inner=args.T,
                max_gibbs_calls=max(1, int(args.max_gibbs_calls) - 1) if args.max_gibbs_calls else None,
                inner_tol=args.warm_tol,
                final_tol=args.warm_tol,
                tol_F=None,
                jitter=args.jitter,
                project_pi=True,
                return_history=True,
            )
            rows.extend(
                rows_from_warmup_stages(
                    warmup=warmup,
                    method=method,
                    small_kind=args.kind,
                    case=args.case,
                    N=args.N,
                    seed=args.seed,
                    H=H,
                    gammas=gammas,
                    dims=dims,
                )
            )
            warm = run_md(
                H=H,
                gammas=gammas,
                dims=dims,
                eps=args.eps,
                T=args.T,
                M=int(M),
                tol_tr=args.tol_tr,
                tol_F=args.tol_F,
                tol_inner=args.tol_inner,
                jitter=args.jitter,
                max_gibbs_calls=remaining_budget(args.max_gibbs_calls, warmup),
                track_dual=not args.no_dual,
                U0=warmup.U_list,
            )
            rows.extend(
                rows_from_result(
                    res=warm,
                    method=method,
                    mode="warm_matched",
                    phase="target_eps",
                    small_kind=args.kind,
                    case=args.case,
                    N=args.N,
                    seed=args.seed,
                    eps=args.eps,
                    H=H,
                    gammas=gammas,
                    dims=dims,
                    gibbs_offset=int(warmup.gibbs_calls),
                    warmup_gibbs_calls=int(warmup.gibbs_calls),
                    record_every=args.record_every,
                )
            )

    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Paper79-style hard-case trajectory")
    parser.add_argument("--kind", choices=["ising", "random"], default="ising")
    parser.add_argument("--N", type=int, default=7)
    parser.add_argument("--case", choices=["ground", "excited", "mixed"], default="mixed")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--eps0", type=float, default=1.0)
    parser.add_argument("--q", type=float, default=10.0)
    parser.add_argument("--T", type=int, default=100000)
    parser.add_argument("--max_gibbs_calls", type=int, default=100000)
    parser.add_argument("--record_every", type=int, default=1)
    parser.add_argument("--no_dual", action="store_true")
    parser.add_argument("--modes", default="cold,warm")
    parser.add_argument("--M_list", default="1,2,5")
    parser.add_argument("--tol_tr", type=float, default=1e-4)
    parser.add_argument("--tol_F", type=float, default=1e-8)
    parser.add_argument("--tol_inner", type=float, default=1e-4)
    parser.add_argument("--warm_tol", type=float, default=1e-2)
    parser.add_argument("--jitter", type=float, default=1e-10)
    parser.add_argument("--out", type=Path, default=Path("results") / "paper79_style_N7_trajectory.csv")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.modes = [x.strip().lower() for x in args.modes.split(",") if x.strip()]
    for mode in args.modes:
        if mode not in {"cold", "warm"}:
            raise ValueError("--modes entries must be cold or warm")
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
