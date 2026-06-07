"""Annealed wrappers for EQOT potential solvers.

This module is intentionally method-neutral.  It runs an epsilon schedule and
uses the final potentials from one stage to initialize the next stage.  The
inner stage solver can currently be KL descent or MD-Sinkhorn.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.SolverofEQOT import (
    F_marg,
    gibbs_state_from_potentials,
    marginal_trace_errors,
    md_type_sinkhorn_potential,
    potential_marginal_kl_descent,
)
from src.linalg import hermitianize, proj_to_density


@dataclass
class AnnealedSolverResult:
    """Solver-style output for epsilon-continuation runs."""

    F_list: List[float]
    e_tr_list: List[float]
    per_i_tr_list: List[np.ndarray]
    times: List[float]
    pi: np.ndarray
    U_list: List[np.ndarray]
    U_hist: Optional[List[List[np.ndarray]]] = None
    pi_list: Optional[List[np.ndarray]] = None
    converged: bool = False
    gibbs_calls: int = 0
    gibbs_calls_list: Optional[List[int]] = None
    eps_schedule: Optional[List[float]] = None
    eps_final: float = 0.0
    stage_history: Optional[List[Dict[str, Any]]] = None
    stage_gibbs_calls_list: Optional[List[int]] = None
    stage_iters_list: Optional[List[int]] = None
    stage_eps_list: Optional[List[float]] = None
    method: str = ""
    method_variant: str = ""
    metadata: Optional[Dict[str, Any]] = None

    def __getitem__(self, key: str) -> Any:
        aliases = {
            "potentials": "U_list",
            "history": "stage_history",
            "total_gibbs_calls": "gibbs_calls",
        }
        return getattr(self, aliases.get(key, key))


def annealing_eps_schedule(*, eps0: float, q: float, n_outer: int) -> List[float]:
    """Return eps_t = eps0 / q**t for t=0,...,n_outer."""

    if eps0 <= 0:
        raise ValueError("eps0 must be positive.")
    if q <= 1:
        raise ValueError("q must be > 1.")
    if n_outer < 0:
        raise ValueError("n_outer must be nonnegative.")
    return [float(eps0) / (float(q) ** t) for t in range(int(n_outer) + 1)]


def geometric_eps_schedule(
    *,
    eps_start: float,
    eps_target: float,
    factor: float = 0.5,
    include_start: bool = True,
) -> List[float]:
    """Build a decreasing geometric epsilon schedule."""

    if eps_start <= 0 or eps_target <= 0:
        raise ValueError("eps_start and eps_target must be positive.")
    if eps_start < eps_target:
        raise ValueError("eps_start must be >= eps_target.")
    if not (0 < factor < 1):
        raise ValueError("factor must lie in (0, 1).")

    schedule: List[float] = [float(eps_start)] if include_start else []
    eps = float(eps_start)
    while eps * factor > eps_target:
        eps *= factor
        schedule.append(float(eps))
    if not schedule or not np.isclose(schedule[-1], float(eps_target), rtol=1e-12, atol=0.0):
        schedule.append(float(eps_target))
    return schedule


def validate_eps_schedule(eps_schedule: Sequence[float]) -> List[float]:
    """Validate and normalize a user-provided nonincreasing epsilon schedule."""

    out = [float(eps) for eps in eps_schedule]
    if not out:
        raise ValueError("eps_schedule cannot be empty.")
    if any(eps <= 0 for eps in out):
        raise ValueError("all epsilon values must be positive.")
    for prev, curr in zip(out, out[1:]):
        if curr > prev:
            raise ValueError("eps_schedule must be nonincreasing.")
    return out


def _stage_tol(
    *,
    eps: float,
    is_final_stage: bool,
    tol_mode: str,
    inner_tol: float,
    final_tol: float,
    C_tol: float,
    p: float,
) -> float:
    if is_final_stage:
        return float(final_tol)
    if tol_mode == "fixed":
        return float(inner_tol)
    if tol_mode == "annealed":
        return float(max(float(final_tol), float(C_tol) * (float(eps) ** float(p))))
    raise ValueError("tol_mode must be 'fixed' or 'annealed'.")


def _regularize_targets(targets: List[np.ndarray], dims: List[int], rho_reg: float) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for i, rho in enumerate(targets):
        rho_i = proj_to_density(np.asarray(rho, dtype=complex), jitter=0.0)
        if rho_reg > 0:
            d = int(dims[i])
            rho_i = (1.0 - float(rho_reg)) * rho_i + float(rho_reg) * np.eye(d, dtype=complex) / float(d)
            rho_i = proj_to_density(rho_i, jitter=0.0)
        out.append(rho_i)
    return out


def _initial_potentials(init: str, dims: List[int], U0: Optional[List[np.ndarray]]) -> List[np.ndarray]:
    if U0 is None:
        if init != "zero":
            raise ValueError("annealed_solver currently supports init='zero' unless U0 is supplied.")
        return [np.zeros((d, d), dtype=complex) for d in dims]
    if len(U0) != len(dims):
        raise ValueError("len(U0) must match len(dims).")
    out: List[np.ndarray] = []
    for i, Ui in enumerate(U0):
        Ui_arr = np.asarray(Ui, dtype=complex)
        if Ui_arr.shape != (dims[i], dims[i]):
            raise ValueError(f"U0[{i}] must have shape {(dims[i], dims[i])}, got {Ui_arr.shape}")
        out.append(hermitianize(Ui_arr.copy()))
    return out


def _method_variant(method: str, kl_eta_rule: str, M_inner: int) -> str:
    if method == "kl":
        return "KL descent (eta=eps/N)" if kl_eta_rule == "eps_over_N" else "KL descent (eta=eps)"
    if method == "md_sinkhorn":
        return f"MD-Sinkhorn (M={int(M_inner)})"
    raise ValueError("method must be 'kl' or 'md_sinkhorn'.")


def annealed_eqot_solver(
    *,
    H: np.ndarray,
    target_marginals: List[np.ndarray],
    dims: List[int],
    eps_schedule: Optional[Sequence[float]] = None,
    eps0: float = 1.0,
    q: float = 10.0,
    n_outer: int = 3,
    method: str = "kl",
    kl_eta_rule: str = "eps_over_N",
    M_inner: int = 1,
    max_inner: int = 5000,
    inner_tol: float = 1e-3,
    final_tol: float = 1e-8,
    tol_mode: str = "fixed",
    C_tol: float = 1.0,
    p: float = 1.0,
    tol_F: Optional[float] = None,
    jitter: float = 1e-12,
    rho_reg: float = 0.0,
    md_tol_inner: float = 1e-4,
    project_pi: bool = True,
    init: str = "zero",
    U0: Optional[List[np.ndarray]] = None,
    return_history: bool = False,
) -> AnnealedSolverResult:
    """Run epsilon continuation using a selected inner EQOT solver.

    Parameters
    ----------
    method:
        ``"kl"`` uses :func:`potential_marginal_kl_descent`.
        ``"md_sinkhorn"`` uses :func:`md_type_sinkhorn_potential`.
    kl_eta_rule:
        Used only for ``method="kl"``.  The rule is evaluated at each stage's
        current epsilon, so ``eps_over_N`` means ``eta=eps_t/N`` stage by stage.
    M_inner:
        Used only for ``method="md_sinkhorn"``.
    """

    dims = [int(d) for d in dims]
    if len(target_marginals) != len(dims):
        raise ValueError("target_marginals length must match dims.")
    if max_inner < 0:
        raise ValueError("max_inner must be nonnegative.")
    if method not in {"kl", "md_sinkhorn"}:
        raise ValueError("method must be 'kl' or 'md_sinkhorn'.")

    schedule = validate_eps_schedule(eps_schedule) if eps_schedule is not None else annealing_eps_schedule(
        eps0=eps0,
        q=q,
        n_outer=n_outer,
    )
    H = hermitianize(np.asarray(H, dtype=complex))
    gammas = _regularize_targets(target_marginals, dims, rho_reg)
    U_list = _initial_potentials(init, dims, U0)

    F_list: List[float] = []
    e_tr_list: List[float] = []
    per_i_tr_list: List[np.ndarray] = []
    times: List[float] = []
    gibbs_calls_list: List[int] = []
    stage_history: List[Dict[str, Any]] = []
    stage_gibbs_calls_list: List[int] = []
    stage_iters_list: List[int] = []
    stage_eps_list: List[float] = []
    U_hist: Optional[List[List[np.ndarray]]] = [] if return_history else None
    pi_list: Optional[List[np.ndarray]] = [] if return_history else None
    total_gibbs = 0
    t0 = time.time()
    final_res: Any = None

    for stage_index, eps in enumerate(schedule):
        is_final = stage_index == len(schedule) - 1
        stage_tol = _stage_tol(
            eps=eps,
            is_final_stage=is_final,
            tol_mode=tol_mode,
            inner_tol=inner_tol,
            final_tol=final_tol,
            C_tol=C_tol,
            p=p,
        )
        if method == "kl":
            res = potential_marginal_kl_descent(
                H=H,
                gammas=gammas,
                eps=eps,
                dims=dims,
                T=max_inner,
                eta=None,
                eta_rule=kl_eta_rule,
                jitter_log=jitter,
                tol_tr=stage_tol,
                tol_F=tol_F if is_final else None,
                project_pi=project_pi,
                U0=U_list,
            )
        else:
            res = md_type_sinkhorn_potential(
                H=H,
                gammas=gammas,
                eps=eps,
                dims=dims,
                T_outer=max_inner,
                tol_tr=stage_tol,
                tol_F=tol_F if is_final else None,
                jitter=jitter,
                M_inner=M_inner,
                tol_inner=md_tol_inner,
                project_pi=project_pi,
                U0=U_list,
            )

        U_list = [hermitianize(Ui.copy()) for Ui in res.U_list]
        pi = res.pi
        per_i = marginal_trace_errors(pi, gammas, dims)
        Fv = float(F_marg(pi, gammas, dims, jitter=jitter))
        elapsed = float(time.time() - t0)
        stage_calls = int(getattr(res, "gibbs_calls", 0) or 0)
        total_gibbs += stage_calls
        inner_iters = max(0, len(getattr(res, "e_tr_list", []) or []) - 1)

        F_list.append(Fv)
        e_tr_list.append(float(np.max(per_i)))
        per_i_tr_list.append(per_i)
        times.append(elapsed)
        gibbs_calls_list.append(int(total_gibbs))
        stage_gibbs_calls_list.append(stage_calls)
        stage_iters_list.append(inner_iters)
        stage_eps_list.append(float(eps))
        if return_history and U_hist is not None and pi_list is not None:
            U_hist.append([Ui.copy() for Ui in U_list])
            pi_list.append(pi.copy())
        stage_history.append(
            {
                "stage": int(stage_index),
                "epsilon": float(eps),
                "lambda": float(1.0 / float(eps)),
                "stage_tol": float(stage_tol),
                "final_F_marg": Fv,
                "final_e_tr": float(np.max(per_i)),
                "final_trace_sum": float(np.sum(per_i)),
                "inner_iterations": int(inner_iters),
                "gibbs_calls": int(stage_calls),
                "total_gibbs_calls": int(total_gibbs),
                "solver_converged": bool(getattr(res, "converged", False)),
                "objective_value": float(np.real(np.trace(H @ pi))),
                "time_sec": elapsed,
            }
        )
        final_res = res

    if final_res is None:
        raise RuntimeError("annealed_eqot_solver produced no stages.")

    return AnnealedSolverResult(
        F_list=F_list,
        e_tr_list=e_tr_list,
        per_i_tr_list=per_i_tr_list,
        times=times,
        pi=final_res.pi,
        U_list=U_list,
        U_hist=U_hist,
        pi_list=pi_list,
        converged=bool(getattr(final_res, "converged", False)),
        gibbs_calls=int(total_gibbs),
        gibbs_calls_list=gibbs_calls_list,
        eps_schedule=schedule,
        eps_final=float(schedule[-1]),
        stage_history=stage_history,
        stage_gibbs_calls_list=stage_gibbs_calls_list,
        stage_iters_list=stage_iters_list,
        stage_eps_list=stage_eps_list,
        method=method,
        method_variant=_method_variant(method, kl_eta_rule, M_inner),
        metadata={
            "method": method,
            "kl_eta_rule": kl_eta_rule if method == "kl" else "",
            "M_inner": int(M_inner) if method == "md_sinkhorn" else "",
            "max_inner": int(max_inner),
            "inner_tol": float(inner_tol),
            "final_tol": float(final_tol),
            "tol_mode": tol_mode,
            "C_tol": float(C_tol),
            "p": float(p),
            "jitter": float(jitter),
            "rho_reg": float(rho_reg),
            "md_tol_inner": float(md_tol_inner),
            "project_pi": bool(project_pi),
            "init": init,
        },
    )


def annealed_kl_descent(**kwargs: Any) -> AnnealedSolverResult:
    """Convenience wrapper for KL-descent epsilon continuation."""

    kwargs["method"] = "kl"
    return annealed_eqot_solver(**kwargs)


def annealed_md_sinkhorn(**kwargs: Any) -> AnnealedSolverResult:
    """Convenience wrapper for MD-Sinkhorn epsilon continuation."""

    kwargs["method"] = "md_sinkhorn"
    return annealed_eqot_solver(**kwargs)


__all__ = [
    "AnnealedSolverResult",
    "annealing_eps_schedule",
    "geometric_eps_schedule",
    "validate_eps_schedule",
    "annealed_eqot_solver",
    "annealed_kl_descent",
    "annealed_md_sinkhorn",
]
