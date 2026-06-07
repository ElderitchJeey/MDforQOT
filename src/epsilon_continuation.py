"""Annealed KL descent for small-epsilon entropic QOT.

This module implements an epsilon-continuation solver for finite-dimensional
entropic quantum optimal transport. It uses the unscaled potential convention

    pi_eps(V) propto exp((sum_i Q_i(V_i) - H) / eps),

and decreases eps stage by stage. At each stage it runs KL/Sinkhorn-type
potential corrections and warm-starts the next, smaller epsilon.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from src.SolverofEQOT import gibbs_state_from_potentials, potential_marginal_kl_descent
from src.linalg import herm_log, hermitianize, proj_to_density, quantum_KL, trace_norm
from src.tensor import partial_trace_except_i


PartialTraceFn = Callable[[np.ndarray, List[int], int], np.ndarray]
LiftFn = Callable[[np.ndarray, List[int], int], np.ndarray]


@dataclass
class AnnealedKLDescentResult:
    """Solver-style output for epsilon-annealed KL descent."""

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
    eta: float = 0.0
    eta_rule: str = "annealed_manual"
    metadata: Optional[Dict[str, Any]] = None
    final_marginals: Optional[List[np.ndarray]] = None

    def __getitem__(self, key: str) -> Any:
        aliases = {
            "potentials": "U_list",
            "history": "stage_history",
            "total_gibbs_calls": "gibbs_calls",
        }
        return getattr(self, aliases.get(key, key))


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


def annealing_eps_schedule(*, eps0: float, q: float, n_outer: int) -> List[float]:
    """Return eps_t = eps0 / q**t for t=0,...,n_outer."""

    if eps0 <= 0:
        raise ValueError("eps0 must be positive.")
    if q <= 1:
        raise ValueError("q must be > 1.")
    if n_outer < 0:
        raise ValueError("n_outer must be nonnegative.")
    return [float(eps0) / (float(q) ** t) for t in range(int(n_outer) + 1)]


def validate_eps_schedule(eps_schedule: Sequence[float]) -> List[float]:
    """Validate and normalize a user-provided epsilon schedule."""

    out = [float(eps) for eps in eps_schedule]
    if not out:
        raise ValueError("eps_schedule cannot be empty.")
    if any(eps <= 0 for eps in out):
        raise ValueError("all epsilon values must be positive.")
    for prev, curr in zip(out, out[1:]):
        if curr > prev:
            raise ValueError("eps_schedule must be nonincreasing.")
    return out


def _regularize_density(rho: np.ndarray, delta: float) -> np.ndarray:
    rho = proj_to_density(rho, jitter=0.0)
    if delta <= 0:
        return rho
    d = rho.shape[0]
    out = (1.0 - float(delta)) * rho + float(delta) * np.eye(d, dtype=complex) / float(d)
    return proj_to_density(out, jitter=0.0)


def _gibbs_from_unscaled_potentials(
    *,
    H: np.ndarray,
    V_list: List[np.ndarray],
    dims: List[int],
    eps: float,
    lift_fn: Optional[LiftFn],
) -> np.ndarray:
    if lift_fn is not None:
        raise NotImplementedError("custom lift_fn is not supported by the solver-backed annealed implementation.")
    return gibbs_state_from_potentials(V_list, H, eps, dims, jitter=0.0, project=False)


def _marginals(pi: np.ndarray, dims: List[int], partial_trace_fn: Optional[PartialTraceFn]) -> List[np.ndarray]:
    fn = partial_trace_except_i if partial_trace_fn is None else partial_trace_fn
    return [hermitianize(fn(pi, dims, i)) for i in range(len(dims))]


def _residual_trace_sum(sigmas: List[np.ndarray], targets: List[np.ndarray]) -> float:
    return float(sum(trace_norm(sigma - rho) for sigma, rho in zip(sigmas, targets)))


def _residual_trace_max(sigmas: List[np.ndarray], targets: List[np.ndarray]) -> float:
    return float(max(trace_norm(sigma - rho) for sigma, rho in zip(sigmas, targets)))


def _target_first_marginal_kl(sigmas: List[np.ndarray], targets: List[np.ndarray], log_floor: float) -> float:
    return float(sum(quantum_KL(rho, sigma, jitter=log_floor) for sigma, rho in zip(sigmas, targets)))


def _entropy(pi: np.ndarray, log_floor: float) -> float:
    return float(np.real(np.trace(pi @ herm_log(pi, jitter=log_floor))))


def _stage_tol(
    *,
    eps: float,
    is_final_stage: bool,
    tol_mode: str,
    inner_tol: float,
    final_tol: float,
    final_stage_tol: Optional[float],
    C_tol: float,
    p: float,
) -> float:
    if is_final_stage:
        return float(final_tol if final_stage_tol is None else final_stage_tol)
    if tol_mode == "fixed":
        return float(inner_tol)
    if tol_mode == "annealed":
        return float(max(float(final_tol), float(C_tol) * (float(eps) ** float(p))))
    raise ValueError("tol_mode must be 'fixed' or 'annealed'.")


def annealed_quantum_kl_descent(
    H: np.ndarray,
    target_marginals: List[np.ndarray],
    dims: List[int],
    partial_trace_fn: Optional[PartialTraceFn] = None,
    lift_fn: Optional[LiftFn] = None,
    eps0: float = 1e-1,
    q: float = 2.0,
    n_outer: int = 10,
    eta: Optional[float] = None,
    inner_tol: float = 1e-8,
    max_inner: int = 5000,
    tol_mode: str = "fixed",
    final_tol: float = 1e-10,
    final_stage_tol: Optional[float] = None,
    C_tol: float = 1.0,
    p: float = 1.0,
    extrapolate: bool = True,
    rho_reg: float = 1e-12,
    log_floor: float = 1e-12,
    return_history: bool = True,
    residual: str = "trace_sum",
    init: str = "zero",
) -> AnnealedKLDescentResult:
    """Annealed KL descent for finite-dimensional entropic QOT.

    Parameters follow the convention in the user's notes. If ``eta`` is None,
    the conservative default eta=1/N is used. Pass eta=1.0 for a full
    KL/Sinkhorn correction step. Earlier epsilon stages use ``inner_tol`` or
    the annealed tolerance. The final epsilon stage uses ``final_tol`` by
    default, or ``final_stage_tol`` if provided.
    """

    dims = [int(d) for d in dims]
    N = len(dims)
    if len(target_marginals) != N:
        raise ValueError("target_marginals length must match dims length.")
    if max_inner < 0:
        raise ValueError("max_inner must be nonnegative.")
    if residual not in {"trace_sum", "trace_max", "F_marg"}:
        raise ValueError("residual must be 'trace_sum', 'trace_max', or 'F_marg'.")
    if init not in {"zero", "eps0_logrho"}:
        raise ValueError("init must be 'zero' or 'eps0_logrho'.")
    if partial_trace_fn is not None or lift_fn is not None:
        raise NotImplementedError(
            "annealed_quantum_kl_descent currently reuses src.SolverofEQOT solvers, "
            "which use the repo-native partial trace and lift operators."
        )

    H = hermitianize(np.asarray(H, dtype=complex))
    targets = [_regularize_density(np.asarray(rho, dtype=complex), rho_reg) for rho in target_marginals]
    eta_eff = (1.0 / float(N)) if eta is None else float(eta)
    eps_schedule = annealing_eps_schedule(eps0=eps0, q=q, n_outer=n_outer)

    if init == "zero":
        V_list = [np.zeros((dims[i], dims[i]), dtype=complex) for i in range(N)]
    else:
        V_list = [float(eps0) * herm_log(targets[i], jitter=log_floor) for i in range(N)]
        V_list = [hermitianize(Vi) for Vi in V_list]

    history: List[Dict[str, Any]] = []
    F_list: List[float] = []
    e_tr_list: List[float] = []
    per_i_tr_list: List[np.ndarray] = []
    times: List[float] = []
    gibbs_calls_list: List[int] = []
    U_hist: Optional[List[List[np.ndarray]]] = [] if return_history else None
    pi_list: Optional[List[np.ndarray]] = [] if return_history else None
    stage_gibbs_calls_list: List[int] = []
    stage_iters_list: List[int] = []
    stage_eps_list: List[float] = []
    total_gibbs_calls = 0
    prev_lambda: Optional[float] = None
    prev_U_exp: Optional[List[np.ndarray]] = None
    curr_U_exp: Optional[List[np.ndarray]] = None
    t0 = time.time()

    final_pi: Optional[np.ndarray] = None
    final_sigmas: Optional[List[np.ndarray]] = None

    for stage_index, eps in enumerate(eps_schedule):
        stage_start_elapsed = time.time() - t0
        stage_global_gibbs_start = int(total_gibbs_calls)
        lam = 1.0 / float(eps)
        stage_tol = _stage_tol(
            eps=eps,
            is_final_stage=(stage_index == len(eps_schedule) - 1),
            tol_mode=tol_mode,
            inner_tol=inner_tol,
            final_tol=final_tol,
            final_stage_tol=final_stage_tol,
            C_tol=C_tol,
            p=p,
        )

        pi = _gibbs_from_unscaled_potentials(H=H, V_list=V_list, dims=dims, eps=eps, lift_fn=lift_fn)
        sigmas = _marginals(pi, dims, partial_trace_fn)
        trace_sum = _residual_trace_sum(sigmas, targets)
        trace_max = _residual_trace_max(sigmas, targets)
        F_target = _target_first_marginal_kl(sigmas, targets, log_floor)

        if residual == "trace_sum":
            res_value = trace_sum
        elif residual == "trace_max":
            res_value = trace_max
        else:
            res_value = F_target
        initial_residual = float(res_value)

        solver_tol_tr: Optional[float]
        if residual == "trace_sum":
            solver_tol_tr = float(stage_tol) / float(N)
        elif residual == "trace_max":
            solver_tol_tr = float(stage_tol)
        else:
            solver_tol_tr = None

        solver_res = potential_marginal_kl_descent(
            H=H,
            gammas=targets,
            eps=eps,
            dims=dims,
            T=int(max_inner),
            eta=eta_eff * float(eps),
            jitter_log=log_floor,
            tol_tr=solver_tol_tr,
            tol_F=None,
            store_hist=False,
            project_pi=False,
            U0=V_list,
        )
        pi = solver_res.pi
        V_list = [hermitianize(Ui) for Ui in solver_res.U_list]
        sigmas = _marginals(pi, dims, partial_trace_fn)
        trace_sum = _residual_trace_sum(sigmas, targets)
        trace_max = _residual_trace_max(sigmas, targets)
        F_target = _target_first_marginal_kl(sigmas, targets, log_floor)
        if residual == "trace_sum":
            res_value = trace_sum
        elif residual == "trace_max":
            res_value = trace_max
        else:
            res_value = F_target

        inner_iters = max(0, len(solver_res.e_tr_list) - 1)
        # One Gibbs call above records the warm-start residual before the
        # solver-backed stage; solver_res.gibbs_calls starts from its own
        # initial Gibbs evaluation.
        stage_gibbs = 1 + int(solver_res.gibbs_calls)
        total_gibbs_calls += stage_gibbs
        stage_gibbs_calls_list.append(int(stage_gibbs))
        stage_iters_list.append(int(inner_iters))
        stage_eps_list.append(float(eps))
        converged = bool(res_value <= stage_tol)

        local_gibbs = list(solver_res.gibbs_calls_list or [])
        for idx, Fv in enumerate(list(solver_res.F_list or [])):
            F_list.append(float(Fv))
            if idx < len(solver_res.e_tr_list):
                e_tr_list.append(float(solver_res.e_tr_list[idx]))
            if idx < len(solver_res.per_i_tr_list):
                per_i_tr_list.append(np.asarray(solver_res.per_i_tr_list[idx], dtype=float))
            if idx < len(solver_res.times):
                times.append(float(stage_start_elapsed + solver_res.times[idx]))
            else:
                times.append(float(time.time() - t0))
            if idx < len(local_gibbs):
                # One extra Gibbs call records the pre-stage warm-start residual.
                gibbs_calls_list.append(int(stage_global_gibbs_start + 1 + local_gibbs[idx]))
            else:
                gibbs_calls_list.append(int(total_gibbs_calls))

        if return_history and U_hist is not None and pi_list is not None:
            U_hist.append([Ui.copy() for Ui in V_list])
            pi_list.append(pi.copy())

        stage_inner_history: List[Dict[str, Any]] = []
        if return_history:
            e_list = list(solver_res.e_tr_list or [])
            f_list = list(solver_res.F_list or [])
            gibbs_list = list(solver_res.gibbs_calls_list or [])
            n_hist = max(len(e_list), len(f_list), len(gibbs_list))
            for inner_iter in range(n_hist):
                stage_inner_history.append(
                    {
                        "inner_iter": int(inner_iter),
                        "gibbs_calls": int(gibbs_list[inner_iter]) if inner_iter < len(gibbs_list) else "",
                        "trace_max": float(e_list[inner_iter]) if inner_iter < len(e_list) else "",
                        "F_marg_sigma_first": float(f_list[inner_iter]) if inner_iter < len(f_list) else "",
                    }
                )

        cost = float(np.real(np.trace(H @ pi)))
        entropy = _entropy(pi, log_floor)
        entropic_objective = cost + float(eps) * entropy
        stage_record: Dict[str, Any] = {
            "stage": int(stage_index),
            "epsilon": float(eps),
            "lambda": float(lam),
            "stage_tol": float(stage_tol),
            "initial_residual": float(initial_residual),
            "final_residual": float(res_value),
            "trace_sum": float(trace_sum),
            "trace_max": float(trace_max),
            "F_marg_target_first": float(F_target),
            "inner_iterations": int(inner_iters),
            "gibbs_calls": int(stage_gibbs),
            "gibbs_calls_since_previous_q_update": int(stage_gibbs),
            "gibbs_calls_before_stage": int(stage_global_gibbs_start),
            "gibbs_calls_after_stage": int(total_gibbs_calls),
            "total_gibbs_calls": int(total_gibbs_calls),
            "solver_converged": bool(solver_res.converged),
            "objective_value": float(cost),
            "entropy": float(entropy),
            "entropic_objective": float(entropic_objective),
            "converged": bool(converged),
            "time_sec": float(time.time() - t0),
        }
        if return_history:
            stage_record["inner_history"] = stage_inner_history
        history.append(stage_record)

        final_pi = pi
        final_sigmas = sigmas
        curr_U_exp = [hermitianize(Vi / float(eps)) for Vi in V_list]

        if stage_index + 1 < len(eps_schedule):
            eps_next = float(eps_schedule[stage_index + 1])
            if extrapolate and prev_U_exp is not None and prev_lambda is not None:
                lam_next = 1.0 / eps_next
                denom = float(lam - prev_lambda)
                if abs(denom) > 0:
                    ratio = float((lam_next - lam) / denom)
                    next_U_exp = [
                        hermitianize(U_curr + ratio * (U_curr - U_prev))
                        for U_curr, U_prev in zip(curr_U_exp, prev_U_exp)
                    ]
                    V_list = [hermitianize(eps_next * Ui) for Ui in next_U_exp]
                else:
                    V_list = [Vi.copy() for Vi in V_list]
            else:
                V_list = [Vi.copy() for Vi in V_list]

        prev_lambda = lam
        prev_U_exp = [Ui.copy() for Ui in curr_U_exp]

    if final_pi is None or final_sigmas is None:
        raise RuntimeError("annealed solver produced no stages.")

    return AnnealedKLDescentResult(
        F_list=F_list,
        e_tr_list=e_tr_list,
        per_i_tr_list=per_i_tr_list,
        times=times,
        pi=final_pi,
        U_list=[hermitianize(Vi) for Vi in V_list],
        U_hist=U_hist,
        pi_list=pi_list,
        converged=bool(history[-1]["converged"]),
        gibbs_calls=int(total_gibbs_calls),
        gibbs_calls_list=gibbs_calls_list,
        eps_schedule=eps_schedule,
        eps_final=float(eps_schedule[-1]),
        stage_history=history,
        stage_gibbs_calls_list=stage_gibbs_calls_list,
        stage_iters_list=stage_iters_list,
        stage_eps_list=stage_eps_list,
        eta=float(eta_eff),
        metadata={
            "q": float(q),
            "n_outer": int(n_outer),
            "tol_mode": tol_mode,
            "inner_tol": float(inner_tol),
            "final_tol": float(final_tol),
            "final_stage_tol": None if final_stage_tol is None else float(final_stage_tol),
            "C_tol": float(C_tol),
            "p": float(p),
            "extrapolate": bool(extrapolate),
            "rho_reg": float(rho_reg),
            "log_floor": float(log_floor),
            "residual": residual,
            "init": init,
        },
        final_marginals=final_sigmas,
    )


def epsilon_continuation_eqot(**kwargs: Any) -> AnnealedKLDescentResult:
    """Backward-compatible alias for the annealed KL implementation."""

    return annealed_quantum_kl_descent(**kwargs)


__all__ = [
    "PartialTraceFn",
    "LiftFn",
    "geometric_eps_schedule",
    "annealing_eps_schedule",
    "validate_eps_schedule",
    "annealed_quantum_kl_descent",
    "epsilon_continuation_eqot",
]
