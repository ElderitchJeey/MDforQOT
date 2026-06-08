"""Paper79-style metrics for repo-native solver results."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
from scipy.special import logsumexp

from src.SolverofEQOT import F_marg, marginal_trace_errors
from src.tensor import L_of_U


def primal_cost(H: np.ndarray, pi: np.ndarray) -> float:
    """Return Tr[H pi] for a candidate coupling."""

    return float(np.real(np.trace(np.asarray(H) @ np.asarray(pi))))


def entropic_dual_value(
    *,
    H: np.ndarray,
    U_list: Sequence[np.ndarray],
    gammas,
    dims,
    eps: float,
) -> float:
    """Entropy-regularized QOT dual value for local potentials.

    Uses the convention

        D(U) = sum_i Tr[U_i gamma_i]
             - eps log Tr exp((L(U) - H) / eps).

    This is the same concave dual objective optimized by the entropy L-BFGS
    baseline, evaluated with NumPy dense linear algebra.
    """

    if eps <= 0:
        raise ValueError("eps must be positive.")
    if len(U_list) != len(dims):
        raise ValueError("U_list length must match dims.")
    linear = 0.0
    for Ui, gamma in zip(U_list, gammas):
        linear += float(np.real(np.trace(np.asarray(Ui, dtype=complex) @ np.asarray(gamma, dtype=complex))))
    A = (L_of_U([np.asarray(Ui, dtype=complex) for Ui in U_list], list(dims)) - np.asarray(H, dtype=complex)) / float(eps)
    evals = np.linalg.eigvalsh(0.5 * (A + A.conj().T))
    return float(linear - float(eps) * float(logsumexp(evals)))


def final_pi(res: Any) -> np.ndarray:
    if hasattr(res, "pi") and res.pi is not None:
        return np.asarray(res.pi)
    pi_list = getattr(res, "pi_list", None)
    if pi_list:
        return np.asarray(pi_list[-1])
    raise AttributeError("solver result has no final pi")


def first_hit(values: Sequence[float], tol: float) -> int:
    arr = np.asarray(values, dtype=float)
    hit = np.where(arr <= float(tol))[0]
    return int(hit[0]) if hit.size else -1


def summarize_solver_result(
    *,
    label: str,
    res: Any,
    H: np.ndarray,
    gammas,
    dims,
    tol_f: Optional[float] = None,
    tol_tr: Optional[float] = None,
    ground_truth: Optional[float] = None,
    eps: Optional[float] = None,
) -> Dict[str, Any]:
    """Create a row compatible with paper79-style benchmark tables."""

    pi = final_pi(res)
    cost = primal_cost(H, pi)
    f_list = list(getattr(res, "F_list", []) or [])
    e_list = list(getattr(res, "e_tr_list", []) or [])
    times = list(getattr(res, "times", []) or [])
    gibbs_list = list(getattr(res, "gibbs_calls_list", []) or [])

    final_f = float(f_list[-1]) if f_list else float(F_marg(pi, gammas, dims))
    per_i = marginal_trace_errors(pi, gammas, dims)
    final_e = float(e_list[-1]) if e_list else float(np.max(per_i))
    elapsed = float(times[-1]) if times else 0.0
    gibbs_calls = int(getattr(res, "gibbs_calls", 0) or 0)
    U_list = getattr(res, "U_list", None)
    final_dual_value = ""
    if eps is not None and U_list is not None:
        try:
            final_dual_value = entropic_dual_value(H=H, U_list=U_list, gammas=gammas, dims=dims, eps=float(eps))
        except Exception as exc:
            final_dual_value = f"error: {type(exc).__name__}: {exc}"

    hit_f = first_hit(f_list, tol_f) if tol_f is not None and f_list else -1
    hit_tr = first_hit(e_list, tol_tr) if tol_tr is not None and e_list else -1

    n_iters = int(getattr(res, "n_iters", max(0, len(e_list) - 1)) or 0)

    row: Dict[str, Any] = {
        "method": label,
        "converged": bool(getattr(res, "converged", False)),
        "iters": n_iters,
        "time_sec": elapsed,
        "gibbs_calls": gibbs_calls,
        "final_cost": cost,
        "final_dual_value": final_dual_value,
        "final_F_marg": final_f,
        "final_e_tr": final_e,
        "hit_F_iter": hit_f,
        "hit_tr_iter": hit_tr,
        "hit_F_gibbs": int(gibbs_list[hit_f]) if hit_f >= 0 and len(gibbs_list) == len(f_list) else -1,
        "hit_tr_gibbs": int(gibbs_list[hit_tr]) if hit_tr >= 0 and len(gibbs_list) == len(e_list) else -1,
    }

    if hasattr(res, "eta"):
        row["eta"] = float(getattr(res, "eta"))
    if hasattr(res, "eta_rule"):
        row["eta_rule"] = str(getattr(res, "eta_rule"))

    if ground_truth is not None:
        row["ground_truth"] = float(ground_truth)
        row["objective_gap"] = abs(cost - float(ground_truth))

    return row
