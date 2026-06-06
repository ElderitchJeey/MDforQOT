"""Paper79-style metrics for repo-native solver results."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np

from src.SolverofEQOT import F_marg, marginal_trace_errors


def primal_cost(H: np.ndarray, pi: np.ndarray) -> float:
    """Return Tr[H pi] for a candidate coupling."""

    return float(np.real(np.trace(np.asarray(H) @ np.asarray(pi))))


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
