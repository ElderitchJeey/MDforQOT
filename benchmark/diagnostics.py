# benchmark/diagnostics.py

from __future__ import annotations

import numpy as np
from typing import Dict, List

from src.linalg import herm_log
from src.SolverofEQOT import F_marg, marginal_trace_errors


def primal_objective(
    pi: np.ndarray,
    H: np.ndarray,
    eps: float,
    jitter: float = 1e-12,
) -> float:
    """
    P_eps(pi) = Tr(H pi) + eps Tr(pi log pi).
    """
    log_pi = herm_log(pi, jitter=jitter)
    val = np.trace(H @ pi).real + eps * np.trace(pi @ log_pi).real
    return float(val)


def final_diagnostics(
    pi: np.ndarray,
    H: np.ndarray,
    gammas: List[np.ndarray],
    eps: float,
    dims: List[int],
    tol_F: float = 1e-8,
    jitter: float = 1e-12,
) -> Dict:
    F = float(F_marg(pi, gammas, dims, jitter=jitter))
    per_i = marginal_trace_errors(pi, gammas, dims)

    return {
        "final_F_marg": F,
        "final_e_tr_max": float(np.max(per_i)),
        "final_e_tr_sum": float(np.sum(per_i)),
        "per_i_trace_errors": per_i.tolist(),
        "final_primal_obj": primal_objective(pi, H, eps, jitter=jitter),
        "hit": bool(F <= tol_F),
    }