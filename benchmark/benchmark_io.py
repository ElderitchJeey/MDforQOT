# benchmark/diagnostics.py

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from src.linalg import hermitianize, herm_log, quantum_KL, trace_norm, proj_to_density
from src.tensor import partial_trace_except_i


def marginal_kl_error(
    pi: np.ndarray,
    gammas: Sequence[np.ndarray],
    dims: Sequence[int],
    jitter: float = 1e-12,
) -> float:
    """
    F_marg(pi) = sum_i KL(Tr_{neq i} pi || gamma_i).

    This is a benchmark/QOT diagnostic, not an SDPLab core function.
    """
    pi = hermitianize(np.asarray(pi, dtype=np.complex128))
    dims = [int(x) for x in dims]

    if len(gammas) != len(dims):
        raise ValueError(f"len(gammas)={len(gammas)} but len(dims)={len(dims)}.")

    total = 0.0
    for i in range(len(dims)):
        rho_i = partial_trace_except_i(pi, dims, i)
        rho_i = proj_to_density(rho_i, jitter=jitter)
        gamma_i = proj_to_density(np.asarray(gammas[i], dtype=np.complex128), jitter=jitter)
        total += float(quantum_KL(rho_i, gamma_i, jitter=jitter))

    return float(total)


def marginal_trace_errors(
    pi: np.ndarray,
    gammas: Sequence[np.ndarray],
    dims: Sequence[int],
) -> np.ndarray:
    """
    Per-marginal trace-norm errors:
        ||Tr_{neq i} pi - gamma_i||_1.
    """
    pi = hermitianize(np.asarray(pi, dtype=np.complex128))
    dims = [int(x) for x in dims]

    if len(gammas) != len(dims):
        raise ValueError(f"len(gammas)={len(gammas)} but len(dims)={len(dims)}.")

    errs = []
    for i in range(len(dims)):
        rho_i = partial_trace_except_i(pi, dims, i)
        gamma_i = np.asarray(gammas[i], dtype=np.complex128)
        errs.append(float(trace_norm(rho_i - gamma_i)))

    return np.asarray(errs, dtype=float)


def primal_objective(
    pi: np.ndarray,
    H: np.ndarray,
    eps: float,
    jitter: float = 1e-12,
) -> float:
    """
    Entropic primal objective:
        P_eps(pi) = Tr(H pi) + eps Tr(pi log pi).
    """
    pi = proj_to_density(np.asarray(pi, dtype=np.complex128), jitter=jitter)
    H = hermitianize(np.asarray(H, dtype=np.complex128))
    log_pi = herm_log(pi, jitter=jitter)

    val = np.trace(H @ pi).real + float(eps) * np.trace(pi @ log_pi).real
    return float(val)


def density_diagnostics(
    pi: np.ndarray,
    jitter: float = 1e-12,
) -> Dict:
    """
    Basic density-matrix sanity checks.
    """
    pi = hermitianize(np.asarray(pi, dtype=np.complex128))
    evals = np.linalg.eigvalsh(pi)

    return {
        "trace_pi": float(np.trace(pi).real),
        "min_eig_pi": float(np.min(evals)),
        "max_eig_pi": float(np.max(evals)),
        "is_trace_one_tol1e-8": bool(abs(np.trace(pi).real - 1.0) <= 1e-8),
        "is_psd_tol1e-10": bool(np.min(evals) >= -1e-10),
    }


def final_diagnostics(
    pi: np.ndarray,
    H: np.ndarray,
    gammas: Sequence[np.ndarray],
    eps: float,
    dims: Sequence[int],
    tol_F: float = 1e-8,
    jitter: float = 1e-12,
) -> Dict:
    """
    Unified final diagnostics for every solver, including SDPLab.
    """
    F = marginal_kl_error(pi, gammas, dims, jitter=jitter)
    per_i = marginal_trace_errors(pi, gammas, dims)
    pobj = primal_objective(pi, H, eps, jitter=jitter)

    out = {
        "final_F_marg": float(F),
        "final_e_tr_max": float(np.max(per_i)) if len(per_i) else 0.0,
        "final_e_tr_sum": float(np.sum(per_i)) if len(per_i) else 0.0,
        "per_i_trace_errors": per_i.tolist(),
        "final_primal_obj": float(pobj),
        "hit": bool(F <= float(tol_F)),
    }
    out.update(density_diagnostics(pi, jitter=jitter))
    return out


def pi_distance(
    pi_a: np.ndarray,
    pi_b: np.ndarray,
    metric: str = "trace",
) -> float:
    """
    Distance between two final couplings.

    metric:
        trace: trace norm
        fro: Frobenius norm
        op: operator norm
    """
    A = hermitianize(np.asarray(pi_a, dtype=np.complex128))
    B = hermitianize(np.asarray(pi_b, dtype=np.complex128))

    if A.shape != B.shape:
        raise ValueError(f"pi shapes differ: {A.shape} vs {B.shape}")

    D = A - B
    m = metric.lower().strip()

    if m in {"trace", "tr", "1", "nuclear"}:
        return float(trace_norm(D))
    if m in {"fro", "frob", "f"}:
        return float(np.linalg.norm(D, ord="fro"))
    if m in {"op", "spec", "2"}:
        return float(np.linalg.norm(D, ord=2))

    raise ValueError(f"Unknown metric={metric}. Use trace, fro, or op.")