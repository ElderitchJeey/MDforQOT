# src/metrics.py  (PHASE 1)
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, Sequence, runtime_checkable

import numpy as np

# Canonical metric definitions live in SolverofEQOT (phase 1: forward them)
from src.SolverofEQOT import F_marg as objective_F_marg
from src.SolverofEQOT import marginal_trace_errors
from src.linalg import trace_norm

@runtime_checkable
class ResultLike(Protocol):
    F_list: Sequence[float]
    e_tr_list: Sequence[float]
    times: Sequence[float]
    converged: bool
    gibbs_calls: int
    # NEW (optional, for precise curves)
    gibbs_calls_list: Optional[Sequence[int]]


def iters(res: ResultLike) -> int:
    n = len(res.e_tr_list) if res.e_tr_list is not None else 0
    return int(max(0, n - 1))


def total_time(res: ResultLike) -> float:
    if res.times is None or len(res.times) == 0:
        return 0.0
    return float(res.times[-1])


def time_per_gibbs_call(res: ResultLike) -> float:
    calls = int(getattr(res, "gibbs_calls", 0) or 0)
    return total_time(res) / float(max(1, calls))


def final_scalar_metrics(res: ResultLike) -> Dict[str, Any]:
    final_e = float(res.e_tr_list[-1]) if res.e_tr_list else float("inf")
    final_F = float(res.F_list[-1]) if res.F_list else float("inf")
    calls = int(getattr(res, "gibbs_calls", 0) or 0)

    return {
        "converged": bool(res.converged),
        "iters": iters(res),
        "final_e_tr": final_e,
        "final_F": final_F,
        "time_sec": total_time(res),
        "gibbs_calls": calls,
        "time_per_gibbs": total_time(res) / float(max(1, calls)),
    }


def trajectory_arrays(res: ResultLike) -> Dict[str, np.ndarray]:
    return {
        "e_tr": np.asarray(res.e_tr_list, dtype=float) if res.e_tr_list is not None else np.zeros(0, dtype=float),
        "F": np.asarray(res.F_list, dtype=float) if res.F_list is not None else np.zeros(0, dtype=float),
        "t": np.asarray(res.times, dtype=float) if res.times is not None else np.zeros(0, dtype=float),
    }


def error_vs_gibbs(res: ResultLike) -> Dict[str, np.ndarray]:
    """
    Returns arrays suitable for plotting error vs gibbs_calls.
    Requires res.gibbs_calls_list to be present for exact x-axis.
    If absent, falls back to a linear proxy using total gibbs_calls.
    """
    e = np.asarray(res.e_tr_list, dtype=float) if res.e_tr_list is not None else np.zeros(0, dtype=float)

    g_list = getattr(res, "gibbs_calls_list", None)
    if g_list is not None and len(g_list) == len(e):
        g = np.asarray(g_list, dtype=int)
        return {"gibbs": g, "e_tr": e}

    # fallback: proportional proxy (not exact)
    total = int(getattr(res, "gibbs_calls", 0) or 0)
    if len(e) <= 1:
        g = np.asarray([total], dtype=int)
    else:
        g = np.round(np.linspace(0, total, num=len(e))).astype(int)
    return {"gibbs": g, "e_tr": e}

# ============================================================
# Same-limit comparison utilities (compare by final coupling pi)
# ============================================================

def final_pi(res: Any) -> np.ndarray:
    """Return the final coupling pi for a solver result.

    Preference order:
      1) res.pi (most solvers store the final coupling here)
      2) res.pi_list[-1] (if a trajectory is stored)

    Raises:
      AttributeError if neither is available.
    """
    if hasattr(res, "pi") and res.pi is not None:
        return np.asarray(res.pi)
    pi_list = getattr(res, "pi_list", None)
    if pi_list is not None and len(pi_list) > 0:
        return np.asarray(pi_list[-1])
    raise AttributeError("Result has no final pi: expected attribute 'pi' or non-empty 'pi_list'.")


def pi_distance(
    pi_a: np.ndarray,
    pi_b: np.ndarray,
    *,
    metric: str = "trace",
) -> float:
    """Distance between two couplings (density operators).

    Supported metrics:
      - 'trace' : ||pi_a - pi_b||_1 (trace norm)
      - 'fro'   : ||pi_a - pi_b||_F (Frobenius)
      - 'op'    : ||pi_a - pi_b||_2 (spectral/operator norm)
    """
    A = np.asarray(pi_a)
    B = np.asarray(pi_b)
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
    raise ValueError(f"Unknown metric='{metric}'. Use 'trace', 'fro', or 'op'.")


def same_limit(
    res_a: Any,
    res_b: Any,
    *,
    tol_pi: float = 1e-6,
    metric: str = "trace",
    require_converged: bool = True,
    tol_e_tr: Optional[float] = None,
) -> Dict[str, Any]:
    """Compare whether two runs converge to the same coupling (final pi).

    This is the *right* notion of "same limit" here because potentials are not
    unique up to gauge transforms, while the coupling pi is the physical object.

    Args:
      res_a, res_b:
        Solver results (e.g., DBGAResult / MDsinkhornResult / PotentialKLDescentResult).
        Must contain a final coupling as 'pi' (or a non-empty 'pi_list').
      tol_pi:
        Threshold for declaring the final couplings equal (under the chosen metric).
      metric:
        'trace' (recommended), 'fro', or 'op'.
      require_converged:
        If True, returns same=False unless both runs are considered converged.
      tol_e_tr:
        Optional override: treat a run as converged iff its final e_tr <= tol_e_tr
        (useful if some experiment didn't set res.converged reliably).

    Returns:
      dict with fields:
        - same_limit: bool
        - dist_pi: float
        - converged_a / converged_b: bool
        - final_e_tr_a / final_e_tr_b: float
        - metric, tol_pi
    """

    def _final_e_tr(res: Any) -> float:
        e = getattr(res, "e_tr_list", None)
        if e is None or len(e) == 0:
            return float("inf")
        return float(e[-1])

    ea = _final_e_tr(res_a)
    eb = _final_e_tr(res_b)

    if tol_e_tr is not None:
        conv_a = bool(ea <= float(tol_e_tr))
        conv_b = bool(eb <= float(tol_e_tr))
    else:
        conv_a = bool(getattr(res_a, "converged", False))
        conv_b = bool(getattr(res_b, "converged", False))

    pi_a = final_pi(res_a)
    pi_b = final_pi(res_b)
    dist = float(pi_distance(pi_a, pi_b, metric=metric))

    same = dist <= float(tol_pi)
    if require_converged:
        same = bool(same and conv_a and conv_b)

    return {
        "same_limit": same,
        "dist_pi": dist,
        "metric": metric,
        "tol_pi": float(tol_pi),
        "require_converged": bool(require_converged),
        "converged_a": conv_a,
        "converged_b": conv_b,
        "final_e_tr_a": ea,
        "final_e_tr_b": eb,
    }


def same_limit_matrix(
    results: Sequence[Any],
    *,
    tol_pi: float = 1e-6,
    metric: str = "trace",
    require_converged: bool = True,
    tol_e_tr: Optional[float] = None,
) -> np.ndarray:
    """Pairwise same-limit boolean matrix for a list of runs."""
    n = len(results)
    M = np.zeros((n, n), dtype=bool)
    for i in range(n):
        M[i, i] = True
        for j in range(i + 1, n):
            out = same_limit(
                results[i],
                results[j],
                tol_pi=tol_pi,
                metric=metric,
                require_converged=require_converged,
                tol_e_tr=tol_e_tr,
            )
            M[i, j] = M[j, i] = bool(out["same_limit"])
    return M
