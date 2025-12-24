# src/metrics.py  (PHASE 1)
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, Sequence, runtime_checkable

import numpy as np

# Canonical metric definitions live in SolverofEQOT (phase 1: forward them)
from src.SolverofEQOT import F_marg as objective_F_marg
from src.SolverofEQOT import marginal_trace_errors


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
