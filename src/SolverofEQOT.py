import time
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Any

from src.linalg import (
    hermitianize,
    proj_to_density,
    herm_expm_scaled,
    herm_log,
    quantum_KL,
    trace_norm,
)
from src.tensor import Q_i_lift, partial_trace_except_i, L_of_U


# ============================================================
# Small utilities
# ============================================================

class _PiCounter:
    """Counts how many times we (re)compute the Gibbs coupling."""
    def __init__(self):
        self.n_calls = 0

    def reset(self):
        self.n_calls = 0

    def inc(self):
        self.n_calls += 1


PI_COUNTER = _PiCounter()


def gibbs_state_from_potentials(
    U_list: List[np.ndarray],
    H: np.ndarray,
    eps: float,
    dims: List[int],
    jitter: float = 1e-12,
    project: bool = True,
) -> np.ndarray:
    """
    pi(U) ∝ exp( (L_of_U(U_list) - H) / eps ), normalized to a density operator.

    BUG FIX: the exponent must be (L(U)-H)/eps, not L(U) - H/eps.
    """
    if eps <= 0:
        raise ValueError("eps must be positive.")

    PI_COUNTER.inc()

    LU = hermitianize(L_of_U(U_list, dims))
    X = hermitianize((LU - H) / eps)   # <-- BUG FIX

    E = herm_expm_scaled(X)            # scaled exp is fine; normalization cancels global factor
    Z = float(np.real(np.trace(E)))
    pi = hermitianize(E / max(Z, 1e-300))

    # Optional conservative projection (can mask bugs, but prevents numerical non-PSD)
    if project:
        pi = proj_to_density(pi, jitter=jitter)
    return pi


def F_marg(pi: np.ndarray, gammas: List[np.ndarray], dims: List[int], jitter: float = 1e-12) -> float:
    """sum_i KL( Tr_{≠i} pi || gamma_i )."""
    N = len(dims)
    if len(gammas) != N:
        raise ValueError("F_marg: len(gammas) must equal len(dims).")

    val = 0.0
    for i in range(N):
        rho_i = partial_trace_except_i(pi, dims, i)
        rho_i = proj_to_density(rho_i, jitter=jitter)
        gamma_i = proj_to_density(gammas[i], jitter=jitter)
        val += float(quantum_KL(rho_i, gamma_i, jitter=jitter))
    return float(val)


def marginal_trace_errors(pi: np.ndarray, gammas: List[np.ndarray], dims: List[int]) -> np.ndarray:
    N = len(dims)
    errs = np.zeros(N, dtype=float)
    for i in range(N):
        rho_i = partial_trace_except_i(pi, dims, i)
        errs[i] = float(trace_norm(rho_i - gammas[i]))
    return errs


def _stop_criterion(pi: np.ndarray, gammas: List[np.ndarray], dims: List[int], tol_tr: float) -> Tuple[bool, float]:
    per_i = marginal_trace_errors(pi, gammas, dims)
    e_tr = float(np.max(per_i)) if len(per_i) else 0.0
    return (e_tr <= tol_tr), e_tr


def _gauge_fix_trace0_all(U_list: List[np.ndarray], dims: List[int]) -> List[np.ndarray]:
    """
    Gauge fixing: enforce Tr(U_i)=0 for each i by subtracting (Tr(U_i)/d_i) I.
    This only shifts L(U) by a multiple of identity on full space (sum_i alpha_i I),
    which cancels after Gibbs normalization.
    """
    out = []
    for i, Ui in enumerate(U_list):
        di = int(dims[i])
        alpha = float(np.real(np.trace(Ui))) / di
        out.append(hermitianize(Ui - alpha * np.eye(di, dtype=complex)))
    return out


# ============================================================
# Result dataclasses (BUG FIX: gibbs_calls must have default or be earlier)
# ============================================================

@dataclass
class DBGAResult:
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


@dataclass
class PotentialKLDescentResult:
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


@dataclass
class MDsinkhornResult:
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


# ============================================================
# Algorithm (3): MD-type Sinkhorn in potential form
# ============================================================

def md_type_sinkhorn_potential(
    H: np.ndarray,
    gammas: List[np.ndarray],
    eps: float,
    dims: List[int],
    T_outer: int,
    tol_tr: float,
    jitter: float = 1e-12,
    eta_inner: float = 1.0,
    M_inner: int = 1,
    name: str = "MD-Sinkhorn",
    keep_U_hist: bool = False,
    keep_pi_hist: bool = False,
    tol_inner: Optional[float] = None,
    project_pi: bool = True,
) -> MDsinkhornResult:
    if abs(eta_inner - 1.0) > 1e-12:
        raise ValueError("This file enforces eta_inner=1.")
    if M_inner < 1:
        raise ValueError("M_inner must be >= 1.")
    if tol_inner is None:
        tol_inner = tol_tr

    PI_COUNTER.reset()
    N = len(dims)
    if len(gammas) != N:
        raise ValueError(f"len(gammas)={len(gammas)} must equal len(dims)={N}")

    U_list = [np.zeros((dims[i], dims[i]), dtype=complex) for i in range(N)]
    pi = gibbs_state_from_potentials(U_list, H, eps, dims, jitter=jitter, project=project_pi)

    F_list: List[float] = []
    e_tr_list: List[float] = []
    per_i_tr_list: List[np.ndarray] = []
    times: List[float] = []
    U_hist = [] if keep_U_hist else None
    pi_list = [] if keep_pi_hist else None

    t0 = time.time()

    def _record(pi_curr: np.ndarray, U_curr: List[np.ndarray]) -> bool:
        converged, e_tr = _stop_criterion(pi_curr, gammas, dims, tol_tr)
        per_i = marginal_trace_errors(pi_curr, gammas, dims)
        Fv = float(F_marg(pi_curr, gammas, dims, jitter=jitter))

        F_list.append(Fv)
        e_tr_list.append(float(e_tr))
        per_i_tr_list.append(per_i)
        times.append(time.time() - t0)

        if keep_U_hist:
            U_hist.append([Ui.copy() for Ui in U_curr])
        if keep_pi_hist:
            pi_list.append(pi_curr.copy())
        return converged

    converged = _record(pi, U_list)

    for _k in range(T_outer):
        if converged:
            break

        for i in range(N):
            for _ in range(M_inner):
                rho_i = partial_trace_except_i(pi, dims, i)
                err_i = float(trace_norm(rho_i - gammas[i]))
                if err_i <= float(tol_inner):
                    break
                V = eps * (herm_log(gammas[i], jitter=jitter) - herm_log(rho_i, jitter=jitter))
                U_list[i] = hermitianize(U_list[i] + V)  # eta_inner = 1
                pi = gibbs_state_from_potentials(U_list, H, eps, dims, jitter=jitter, project=project_pi)

        converged = _record(pi, U_list)

    return MDsinkhornResult(
        F_list=F_list,
        e_tr_list=e_tr_list,
        per_i_tr_list=per_i_tr_list,
        times=times,
        pi=pi,
        U_list=U_list,
        U_hist=U_hist,
        pi_list=pi_list,
        converged=converged,
        gibbs_calls=PI_COUNTER.n_calls,
    )


# ============================================================
# Algorithm (1): Potential marginal KL descent (your "KL algorithm")
# ============================================================

def potential_marginal_kl_descent(
    H: np.ndarray,
    gammas: List[np.ndarray],
    eps: float,
    dims: List[int],
    T: int = 200,
    eta: Optional[float] = None,
    jitter_log: float = 1e-12,
    tol_tr: Optional[float] = None,
    store_hist: bool = False,
    project_pi: bool = True,
) -> PotentialKLDescentResult:
    """
    U_i <- U_i - eta (log rho_i - log gamma_i),  pi <- Gibbs(U).
    """
    PI_COUNTER.reset()
    N = len(dims)
    if len(gammas) != N:
        raise ValueError(f"len(gammas)={len(gammas)} must equal len(dims)={N}")
    if eta is None:
        eta = 1.0 / N

    U_list = [np.zeros((dims[i], dims[i]), dtype=complex) for i in range(N)]
    pi = gibbs_state_from_potentials(U_list, H, eps, dims, jitter=jitter_log, project=project_pi)

    F_list = [F_marg(pi, gammas, dims, jitter=jitter_log)]
    per_i = marginal_trace_errors(pi, gammas, dims)
    per_i_tr_list = [per_i]
    e_tr_list = [float(np.max(per_i))]
    times = [0.0]

    U_hist = [[Ui.copy() for Ui in U_list]] if store_hist else None
    pi_list = [pi.copy()] if store_hist else None

    t0 = time.time()
    converged = False

    for n in range(T):
        rho_list = [partial_trace_except_i(pi, dims, i) for i in range(N)]
        for i in range(N):
            log_rho = herm_log(rho_list[i], jitter=jitter_log)
            log_gam = herm_log(gammas[i], jitter=jitter_log)
            U_list[i] = hermitianize(U_list[i] - eta * (log_rho - log_gam))

        pi = gibbs_state_from_potentials(U_list, H, eps, dims, jitter=jitter_log, project=project_pi)

        if store_hist:
            U_hist.append([Ui.copy() for Ui in U_list])
            pi_list.append(pi.copy())

        elapsed = time.time() - t0
        F_list.append(F_marg(pi, gammas, dims, jitter=jitter_log))
        per_i = marginal_trace_errors(pi, gammas, dims)
        per_i_tr_list.append(per_i)
        e_tr = float(np.max(per_i))
        e_tr_list.append(e_tr)
        times.append(elapsed)

        if tol_tr is not None and e_tr <= tol_tr:
            converged = True
            break

    return PotentialKLDescentResult(
        F_list=F_list,
        e_tr_list=e_tr_list,
        per_i_tr_list=per_i_tr_list,
        times=times,
        pi=pi,
        U_list=U_list,
        U_hist=U_hist,
        pi_list=pi_list,
        converged=converged,
        gibbs_calls=PI_COUNTER.n_calls,
    )


# ============================================================
# Algorithm (2): BGDA / DBGA baseline (general N, default N=2 use-case)
# ============================================================

def block_gradient_ascent(
    H: np.ndarray,
    gammas: List[np.ndarray],
    eps: float,
    dims: List[int],
    T: int = 300,
    eta: float = 4.0,
    tol_tr: float = 1e-6,
    gauge_trace0: bool = True,
    store_hist: bool = False,
    project_pi: bool = True,
) -> DBGAResult:
    """
    General N-marginal block gradient ascent in potentials:

      U_i <- U_i + eta * (gamma_i - Tr_{≠i}(pi))
      pi  <- Gibbs(U)

    Default use-case is N=2 (two marginals). This general version also works for N>2.
    """
    PI_COUNTER.reset()
    N = len(dims)
    if len(gammas) != N:
        raise ValueError(f"len(gammas)={len(gammas)} must equal len(dims)={N}")

    U_list = [np.zeros((dims[i], dims[i]), dtype=complex) for i in range(N)]
    pi = gibbs_state_from_potentials(U_list, H, eps, dims, project=project_pi)

    F_list: List[float] = []
    e_tr_list: List[float] = []
    per_i_tr_list: List[np.ndarray] = []
    times: List[float] = []
    U_hist = [[Ui.copy() for Ui in U_list]] if store_hist else None
    pi_list = [pi.copy()] if store_hist else None

    t0 = time.time()
    converged = False

    for k in range(T):
        per_i = marginal_trace_errors(pi, gammas, dims)
        e_tr = float(np.max(per_i))
        Fv = float(F_marg(pi, gammas, dims))
        F_list.append(Fv)
        e_tr_list.append(e_tr)
        per_i_tr_list.append(per_i)
        times.append(time.time() - t0)

        if e_tr <= tol_tr:
            converged = True
            break

        # block updates
        rho_list = [partial_trace_except_i(pi, dims, i) for i in range(N)]
        for i in range(N):
            Ui = U_list[i] + eta * hermitianize(gammas[i] - rho_list[i])
            U_list[i] = hermitianize(Ui)

        if gauge_trace0:
            U_list = _gauge_fix_trace0_all(U_list, dims)

        pi = gibbs_state_from_potentials(U_list, H, eps, dims, project=project_pi)

        if store_hist:
            U_hist.append([Ui.copy() for Ui in U_list])
            pi_list.append(pi.copy())

    # record one final point if loop ended by max_iter without recording last pi update
    if len(F_list) == 0 or (times and times[-1] < (time.time() - t0) - 1e-12):
        per_i = marginal_trace_errors(pi, gammas, dims)
        e_tr = float(np.max(per_i))
        Fv = float(F_marg(pi, gammas, dims))
        F_list.append(Fv)
        e_tr_list.append(e_tr)
        per_i_tr_list.append(per_i)
        times.append(time.time() - t0)

    return DBGAResult(
        F_list=F_list,
        e_tr_list=e_tr_list,
        per_i_tr_list=per_i_tr_list,
        times=times,
        pi=pi,
        U_list=U_list,
        U_hist=U_hist,
        pi_list=pi_list,
        converged=converged,
        gibbs_calls=PI_COUNTER.n_calls,
    )

# ============================================================
# Algorithm (2) supplement: find the optimal eta
# ============================================================
@dataclass
class BGDATuneRecord:
    eta: float
    converged: bool
    stop_iter: int
    final_e_tr: float
    final_F: float
    time_sec: float
    gibbs_calls: int
    message: str = ""


@dataclass
class BGDATuneResult:
    best_eta: float
    best_run: DBGAResult
    records: List[BGDATuneRecord]


def tune_bgda_eta(
    H: np.ndarray,
    gammas: List[np.ndarray],
    eps: float,
    dims: List[int],
    eta_grid: List[float],
    T: int = 300,
    tol_tr: float = 1e-6,
    gauge_trace0: bool = True,
    project_pi: bool = True,
    store_hist: bool = False,
    prefer: str = "gibbs_calls",   # or "time_sec"
) -> BGDATuneResult:
    """
    Grid-search eta for BGDA (block_gradient_ascent).

    Selection rule:
      - If any run converges: pick the converged run with minimal (prefer) then minimal final_e_tr.
      - Else: pick run with minimal final_e_tr, tie-break by (prefer).

    prefer: "gibbs_calls" or "time_sec"
    """
    if prefer not in ("gibbs_calls", "time_sec"):
        raise ValueError("prefer must be 'gibbs_calls' or 'time_sec'.")

    records: List[BGDATuneRecord] = []
    best_run: Optional[DBGAResult] = None
    best_eta: Optional[float] = None

    for eta in eta_grid:
        t0 = time.time()
        try:
            res = block_gradient_ascent(
                H=H,
                gammas=gammas,
                eps=eps,
                dims=dims,
                T=T,
                eta=float(eta),
                tol_tr=tol_tr,
                gauge_trace0=gauge_trace0,
                store_hist=store_hist,
                project_pi=project_pi,
            )
            elapsed = time.time() - t0

            final_e = float(res.e_tr_list[-1]) if len(res.e_tr_list:=(res.e_tr_list)) else float("inf")
            final_F = float(res.F_list[-1]) if len(res.F_list) else float("inf")
            stop_iter = len(res.e_tr_list) - 1

            rec = BGDATuneRecord(
                eta=float(eta),
                converged=bool(res.converged),
                stop_iter=int(stop_iter),
                final_e_tr=final_e,
                final_F=final_F,
                time_sec=float(elapsed),
                gibbs_calls=int(res.gibbs_calls),
                message="ok",
            )
            records.append(rec)

        except Exception as e:
            elapsed = time.time() - t0
            records.append(
                BGDATuneRecord(
                    eta=float(eta),
                    converged=False,
                    stop_iter=-1,
                    final_e_tr=float("inf"),
                    final_F=float("inf"),
                    time_sec=float(elapsed),
                    gibbs_calls=0,
                    message=f"error: {type(e).__name__}: {e}",
                )
            )
            continue

        # update best choice
        if best_run is None:
            best_run, best_eta = res, float(eta)
        else:
            # compare current run to best_run using the selection rule
            cur_conv = bool(res.converged)
            best_conv = bool(best_run.converged)

            cur_final_e = float(res.e_tr_list[-1])
            best_final_e = float(best_run.e_tr_list[-1])

            cur_pref = res.gibbs_calls if prefer == "gibbs_calls" else (res.times[-1] if res.times else float("inf"))
            best_pref = best_run.gibbs_calls if prefer == "gibbs_calls" else (best_run.times[-1] if best_run.times else float("inf"))

            def better() -> bool:
                # If one converges and the other doesn't, convergent wins.
                if cur_conv and not best_conv:
                    return True
                if best_conv and not cur_conv:
                    return False
                # Both converge OR both don't: compare primary criterion
                if cur_conv and best_conv:
                    # among converged, minimize prefer then final_e
                    if cur_pref < best_pref - 1e-12:
                        return True
                    if abs(cur_pref - best_pref) <= 1e-12 and cur_final_e < best_final_e - 1e-12:
                        return True
                    return False
                else:
                    # among non-converged, minimize final_e then prefer
                    if cur_final_e < best_final_e - 1e-12:
                        return True
                    if abs(cur_final_e - best_final_e) <= 1e-12 and cur_pref < best_pref - 1e-12:
                        return True
                    return False

            if better():
                best_run, best_eta = res, float(eta)

    if best_run is None or best_eta is None:
        raise RuntimeError("tune_bgda_eta: no successful runs (all errored).")

    return BGDATuneResult(best_eta=best_eta, best_run=best_run, records=records)