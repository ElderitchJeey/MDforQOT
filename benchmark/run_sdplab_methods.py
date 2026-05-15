# benchmark/run_sdplab_methods.py

from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

import os
import time
import numpy as np

from benchmark.diagnostics import final_diagnostics


@dataclass
class SDPLabBenchmarkResult:
    F_list: List[float]
    e_tr_list: List[float]
    times: List[float]

    pi: np.ndarray
    U_list: Optional[List[np.ndarray]]

    converged: bool
    gibbs_calls: int
    gibbs_calls_list: List[int]

    dual_obj: Optional[List[float]] = None
    grad_norm: Optional[List[float]] = None
    compile_time: float = 0.0
    raw: Optional[Any] = None


def _make_jax_context():
    """
    Build a complex128 JAX context for SDPLab.
    """
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

    import jax.numpy as jnp
    from spacecore import Context, JaxOps

    try:
        return Context(JaxOps(), dtype=jnp.complex128, enable_checks=False)
    except TypeError:
        return Context(JaxOps(), dtype=jnp.complex128)


def build_sdplab_qot_regularized_problem(
    *,
    H: np.ndarray,
    gammas: Sequence[np.ndarray],
    eps: float,
    dims: Sequence[int],
):
    """
    Build SDPLab's trace-normalized regularized QOT problem.

    Assumes equal local dimensions dims=[d,...,d].
    """
    dims = [int(x) for x in dims]

    if len(set(dims)) != 1:
        raise ValueError(
            "SDPLab QOTConstraintOp expects equal local dimensions. "
            f"Got dims={dims}."
        )

    d = int(dims[0])
    N = int(len(dims))
    D = d ** N

    H = np.asarray(H, dtype=np.complex128)
    if H.shape != (D, D):
        raise ValueError(f"H has shape {H.shape}, expected {(D, D)}.")

    if len(gammas) != N:
        raise ValueError(f"len(gammas)={len(gammas)} but N={N}.")

    ctx = _make_jax_context()

    from sdplab.special.qot import QOTConstraintOp
    from sdplab.sdp import SDPDenseProblem
    from sdplab.regularization import SDPRegularized, EntropyRegLog

    qot_op = QOTConstraintOp(
        d=d,
        N=N,
        atol=1e-9,
        rtol=1e-9,
        enforce_herm=True,
        ctx=ctx,
    )

    C = ctx.asarray(H)
    b = ctx.asarray(np.stack(gammas, axis=0).astype(np.complex128))

    base_sdp = SDPDenseProblem(
        C=C,
        A=qot_op,
        b=b,
        tau=1.0,
        ctx=ctx,
    )

    reg = EntropyRegLog(float(eps), ctx=ctx)
    reg_sdp = SDPRegularized(base_sdp, reg)

    return reg_sdp


def _recover_pi(reg_sdp, dual) -> np.ndarray:
    """
    Recover normalized density matrix from SDPLab dual variable.
    """
    primal = reg_sdp.primal_from_dual(dual, normalized=True)
    pi = np.asarray(primal.X)
    pi = (pi + pi.conj().T) / 2
    tr = float(np.trace(pi).real)
    if abs(tr) <= 1e-300:
        raise ValueError("Recovered primal has near-zero trace.")
    pi = pi / tr
    return pi


def _dual_to_U_list(dual) -> List[np.ndarray]:
    y = np.asarray(dual.y)
    return [(y[i] + y[i].conj().T) / 2 for i in range(y.shape[0])]


def run_sdplab_adam(
    *,
    H: np.ndarray,
    gammas: Sequence[np.ndarray],
    eps: float,
    dims: Sequence[int],
    tol_F: float,
    max_gibbs_calls: int,
    lr: float = 1e-2,
    grad_tol: float = 1e-8,
    checkpoint_every: Optional[int] = 10,
    verbose: bool = False,
    log_every: int = 50,
) -> SDPLabBenchmarkResult:
    """
    Run SDPLab Optax regularized-dual solver with Adam.

    For Adam, we count:
        gibbs_calls = optimizer iterations.
    """
    import optax
    from sdplab.solvers import run_optax_solver

    reg_sdp = build_sdplab_qot_regularized_problem(
        H=H,
        gammas=gammas,
        eps=eps,
        dims=dims,
    )

    base_sdp = reg_sdp.sdp
    dims = [int(x) for x in dims]
    N = len(dims)
    d = dims[0]

    init_y = np.zeros((N, d, d), dtype=np.complex128)
    init_dual = base_sdp.dual_from_array(base_sdp.ctx.asarray(init_y))

    opt = optax.adam(float(lr))

    wall_start = time.time()

    try:
        info = run_optax_solver(
            reg_sdp,
            init_dual,
            opt=opt,
            max_iter=int(max_gibbs_calls),
            tol=float(grad_tol),
            verbose=verbose,
            log_every=log_every,
            checkpoint_every=checkpoint_every,
            return_primal=True,
            primal_normalized=True,
        )
    except TypeError:
        # Fallback for unpatched SDPLab.
        info = run_optax_solver(
            reg_sdp,
            init_dual,
            opt=opt,
            max_iter=int(max_gibbs_calls),
            tol=float(grad_tol),
            verbose=verbose,
            log_every=log_every,
        )

    wall_total = time.time() - wall_start

    if getattr(info, "primal", None) is not None:
        pi = np.asarray(info.primal.X)
        pi = (pi + pi.conj().T) / 2
        pi = pi / float(np.trace(pi).real)
    else:
        pi = _recover_pi(reg_sdp, info.dual)

    U_list = _dual_to_U_list(info.dual)

    dual_obj_arr = np.asarray(info.dual_obj, dtype=float) if info.dual_obj is not None else np.zeros(0)
    grad_norm_arr = np.asarray(info.grad_norm, dtype=float) if info.grad_norm is not None else np.zeros(0)

    n_iters = int(getattr(info, "iterations", len(dual_obj_arr)) or len(dual_obj_arr))
    gibbs_calls = int(getattr(info, "gibbs_calls", n_iters) or n_iters)

    run_time = float(getattr(info, "time", wall_total) or wall_total)
    compile_time = float(getattr(info, "compile_time", max(0.0, wall_total - run_time)) or 0.0)

    F_list: List[float] = []
    e_tr_list: List[float] = []
    times: List[float] = []
    gibbs_calls_list: List[int] = []

    dual_checkpoints = getattr(info, "dual_checkpoints", None)
    checkpoint_gibbs_calls = getattr(info, "checkpoint_gibbs_calls", None)

    if dual_checkpoints is not None and checkpoint_gibbs_calls is not None:
        y_ckpts = np.asarray(dual_checkpoints)
        g_ckpts = np.asarray(checkpoint_gibbs_calls)

        for y, g in zip(y_ckpts, g_ckpts):
            dual_k = base_sdp.dual_from_array(base_sdp.ctx.asarray(np.asarray(y)))
            pi_k = _recover_pi(reg_sdp, dual_k)

            diag_k = final_diagnostics(
                pi=pi_k,
                H=H,
                gammas=list(gammas),
                eps=eps,
                dims=dims,
                tol_F=tol_F,
            )

            F_list.append(float(diag_k["final_F_marg"]))
            e_tr_list.append(float(diag_k["final_e_tr_max"]))
            gibbs_calls_list.append(int(g))

        if gibbs_calls > 0:
            times = [run_time * (g / float(gibbs_calls)) for g in gibbs_calls_list]
        else:
            times = [0.0 for _ in gibbs_calls_list]

    # If no checkpoints are available, record final point only.
    if len(F_list) == 0:
        diag = final_diagnostics(
            pi=pi,
            H=H,
            gammas=list(gammas),
            eps=eps,
            dims=dims,
            tol_F=tol_F,
        )
        F_list = [float(diag["final_F_marg"])]
        e_tr_list = [float(diag["final_e_tr_max"])]
        gibbs_calls_list = [gibbs_calls]
        times = [run_time]

        # Always append the final point if it is not already present.
    diag_final = final_diagnostics(
        pi=pi,
        H=H,
        gammas=list(gammas),
        eps=eps,
        dims=dims,
        tol_F=tol_F,
    )

    if len(gibbs_calls_list) == 0 or int(gibbs_calls_list[-1]) != int(gibbs_calls):
        F_list.append(float(diag_final["final_F_marg"]))
        e_tr_list.append(float(diag_final["final_e_tr_max"]))
        gibbs_calls_list.append(int(gibbs_calls))
        times.append(float(run_time))

    final_hit = bool(F_list[-1] <= float(tol_F))

    return SDPLabBenchmarkResult(
        F_list=F_list,
        e_tr_list=e_tr_list,
        times=times,
        pi=pi,
        U_list=U_list,
        converged=final_hit,
        gibbs_calls=gibbs_calls,
        gibbs_calls_list=gibbs_calls_list,
        dual_obj=dual_obj_arr.tolist(),
        grad_norm=grad_norm_arr.tolist(),
        compile_time=compile_time,
        raw=info,
    )