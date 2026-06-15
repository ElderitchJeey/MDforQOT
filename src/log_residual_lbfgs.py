"""L-BFGS acceleration of logarithmic marginal residuals.

This module is intentionally separate from ``SolverofEQOT.py``.  It is an
experimental solver meant to be easy to read while staying close to the
paper79 L-BFGS structure:

    1. Define a scalar objective.
    2. Use ``optax.lbfgs`` for the quasi-Newton direction and line search.
    3. Record a QOT-style trajectory in Gibbs calls.

The only conceptual change is the scalar objective.  Instead of minimizing the
negative entropy dual, we minimize a merit function built from the logarithmic
marginal residual

    S_i(U) = log T_i(pi_U) - log rho_i.

This is not "L-BFGS on the dual".  It is a quasi-Newton acceleration of the
nonlinear equation S(U)=0, implemented through the least-squares merit

    Phi(U) = 1/2 sum_i ||S_i(U)||_F^2.

That choice lets us reuse JAX autodiff, Optax L-BFGS, and Optax line search.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.linalg import hermitianize


@dataclass
class LogResidualLBFGSResult:
    """Result object matching the repo-native solver protocol."""

    F_list: List[float]
    e_tr_list: List[float]
    merit_list: List[float]
    per_i_tr_list: List[np.ndarray]
    times: List[float]
    pi: np.ndarray
    U_list: List[np.ndarray]
    U_hist: Optional[List[List[np.ndarray]]] = None
    pi_list: Optional[List[np.ndarray]] = None
    converged: bool = False
    gibbs_calls: int = 0
    gibbs_calls_list: Optional[List[int]] = None
    line_search_steps_list: Optional[List[int]] = None


def _require_jax_optax():
    """Import JAX/Optax lazily so the module can still be imported without them."""

    import jax  # type: ignore
    import jax.numpy as jnp  # type: ignore
    import optax  # type: ignore
    import optax.tree_utils as otu  # type: ignore

    return jax, jnp, optax, otu


def _validate_inputs(
    H: np.ndarray,
    gammas: Sequence[np.ndarray],
    dims: Sequence[int],
    eps: float,
    U0: Optional[Sequence[np.ndarray]],
) -> None:
    if eps <= 0:
        raise ValueError("eps must be positive.")
    if len(gammas) != len(dims):
        raise ValueError("len(gammas) must equal len(dims).")
    D = int(np.prod([int(d) for d in dims]))
    if np.asarray(H).shape != (D, D):
        raise ValueError(f"H must have shape {(D, D)}.")
    for i, (gamma, di) in enumerate(zip(gammas, dims)):
        if np.asarray(gamma).shape != (int(di), int(di)):
            raise ValueError(f"gammas[{i}] must have shape {(int(di), int(di))}.")
    if U0 is not None:
        if len(U0) != len(dims):
            raise ValueError("len(U0) must equal len(dims).")
        for i, (Ui, di) in enumerate(zip(U0, dims)):
            if np.asarray(Ui).shape != (int(di), int(di)):
                raise ValueError(f"U0[{i}] must have shape {(int(di), int(di))}.")


def _to_numpy_U(U_list: Sequence[Any]) -> List[np.ndarray]:
    return [hermitianize(np.asarray(Ui, dtype=complex)) for Ui in U_list]


def accelerated_log_marginal_lbfgs(
    *,
    H: np.ndarray,
    gammas: Sequence[np.ndarray],
    eps: float,
    dims: Sequence[int],
    max_iter: int = 1000,
    tol_tr: float = 1e-6,
    tol_merit: Optional[float] = None,
    log_floor: float = 1e-12,
    U0: Optional[Sequence[np.ndarray]] = None,
    history: int = 10,
    store_hist: bool = False,
    gauge_fix: bool = True,
    marginal_mode: str = "density",
    max_gibbs_calls: Optional[int] = None,
    dtype: Any = None,
) -> LogResidualLBFGSResult:
    """Run L-BFGS on the logarithmic marginal residual merit.

    Args:
        H:
            Full Hamiltonian on ``tensor_i C^{dims[i]}``.
        gammas:
            Target one-site marginals ``rho_i``.
        eps:
            Entropic regularization parameter in
            ``pi_U proportional exp((L(U)-H)/eps)``.
        dims:
            Local dimensions.
        max_iter:
            Maximum number of L-BFGS outer iterations.  One outer iteration may
            use several Gibbs evaluations because the Optax line search tries
            multiple step lengths.
        tol_tr:
            Stop when ``max_i ||T_i(pi_U)-rho_i||_1 <= tol_tr``.
        tol_merit:
            Optional stopping threshold for ``Phi(U)``.
        log_floor:
            Eigenvalue floor used in all matrix logarithms.
        U0:
            Optional initial potentials.  If omitted, all potentials start at 0.
        history:
            Number of curvature pairs stored by L-BFGS.
        store_hist:
            If true, store full potential/coupling histories.  This can be
            expensive for large systems.
        gauge_fix:
            If true, subtract ``Tr(U_i)/d_i I`` after every update.  This only
            changes ``L(U)`` by a scalar multiple of identity and therefore does
            not change the normalized Gibbs state.
        marginal_mode:
            ``"density"`` constructs the full Gibbs state and then takes partial
            traces. ``"eigen"`` follows paper79's idea of mixing eigenvector
            marginals without first forming the full density matrix.  The latter
            can be useful to experiment with, but is not always faster for small
            qubit systems.
        max_gibbs_calls:
            Optional budget in counted Gibbs-state evaluations.  The count
            includes the L-BFGS objective/line-search calls and one extra
            Gibbs evaluation per recorded diagnostic point.
        dtype:
            Optional JAX dtype.  Defaults to float64/complex128 when enabled.

    Returns:
        ``LogResidualLBFGSResult`` with the same public fields as the existing
        KL/MD solvers plus ``merit_list`` and ``line_search_steps_list``.
    """

    _validate_inputs(H, gammas, dims, eps, U0)
    if marginal_mode not in {"density", "eigen"}:
        raise ValueError("marginal_mode must be 'density' or 'eigen'.")
    jax, jnp, optax, otu = _require_jax_optax()
    jax.config.update("jax_enable_x64", True)

    dims_tuple = tuple(int(d) for d in dims)
    N = len(dims_tuple)
    if dtype is None:
        has_complex = np.iscomplexobj(H) or any(np.iscomplexobj(g) for g in gammas)
        dtype = jnp.complex128 if has_complex else jnp.float64

    H_j = jnp.asarray(hermitianize(np.asarray(H, dtype=complex)), dtype=dtype)
    gammas_j = tuple(jnp.asarray(hermitianize(np.asarray(g, dtype=complex)), dtype=dtype) for g in gammas)

    if U0 is None:
        U = tuple(jnp.zeros((di, di), dtype=dtype) for di in dims_tuple)
    else:
        U = tuple(jnp.asarray(hermitianize(np.asarray(Ui, dtype=complex)), dtype=dtype) for Ui in U0)

    def hermitianize_j(A):
        return 0.5 * (A + jnp.conj(jnp.swapaxes(A, -1, -2)))

    def herm_log_2x2_j(A):
        """Fast log for 2x2 Hermitian positive definite matrices.

        For a 2x2 Hermitian matrix with eigenvalues mu +/- delta,
        log(A) = c I + s (A - mu I).  The small-delta branch uses the
        derivative limit of log at mu.
        """

        A = hermitianize_j(A)
        a = jnp.real(A[0, 0])
        d = jnp.real(A[1, 1])
        b = A[0, 1]
        mu = 0.5 * (a + d)
        delta = jnp.sqrt(jnp.maximum(0.25 * (a - d) ** 2 + jnp.real(b * jnp.conj(b)), 0.0))
        I = jnp.eye(2, dtype=A.dtype)

        floor = float(log_floor)
        mu_safe = jnp.maximum(mu, floor)
        lp = jnp.maximum(mu + delta, floor)
        lm = jnp.maximum(mu - delta, floor)

        c = 0.5 * (jnp.log(lp) + jnp.log(lm))
        s = (jnp.log(lp) - jnp.log(lm)) / (2.0 * delta)
        out_exact = c * I + s * (A - mu * I)
        out_small = jnp.log(mu_safe) * I + (A - mu * I) / mu_safe
        use_small = delta < 1e-14 * jnp.maximum(mu_safe, 1.0)
        return hermitianize_j(jnp.where(use_small, out_small, out_exact))

    def herm_log_j(A):
        A = hermitianize_j(A)
        if A.shape == (2, 2):
            return herm_log_2x2_j(A)
        w, V = jnp.linalg.eigh(A)
        w = jnp.clip(w, float(log_floor), jnp.inf)
        return hermitianize_j((V * jnp.log(w)) @ jnp.conj(V).T)

    log_gammas_j = tuple(herm_log_j(gamma_i) for gamma_i in gammas_j)

    def gauge_fix_U(U_list):
        if not gauge_fix:
            return tuple(hermitianize_j(Ui) for Ui in U_list)
        out = []
        for Ui, di in zip(U_list, dims_tuple):
            Ui = hermitianize_j(Ui)
            alpha = jnp.real(jnp.trace(Ui)) / float(di)
            out.append(hermitianize_j(Ui - alpha * jnp.eye(di, dtype=Ui.dtype)))
        return tuple(out)

    def q_i_lift(Ui, i: int):
        out = jnp.asarray([[1.0]], dtype=Ui.dtype)
        for j, dj in enumerate(dims_tuple):
            if j == i:
                factor = Ui
            else:
                factor = jnp.eye(dj, dtype=Ui.dtype)
            out = jnp.kron(out, factor)
        return hermitianize_j(out)

    def L_of_U_j(U_list):
        D = int(np.prod(dims_tuple))
        out = jnp.zeros((D, D), dtype=H_j.dtype)
        for i, Ui in enumerate(U_list):
            out = out + q_i_lift(Ui, i)
        return hermitianize_j(out)

    def gibbs_eigendecomp_j(U_list):
        A = hermitianize_j((L_of_U_j(U_list) - H_j) / float(eps))
        w, V = jnp.linalg.eigh(A)
        w_shift = w - jnp.max(w)
        weights = jnp.exp(w_shift)
        weights = weights / jnp.sum(weights)
        return weights, V

    def gibbs_state_j(U_list):
        weights, V = gibbs_eigendecomp_j(U_list)
        pi = (V * weights) @ jnp.conj(V).T
        return hermitianize_j(pi)

    def pure_marginal_i_j(vector, i: int):
        psi = jnp.reshape(vector, dims_tuple)
        qtrace = tuple(j for j in range(N) if j != i)
        rho = jnp.tensordot(psi, jnp.conj(psi), axes=(qtrace, qtrace))
        di = dims_tuple[i]
        return hermitianize_j(jnp.reshape(rho, (di, di)))

    def mixed_marginal_i_from_eig_j(vectors, weights, i: int):
        def body(k, carry):
            return carry + weights[k] * pure_marginal_i_j(vectors[:, k], i)

        di = dims_tuple[i]
        init = weights[0] * pure_marginal_i_j(vectors[:, 0], i)
        return hermitianize_j(jax.lax.fori_loop(1, vectors.shape[1], body, init))

    def gibbs_marginals_j(U_list):
        weights, V = gibbs_eigendecomp_j(U_list)
        return tuple(mixed_marginal_i_from_eig_j(V, weights, i) for i in range(N))

    def gibbs_marginals_density_j(U_list):
        pi = gibbs_state_j(U_list)
        return tuple(partial_trace_except_i_j(pi, i) for i in range(N))

    def get_gibbs_marginals_j(U_list):
        if marginal_mode == "eigen":
            return gibbs_marginals_j(U_list)
        return gibbs_marginals_density_j(U_list)

    def partial_trace_except_i_j(rho, i: int):
        tensor = jnp.reshape(rho, dims_tuple + dims_tuple)
        keep_axes = (i, N + i)
        perm = keep_axes + tuple(ax for ax in range(2 * N) if ax not in keep_axes)
        tensor = jnp.transpose(tensor, perm)
        di = dims_tuple[i]
        rest_dim = int(np.prod([dims_tuple[j] for j in range(N) if j != i]))
        tensor = jnp.reshape(tensor, (di, di, rest_dim, rest_dim))
        out = jnp.einsum("abrr->ab", tensor)
        return hermitianize_j(out)

    def log_residuals_j(U_list):
        sigmas = get_gibbs_marginals_j(U_list)
        residuals = []
        for i in range(N):
            sigma_i = sigmas[i]
            residuals.append(herm_log_j(sigma_i) - log_gammas_j[i])
        return tuple(residuals)

    def merit_fn(U_list):
        U_list = gauge_fix_U(U_list)
        residuals = log_residuals_j(U_list)
        val = 0.0
        for Ri in residuals:
            val = val + jnp.real(jnp.trace(jnp.conj(Ri).T @ Ri))
        return 0.5 * val

    def trace_norm_hermitian_j(A):
        A = hermitianize_j(A)
        return jnp.sum(jnp.abs(jnp.linalg.eigvalsh(A)))

    def diagnostics_j(U_list):
        """Compute trajectory diagnostics from one Gibbs state evaluation."""

        U_list = gauge_fix_U(U_list)
        sigmas = get_gibbs_marginals_j(U_list)
        F_val = 0.0
        merit_val = 0.0
        per_i = []
        for i in range(N):
            sigma_i = sigmas[i]
            log_sigma = herm_log_j(sigma_i)
            log_gamma = log_gammas_j[i]
            # Repo convention: KL(T_i(pi) || gamma_i).
            F_val = F_val + jnp.real(jnp.trace(sigma_i @ (log_sigma - log_gamma)))
            Ri = log_sigma - log_gamma
            merit_val = merit_val + jnp.real(jnp.trace(jnp.conj(Ri).T @ Ri))
            per_i.append(trace_norm_hermitian_j(sigma_i - gammas_j[i]))
        per_i_arr = jnp.stack(per_i) if per_i else jnp.zeros((0,), dtype=jnp.float64)
        e_tr = jnp.max(per_i_arr) if per_i else jnp.asarray(0.0, dtype=jnp.float64)
        return 0.5 * merit_val, F_val, e_tr, per_i_arr

    opt = optax.lbfgs(memory_size=int(history))
    opt_state = opt.init(U)
    value_and_grad = optax.value_and_grad_from_state(merit_fn)
    diagnostics_jit = jax.jit(diagnostics_j)

    @jax.jit
    def step_jit(U_list, state):
        value, grad = value_and_grad(U_list, state=state)
        updates, state = opt.update(
            grad,
            state,
            U_list,
            value=value,
            grad=grad,
            value_fn=merit_fn,
        )
        U_next = gauge_fix_U(optax.apply_updates(U_list, updates))
        merit_next, F_next, e_next, per_i_next = diagnostics_jit(U_next)
        line_steps = jnp.asarray(otu.tree_get(state, "num_linesearch_steps", default=0), dtype=jnp.int32)
        return U_next, state, merit_next, F_next, e_next, per_i_next, line_steps

    F_list: List[float] = []
    e_tr_list: List[float] = []
    merit_list: List[float] = []
    per_i_tr_list: List[np.ndarray] = []
    times: List[float] = []
    gibbs_calls_list: List[int] = []
    line_search_steps_list: List[int] = []
    U_hist: Optional[List[List[np.ndarray]]] = [] if store_hist else None
    pi_list: Optional[List[np.ndarray]] = [] if store_hist else None

    t0 = time.time()
    gibbs_calls = 0
    converged = False

    def record_metrics(
        *,
        F_value: Any,
        e_tr_value: Any,
        merit_value: Any,
        per_i_value: Any,
        U_list: Any,
    ) -> None:
        Fv = float(jax.device_get(F_value))
        e_tr = float(jax.device_get(e_tr_value))
        merit = float(jax.device_get(merit_value))
        per_i = np.asarray(jax.device_get(per_i_value), dtype=float)

        F_list.append(Fv)
        e_tr_list.append(e_tr)
        merit_list.append(merit)
        per_i_tr_list.append(per_i)
        times.append(time.time() - t0)
        gibbs_calls_list.append(int(gibbs_calls))
        if store_hist:
            assert U_hist is not None and pi_list is not None
            U_np = _to_numpy_U(jax.device_get(U_list))
            pi_np = np.asarray(jax.device_get(gibbs_state_j(U_list)), dtype=complex)
            U_hist.append([Ui.copy() for Ui in U_np])
            pi_list.append(pi_np.copy())

    # Initial diagnostics evaluate one Gibbs state.
    initial_merit, initial_F, initial_e, initial_per_i = diagnostics_jit(U)
    gibbs_calls += 1
    record_metrics(
        F_value=initial_F,
        e_tr_value=initial_e,
        merit_value=initial_merit,
        per_i_value=initial_per_i,
        U_list=U,
    )
    converged = bool(e_tr_list[-1] <= float(tol_tr))
    if tol_merit is not None:
        converged = bool(converged or merit_list[-1] <= float(tol_merit))

    for _k in range(int(max_iter)):
        if converged:
            break
        if max_gibbs_calls is not None and gibbs_calls >= int(max_gibbs_calls):
            break

        U, opt_state, current_merit, current_F, current_e, current_per_i, line_steps_j = step_jit(U, opt_state)
        line_steps = int(jax.device_get(line_steps_j))
        line_search_steps_list.append(line_steps)
        gibbs_calls += 1 + line_steps  # L-BFGS objective/line-search calls.
        gibbs_calls += 1  # Recorded diagnostics from diagnostics_jit.
        record_metrics(
            F_value=current_F,
            e_tr_value=current_e,
            merit_value=current_merit,
            per_i_value=current_per_i,
            U_list=U,
        )
        converged = bool(e_tr_list[-1] <= float(tol_tr))
        if tol_merit is not None:
            converged = bool(converged or merit_list[-1] <= float(tol_merit))

    U_final = _to_numpy_U(jax.device_get(U))
    pi_j = gibbs_state_j(U)
    gibbs_calls += 1
    pi = np.asarray(jax.device_get(pi_j), dtype=complex)
    if gibbs_calls_list:
        gibbs_calls_list[-1] = int(gibbs_calls)
    return LogResidualLBFGSResult(
        F_list=F_list,
        e_tr_list=e_tr_list,
        merit_list=merit_list,
        per_i_tr_list=per_i_tr_list,
        times=times,
        pi=pi,
        U_list=U_final,
        U_hist=U_hist,
        pi_list=pi_list,
        converged=converged,
        gibbs_calls=int(gibbs_calls),
        gibbs_calls_list=gibbs_calls_list,
        line_search_steps_list=line_search_steps_list,
    )


__all__ = ["LogResidualLBFGSResult", "accelerated_log_marginal_lbfgs"]
