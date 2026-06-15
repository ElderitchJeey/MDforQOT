"""JAX experimental KL/MD solvers using eigenvector-marginal mixtures.

These solvers are separate from ``SolverofEQOT.py``.  They borrow the main
engineering idea from the paper79 implementation: after the global Hermitian
eigendecomposition, compute one-site marginals directly as weighted mixtures of
pure-state marginals,

    T_i(pi_U) = sum_k w_k T_i(|v_k><v_k|),

instead of always constructing the full Gibbs density matrix first.  The full
coupling is reconstructed only once at the end.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from src.linalg import hermitianize


@dataclass
class JAXPotentialKLResult:
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
    eta: float = 0.0
    eta_rule: str = ""


@dataclass
class JAXMDSinkhornResult:
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


def _require_jax():
    import jax  # type: ignore
    import jax.numpy as jnp  # type: ignore

    return jax, jnp


def _resolve_kl_eta(eps: float, N: int, eta: Optional[float], eta_rule: str) -> Tuple[float, str]:
    if eta is not None:
        return float(eta), "manual"
    normalized = eta_rule.lower().replace("-", "_").strip()
    if normalized in {"eps", "full"}:
        return float(eps), "eps"
    if normalized in {"eps_over_n", "eps/n", "theory", "default", "eps_over_N".lower()}:
        return float(eps) / float(N), "eps_over_N"
    raise ValueError("eta_rule must be one of 'eps_over_N' or 'eps'.")


def _validate(H: np.ndarray, gammas: Sequence[np.ndarray], dims: Sequence[int], eps: float) -> None:
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


def _to_numpy_U(U_list: Sequence[Any]) -> List[np.ndarray]:
    return [hermitianize(np.asarray(Ui, dtype=complex)) for Ui in U_list]


def _make_kernel(
    *,
    H: np.ndarray,
    gammas: Sequence[np.ndarray],
    dims: Sequence[int],
    eps: float,
    log_floor: float,
    dtype: Any,
):
    jax, jnp = _require_jax()
    jax.config.update("jax_enable_x64", True)

    dims_tuple = tuple(int(d) for d in dims)
    N = len(dims_tuple)
    H_j = jnp.asarray(hermitianize(np.asarray(H, dtype=complex)), dtype=dtype)
    gammas_j = tuple(jnp.asarray(hermitianize(np.asarray(g, dtype=complex)), dtype=dtype) for g in gammas)

    def hermitianize_j(A):
        return 0.5 * (A + jnp.conj(jnp.swapaxes(A, -1, -2)))

    def herm_log_2x2_j(A):
        A = hermitianize_j(A)
        a = jnp.real(A[0, 0])
        d = jnp.real(A[1, 1])
        b2 = jnp.real(A[0, 1] * A[1, 0])
        mu = 0.5 * (a + d)
        delta = jnp.sqrt(jnp.maximum(0.25 * (a - d) * (a - d) + b2, 0.0))
        I = jnp.eye(2, dtype=A.dtype)
        floor = float(log_floor)
        mu_safe = jnp.maximum(mu, floor)
        lp = jnp.maximum(mu + delta, floor)
        lm = jnp.maximum(mu - delta, floor)
        c = 0.5 * (jnp.log(lp) + jnp.log(lm))
        s = (jnp.log(lp) - jnp.log(lm)) / (2.0 * delta)
        exact = c * I + s * (A - mu * I)
        small = jnp.log(mu_safe) * I + (A - mu * I) / mu_safe
        return hermitianize_j(jnp.where(delta < 1e-14 * jnp.maximum(mu_safe, 1.0), small, exact))

    def herm_log_j(A):
        A = hermitianize_j(A)
        if A.shape == (2, 2):
            return herm_log_2x2_j(A)
        w, V = jnp.linalg.eigh(A)
        w = jnp.clip(w, float(log_floor), jnp.inf)
        return hermitianize_j((V * jnp.log(w)) @ jnp.conj(V).T)

    log_gammas_j = tuple(herm_log_j(g) for g in gammas_j)

    def q_i_lift(Ui, i: int):
        out = jnp.asarray([[1.0]], dtype=Ui.dtype)
        for j, dj in enumerate(dims_tuple):
            factor = Ui if j == i else jnp.eye(dj, dtype=Ui.dtype)
            out = jnp.kron(out, factor)
        return hermitianize_j(out)

    def L_of_U_j(U_list):
        D = int(np.prod(dims_tuple))
        out = jnp.zeros((D, D), dtype=H_j.dtype)
        for i, Ui in enumerate(U_list):
            out = out + q_i_lift(Ui, i)
        return hermitianize_j(out)

    def gauge_fix_U(U_list):
        out = []
        for Ui, di in zip(U_list, dims_tuple):
            Ui = hermitianize_j(Ui)
            out.append(hermitianize_j(Ui - jnp.real(jnp.trace(Ui)) / float(di) * jnp.eye(di, dtype=Ui.dtype)))
        return tuple(out)

    def gibbs_eigendecomp_j(U_list):
        A = hermitianize_j((L_of_U_j(U_list) - H_j) / float(eps))
        w, V = jnp.linalg.eigh(A)
        weights = jnp.exp(w - jnp.max(w))
        weights = weights / jnp.sum(weights)
        return weights, V

    def pure_marginal_i_j(vector, i: int):
        psi = jnp.reshape(vector, dims_tuple)
        qtrace = tuple(j for j in range(N) if j != i)
        rho = jnp.tensordot(psi, jnp.conj(psi), axes=(qtrace, qtrace))
        di = dims_tuple[i]
        return hermitianize_j(jnp.reshape(rho, (di, di)))

    def mixed_marginal_i_j(vectors, weights, i: int):
        def body(k, carry):
            return carry + weights[k] * pure_marginal_i_j(vectors[:, k], i)

        init = weights[0] * pure_marginal_i_j(vectors[:, 0], i)
        return hermitianize_j(jax.lax.fori_loop(1, vectors.shape[1], body, init))

    def eig_marginals_j(U_list):
        weights, vectors = gibbs_eigendecomp_j(U_list)
        sigmas = tuple(mixed_marginal_i_j(vectors, weights, i) for i in range(N))
        return weights, vectors, sigmas

    def gibbs_state_from_eig_j(weights, vectors):
        return hermitianize_j((vectors * weights) @ jnp.conj(vectors).T)

    def trace_norm_hermitian_j(A):
        return jnp.sum(jnp.abs(jnp.linalg.eigvalsh(hermitianize_j(A))))

    def diagnostics_from_sigmas_j(sigmas):
        F_val = 0.0
        per_i = []
        for i in range(N):
            sigma = hermitianize_j(sigmas[i])
            log_sigma = herm_log_j(sigma)
            F_val = F_val + jnp.real(jnp.trace(sigma @ (log_sigma - log_gammas_j[i])))
            per_i.append(trace_norm_hermitian_j(sigma - gammas_j[i]))
        per_i_arr = jnp.stack(per_i) if per_i else jnp.zeros((0,), dtype=jnp.float64)
        e_tr = jnp.max(per_i_arr) if per_i else jnp.asarray(0.0, dtype=jnp.float64)
        return F_val, e_tr, per_i_arr

    @jax.jit
    def eval_state(U_list):
        U_list = gauge_fix_U(U_list)
        weights, vectors, sigmas = eig_marginals_j(U_list)
        F_val, e_tr, per_i = diagnostics_from_sigmas_j(sigmas)
        return U_list, weights, vectors, sigmas, F_val, e_tr, per_i

    @jax.jit
    def kl_update_from_sigmas(U_list, sigmas, eta):
        U_list = gauge_fix_U(U_list)
        updated = []
        for i, Ui in enumerate(U_list):
            log_sigma = herm_log_j(sigmas[i])
            updated.append(hermitianize_j(Ui - eta * (log_sigma - log_gammas_j[i])))
        return gauge_fix_U(tuple(updated))

    block_steps = []
    for block_i in range(N):
        def make_step(i: int):
            @jax.jit
            def md_update_from_sigmas(U_list, sigmas):
                U_list = gauge_fix_U(U_list)
                updated = list(U_list)
                V = float(eps) * (log_gammas_j[i] - herm_log_j(sigmas[i]))
                updated[i] = hermitianize_j(updated[i] + V)
                return gauge_fix_U(tuple(updated))

            return md_update_from_sigmas

        block_steps.append(make_step(block_i))

    @jax.jit
    def final_pi(U_list):
        U_list = gauge_fix_U(U_list)
        weights, vectors = gibbs_eigendecomp_j(U_list)
        return gibbs_state_from_eig_j(weights, vectors)

    return {
        "eval_state": eval_state,
        "kl_update_from_sigmas": kl_update_from_sigmas,
        "block_steps": block_steps,
        "final_pi": final_pi,
        "jnp": jnp,
        "jax": jax,
        "dims_tuple": dims_tuple,
    }


def jax_potential_marginal_kl_descent(
    H: np.ndarray,
    gammas: Sequence[np.ndarray],
    eps: float,
    dims: Sequence[int],
    T: int = 200,
    eta: Optional[float] = None,
    eta_rule: str = "eps_over_N",
    jitter_log: float = 1e-12,
    tol_tr: Optional[float] = None,
    tol_F: Optional[float] = None,
    U0: Optional[Sequence[np.ndarray]] = None,
    max_gibbs_calls: Optional[int] = None,
    dtype: Any = None,
    full_jit: bool = False,
) -> JAXPotentialKLResult:
    """JAX KL descent using paper79-style eigenvector marginal mixtures."""

    _validate(H, gammas, dims, eps)
    jax, jnp = _require_jax()
    if dtype is None:
        dtype = jnp.complex128 if (np.iscomplexobj(H) or any(np.iscomplexobj(g) for g in gammas)) else jnp.float64
    N = len(dims)
    eta_value, eta_rule_resolved = _resolve_kl_eta(eps, N, eta, eta_rule)
    kernel = _make_kernel(H=H, gammas=gammas, dims=dims, eps=eps, log_floor=jitter_log, dtype=dtype)

    if U0 is None:
        U = tuple(jnp.zeros((int(d), int(d)), dtype=dtype) for d in dims)
    else:
        U = tuple(jnp.asarray(hermitianize(np.asarray(Ui, dtype=complex)), dtype=dtype) for Ui in U0)

    if not full_jit:
        F_list: List[float] = []
        e_tr_list: List[float] = []
        per_i_tr_list: List[np.ndarray] = []
        times: List[float] = []
        gibbs_calls_list: List[int] = []
        gibbs_calls = 0
        converged = False
        t0 = time.time()

        def evaluate() -> Tuple[Any, Any, Any, Any]:
            nonlocal U, gibbs_calls
            U, _weights, _vectors, sigmas_j, F_j, e_j, per_i_j = kernel["eval_state"](U)
            gibbs_calls += 1
            return sigmas_j, F_j, e_j, per_i_j

        def record(F_j: Any, e_j: Any, per_i_j: Any) -> None:
            nonlocal converged
            Fv = float(jax.device_get(F_j))
            e = float(jax.device_get(e_j))
            F_list.append(Fv)
            e_tr_list.append(e)
            per_i_tr_list.append(np.asarray(jax.device_get(per_i_j), dtype=float))
            times.append(time.time() - t0)
            gibbs_calls_list.append(int(gibbs_calls))
            converged = bool((tol_F is not None and Fv <= float(tol_F)) or (tol_tr is not None and e <= float(tol_tr)))

        sigmas, F_j, e_j, per_i_j = evaluate()
        record(F_j, e_j, per_i_j)
        eta_j = jnp.asarray(float(eta_value), dtype=jnp.float64)
        for _ in range(int(T)):
            if converged:
                break
            if max_gibbs_calls is not None and gibbs_calls >= int(max_gibbs_calls):
                break
            U = kernel["kl_update_from_sigmas"](U, sigmas, eta_j)
            sigmas, F_j, e_j, per_i_j = evaluate()
            record(F_j, e_j, per_i_j)

        pi = np.asarray(jax.device_get(kernel["final_pi"](U)), dtype=complex)
        gibbs_calls += 1
        if gibbs_calls_list:
            gibbs_calls_list[-1] = int(gibbs_calls)
        return JAXPotentialKLResult(
            F_list=F_list,
            e_tr_list=e_tr_list,
            per_i_tr_list=per_i_tr_list,
            times=times,
            pi=pi,
            U_list=_to_numpy_U(jax.device_get(U)),
            converged=converged,
            gibbs_calls=int(gibbs_calls),
            gibbs_calls_list=gibbs_calls_list,
            eta=float(eta_value),
            eta_rule=eta_rule_resolved,
        )

    t0 = time.time()
    eta_j = jnp.asarray(float(eta_value), dtype=jnp.float64)

    tol_f_value = -np.inf if tol_F is None else float(tol_F)
    tol_tr_value = -np.inf if tol_tr is None else float(tol_tr)
    max_calls_value = int(max_gibbs_calls) if max_gibbs_calls is not None else (int(T) + 1)

    @jax.jit
    def run_scan(U_init):
        U0_eval, _weights, _vectors, sigmas0, F0, e0, per0 = kernel["eval_state"](U_init)
        done0 = (F0 <= tol_f_value) | (e0 <= tol_tr_value) | (jnp.asarray(1, dtype=jnp.int32) >= max_calls_value)
        gibbs0 = jnp.asarray(1, dtype=jnp.int32)

        def body(carry, _):
            U_curr, sigmas_curr, F_curr, e_curr, per_curr, done_curr, gibbs_curr = carry

            def do_update(args):
                U_in, sigmas_in, _F_in, _e_in, _per_in, _done_in, gibbs_in = args
                U_next = kernel["kl_update_from_sigmas"](U_in, sigmas_in, eta_j)
                U_next, _w, _v, sigmas_next, F_next, e_next, per_next = kernel["eval_state"](U_next)
                gibbs_next = gibbs_in + jnp.asarray(1, dtype=jnp.int32)
                done_next = (
                    (F_next <= tol_f_value)
                    | (e_next <= tol_tr_value)
                    | (gibbs_next >= max_calls_value)
                )
                return U_next, sigmas_next, F_next, e_next, per_next, done_next, gibbs_next

            def keep_state(args):
                U_in, sigmas_in, F_in, e_in, per_in, done_in, gibbs_in = args
                return U_in, sigmas_in, F_in, e_in, per_in, done_in, gibbs_in

            next_carry = jax.lax.cond(
                done_curr,
                keep_state,
                do_update,
                operand=(U_curr, sigmas_curr, F_curr, e_curr, per_curr, done_curr, gibbs_curr),
            )
            U_next, sigmas_next, F_next, e_next, per_next, done_next, gibbs_next = next_carry
            return next_carry, (F_next, e_next, per_next, gibbs_next, done_next)

        init = (U0_eval, sigmas0, F0, e0, per0, done0, gibbs0)
        final_carry, scan_out = jax.lax.scan(body, init, xs=None, length=int(T))
        F_tail, e_tail, per_tail, gibbs_tail, done_tail = scan_out
        F_hist = jnp.concatenate([jnp.asarray([F0]), F_tail])
        e_hist = jnp.concatenate([jnp.asarray([e0]), e_tail])
        per_hist = jnp.concatenate([per0[jnp.newaxis, :], per_tail], axis=0)
        gibbs_hist = jnp.concatenate([jnp.asarray([gibbs0]), gibbs_tail])
        done_hist = jnp.concatenate([jnp.asarray([done0]), done_tail])
        U_final = final_carry[0]
        return U_final, F_hist, e_hist, per_hist, gibbs_hist, done_hist

    U, F_j, e_j, per_i_j, gibbs_j, done_j = run_scan(U)
    F_arr = np.asarray(jax.device_get(F_j), dtype=float)
    e_arr = np.asarray(jax.device_get(e_j), dtype=float)
    per_arr = np.asarray(jax.device_get(per_i_j), dtype=float)
    gibbs_arr = np.asarray(jax.device_get(gibbs_j), dtype=int)
    done_arr = np.asarray(jax.device_get(done_j), dtype=bool)

    if np.any(done_arr):
        stop = int(np.argmax(done_arr)) + 1
    else:
        stop = len(F_arr)
    F_list = [float(x) for x in F_arr[:stop]]
    e_tr_list = [float(x) for x in e_arr[:stop]]
    per_i_tr_list = [np.asarray(x, dtype=float) for x in per_arr[:stop]]
    elapsed = time.time() - t0
    times = list(np.linspace(0.0, elapsed, num=stop))
    gibbs_calls_list = [int(x) for x in gibbs_arr[:stop]]
    converged = bool((tol_F is not None and F_list[-1] <= float(tol_F)) or (tol_tr is not None and e_tr_list[-1] <= float(tol_tr)))

    pi = np.asarray(jax.device_get(kernel["final_pi"](U)), dtype=complex)
    gibbs_calls = int(gibbs_calls_list[-1]) + 1
    gibbs_calls_list[-1] = int(gibbs_calls)
    return JAXPotentialKLResult(
        F_list=F_list,
        e_tr_list=e_tr_list,
        per_i_tr_list=per_i_tr_list,
        times=times,
        pi=pi,
        U_list=_to_numpy_U(jax.device_get(U)),
        converged=converged,
        gibbs_calls=int(gibbs_calls),
        gibbs_calls_list=gibbs_calls_list,
        eta=float(eta_value),
        eta_rule=eta_rule_resolved,
    )


def jax_md_type_sinkhorn_potential(
    H: np.ndarray,
    gammas: Sequence[np.ndarray],
    eps: float,
    dims: Sequence[int],
    T_outer: int,
    tol_tr: float,
    tol_F: Optional[float] = None,
    jitter: float = 1e-12,
    M_inner: int = 1,
    U0: Optional[Sequence[np.ndarray]] = None,
    max_gibbs_calls: Optional[int] = None,
    dtype: Any = None,
    full_jit: bool = False,
) -> JAXMDSinkhornResult:
    """JAX MD-Sinkhorn-style block correction using eigenvector marginals."""

    _validate(H, gammas, dims, eps)
    if M_inner < 1:
        raise ValueError("M_inner must be >= 1.")
    jax, jnp = _require_jax()
    if dtype is None:
        dtype = jnp.complex128 if (np.iscomplexobj(H) or any(np.iscomplexobj(g) for g in gammas)) else jnp.float64
    kernel = _make_kernel(H=H, gammas=gammas, dims=dims, eps=eps, log_floor=jitter, dtype=dtype)

    if U0 is None:
        U = tuple(jnp.zeros((int(d), int(d)), dtype=dtype) for d in dims)
    else:
        U = tuple(jnp.asarray(hermitianize(np.asarray(Ui, dtype=complex)), dtype=dtype) for Ui in U0)

    if not full_jit:
        F_list: List[float] = []
        e_tr_list: List[float] = []
        per_i_tr_list: List[np.ndarray] = []
        times: List[float] = []
        gibbs_calls_list: List[int] = []
        gibbs_calls = 0
        converged = False
        t0 = time.time()

        def evaluate() -> Tuple[Any, Any, Any, Any]:
            nonlocal U, gibbs_calls
            U, _weights, _vectors, sigmas_j, F_j, e_j, per_i_j = kernel["eval_state"](U)
            gibbs_calls += 1
            return sigmas_j, F_j, e_j, per_i_j

        def record(F_j: Any, e_j: Any, per_i_j: Any) -> None:
            nonlocal converged
            Fv = float(jax.device_get(F_j))
            e = float(jax.device_get(e_j))
            F_list.append(Fv)
            e_tr_list.append(e)
            per_i_tr_list.append(np.asarray(jax.device_get(per_i_j), dtype=float))
            times.append(time.time() - t0)
            gibbs_calls_list.append(int(gibbs_calls))
            converged = bool(e <= float(tol_tr) or (tol_F is not None and Fv <= float(tol_F)))

        sigmas, F_j, e_j, per_i_j = evaluate()
        record(F_j, e_j, per_i_j)
        for _outer in range(int(T_outer)):
            if converged:
                break
            if max_gibbs_calls is not None and gibbs_calls >= int(max_gibbs_calls):
                break
            for step_i in kernel["block_steps"]:
                for _inner in range(int(M_inner)):
                    if max_gibbs_calls is not None and gibbs_calls >= int(max_gibbs_calls):
                        break
                    U = step_i(U, sigmas)
                    sigmas, F_j, e_j, per_i_j = evaluate()
            record(F_j, e_j, per_i_j)

        pi = np.asarray(jax.device_get(kernel["final_pi"](U)), dtype=complex)
        gibbs_calls += 1
        if gibbs_calls_list:
            gibbs_calls_list[-1] = int(gibbs_calls)
        return JAXMDSinkhornResult(
            F_list=F_list,
            e_tr_list=e_tr_list,
            per_i_tr_list=per_i_tr_list,
            times=times,
            pi=pi,
            U_list=_to_numpy_U(jax.device_get(U)),
            converged=converged,
            gibbs_calls=int(gibbs_calls),
            gibbs_calls_list=gibbs_calls_list,
        )

    t0 = time.time()
    tol_f_value = -np.inf if tol_F is None else float(tol_F)
    max_calls_value = (
        int(max_gibbs_calls)
        if max_gibbs_calls is not None
        else (1 + int(T_outer) * len(dims) * int(M_inner))
    )

    @jax.jit
    def run_scan(U_init):
        U0_eval, _weights, _vectors, sigmas0, F0, e0, per0 = kernel["eval_state"](U_init)
        done0 = (e0 <= float(tol_tr)) | (F0 <= tol_f_value) | (jnp.asarray(1, dtype=jnp.int32) >= max_calls_value)
        gibbs0 = jnp.asarray(1, dtype=jnp.int32)

        def one_outer(carry, _):
            U_curr, sigmas_curr, F_curr, e_curr, per_curr, done_curr, gibbs_curr = carry

            def do_outer(args):
                U_in, sigmas_in, F_in, e_in, per_in, done_in, gibbs_in = args
                U_work, sigmas_work, F_work, e_work, per_work, gibbs_work = (
                    U_in,
                    sigmas_in,
                    F_in,
                    e_in,
                    per_in,
                    gibbs_in,
                )
                for step_i in kernel["block_steps"]:
                    for _inner in range(int(M_inner)):
                        U_work = step_i(U_work, sigmas_work)
                        U_work, _w, _v, sigmas_work, F_work, e_work, per_work = kernel["eval_state"](U_work)
                        gibbs_work = gibbs_work + jnp.asarray(1, dtype=jnp.int32)
                done_work = (e_work <= float(tol_tr)) | (F_work <= tol_f_value) | (gibbs_work >= max_calls_value)
                return U_work, sigmas_work, F_work, e_work, per_work, done_work, gibbs_work

            def keep_state(args):
                return args

            next_carry = jax.lax.cond(
                done_curr,
                keep_state,
                do_outer,
                operand=(U_curr, sigmas_curr, F_curr, e_curr, per_curr, done_curr, gibbs_curr),
            )
            U_next, sigmas_next, F_next, e_next, per_next, done_next, gibbs_next = next_carry
            return next_carry, (F_next, e_next, per_next, gibbs_next, done_next)

        init = (U0_eval, sigmas0, F0, e0, per0, done0, gibbs0)
        final_carry, scan_out = jax.lax.scan(one_outer, init, xs=None, length=int(T_outer))
        F_tail, e_tail, per_tail, gibbs_tail, done_tail = scan_out
        F_hist = jnp.concatenate([jnp.asarray([F0]), F_tail])
        e_hist = jnp.concatenate([jnp.asarray([e0]), e_tail])
        per_hist = jnp.concatenate([per0[jnp.newaxis, :], per_tail], axis=0)
        gibbs_hist = jnp.concatenate([jnp.asarray([gibbs0]), gibbs_tail])
        done_hist = jnp.concatenate([jnp.asarray([done0]), done_tail])
        U_final = final_carry[0]
        return U_final, F_hist, e_hist, per_hist, gibbs_hist, done_hist

    U, F_j, e_j, per_i_j, gibbs_j, done_j = run_scan(U)
    F_arr = np.asarray(jax.device_get(F_j), dtype=float)
    e_arr = np.asarray(jax.device_get(e_j), dtype=float)
    per_arr = np.asarray(jax.device_get(per_i_j), dtype=float)
    gibbs_arr = np.asarray(jax.device_get(gibbs_j), dtype=int)
    done_arr = np.asarray(jax.device_get(done_j), dtype=bool)
    if np.any(done_arr):
        stop = int(np.argmax(done_arr)) + 1
    else:
        stop = len(F_arr)
    F_list = [float(x) for x in F_arr[:stop]]
    e_tr_list = [float(x) for x in e_arr[:stop]]
    per_i_tr_list = [np.asarray(x, dtype=float) for x in per_arr[:stop]]
    elapsed = time.time() - t0
    times = list(np.linspace(0.0, elapsed, num=stop))
    gibbs_calls_list = [int(x) for x in gibbs_arr[:stop]]
    converged = bool(e_tr_list[-1] <= float(tol_tr) or (tol_F is not None and F_list[-1] <= float(tol_F)))

    pi = np.asarray(jax.device_get(kernel["final_pi"](U)), dtype=complex)
    gibbs_calls = int(gibbs_calls_list[-1]) + 1
    gibbs_calls_list[-1] = int(gibbs_calls)
    return JAXMDSinkhornResult(
        F_list=F_list,
        e_tr_list=e_tr_list,
        per_i_tr_list=per_i_tr_list,
        times=times,
        pi=pi,
        U_list=_to_numpy_U(jax.device_get(U)),
        converged=converged,
        gibbs_calls=int(gibbs_calls),
        gibbs_calls_list=gibbs_calls_list,
    )


__all__ = [
    "JAXMDSinkhornResult",
    "JAXPotentialKLResult",
    "jax_md_type_sinkhorn_potential",
    "jax_potential_marginal_kl_descent",
]
