import jax
import jax.numpy as jnp
from functools import partial
import optax
import optax.tree_utils as otu
import time

from utils import get_all_ptraces_pure, stochastic_lanczos, get_dual_value, get_entropy_reg_qot

VAREPSILON = 1e-12


@partial(jax.jit, static_argnums=(2, 3, 4, 5, 6, 7, 8, 9))
def run_subgrad_polyak(
        cost_matrix: jnp.ndarray,
        ptraces: tuple[jnp.ndarray, ...],
        dims: tuple[int, ...],
        system_parts: tuple[tuple[int, ...], ...],
        max_iter: int,
        tol: float,
        lanczos_max_iter: int = 2,
        lanczos_tol: float = 1e-6,
        verbose: bool = False,
        log_every: int = 50
):
    def step(carry):
        duals, subg_prev, vec, f_best, eigval, it, lz_calls, loss_log, grad_log = carry

        # --- Lanczos call & counter ---
        new_eigval, new_vec = stochastic_lanczos(
            cost_matrix, duals, vec,
            max_iter=lanczos_max_iter,
            tol=lanczos_tol
        )
        lz_calls += 1

        # --- 2) gradient & update ---
        value = -(get_dual_value(duals, ptraces) + new_eigval)
        f_best = jnp.where(value < f_best, value, f_best)
        ptraces_pure = get_all_ptraces_pure(new_vec, dims, system_parts)
        subg = jax.tree_util.tree_map(lambda p, pp: pp - p,
                                      ptraces, ptraces_pure)
        correction = jnp.abs(f_best) / (it + 1)
        step_size = (value - f_best + correction) / (4. * N)
        duals = jax.tree_util.tree_map(lambda d, s: d - step_size * s,
                                       duals, subg)

        # --- log the “loss” (we use the eigenvalue as proxy) ---
        loss_log = loss_log.at[it].set(-value)

        grad_norm = optax.tree_utils.tree_l2_norm(subg)
        grad_log = grad_log.at[it].set(grad_norm)

        # --- debug print every log_every iterations ---
        cond = verbose & (it % log_every == 0)

        # dtype for formatting must be array-scalar; debug.print handles it
        def _do_print(args):
            i, val, calls = args
            jax.debug.print("[iter {i}] eigval={val:.6e} lz_calls={calls}",
                            i=i, val=val, calls=calls)

        _ = jax.lax.cond(cond,
                         _do_print,
                         lambda _: None,
                         operand=(it, new_eigval, lz_calls))

        return (duals, subg, new_vec, f_best, new_eigval,
                it + 1, lz_calls, loss_log, grad_log)

    def cond(carry):
        _, _, _, _, _, it, _, _, grad_log = carry
        err = grad_log[it]
        # return (it == 0) | ((it < max_iter) & (err >= tol))
        return (it < 2) | ((it < max_iter) & (err >= tol))

    # --- initialize ---
    N = len(ptraces)
    init_duals = jax.tree_util.tree_map(jnp.zeros_like, ptraces)
    # key = jax.random.PRNGKey(10)
    # init_duals = tuple(jax.random.normal(key, p.shape) for p in ptraces)
    # init_duals = jax.tree_util.tree_map(make_herm, init_duals)
    # init_duals = jax.tree_util.tree_map(make_traceless, init_duals)
    init_step_prev = jax.tree_util.tree_map(jnp.zeros_like, ptraces)

    vec0 = jax.random.normal(jax.random.PRNGKey(0), cost_matrix.shape[0])
    vec0 = vec0 / jnp.linalg.norm(vec0)

    loss_log0 = jnp.zeros((max_iter,))
    grad_log0 = jnp.zeros((max_iter,))

    init_carry = (
        init_duals,  # duals
        init_step_prev,  # step
        vec0,  # eigvec
        0.,  # optimal value
        0.,  # eigval
        0,  # iteration counter
        0,  # lanczos call counter
        loss_log0,  # loss history
        grad_log0  # grad norm history
    )

    final = jax.lax.while_loop(cond, step, init_carry)
    duals_f, step_f, vec_f, f_best_f, eig_f, it_f, lz_f, loss_f, grad_f = final

    tol_reached = jnp.where(it_f < max_iter, True, False)

    return {
        "duals": duals_f,
        "grad": step_f,
        "best_f": -f_best_f,
        "eigvec": vec_f,
        "eigval": eig_f,
        "n_iters": it_f,
        "lanczos_calls": lz_f,
        "loss_history": loss_f,
        "grad_norm_history": grad_f,
        "tol_reached": tol_reached
    }


@partial(jax.jit, static_argnums=(3, 4, 5, 6))
def run_lbfgs(
    cost_matrix: jnp.ndarray,
    ptraces: tuple[jnp.ndarray, ...],
    reg: float,
    max_iter: int = 100000,
    tol: float = 1e-6,
    verbose: bool = False,
    log_every: int = 50
):
    """
    Runs L-BFGS on the entropy-regularized QOT dual.
    Adds counters for iteration, logs the objective value each step,
    and prints progress every `log_every` iters if `verbose=True`.

    Returns a dict with:
      - 'params': final dual variables
      - 'state': final Optax state (contains 'count' & 'grad')
      - 'loss_history': array of objective values per iteration
    """
    # 1) Define objective + grad
    fun = lambda dls: -get_entropy_reg_qot(dls, ptraces, cost_matrix, reg)

    # 2) Initialize duals
    # key = jax.random.PRNGKey(10)
    # init_duals = tuple(jax.random.normal(key, p.shape) for p in ptraces)
    # init_duals = jax.tree_util.tree_map(make_herm, init_duals)
    # init_duals = jax.tree_util.tree_map(make_traceless, init_duals)
    init_duals = jax.tree_util.tree_map(jnp.zeros_like, ptraces)

    # 3) Setup L-BFGS
    opt = optax.lbfgs()
    value_and_grad_fun = optax.value_and_grad_from_state(fun)
    state0 = opt.init(init_duals)

    # 4) Pre-allocate loss log
    loss_log0 = jnp.zeros((max_iter,))

    # 5) Loop body
    def step(carry):
        params, state, loss_log = carry

        # compute value & gradient
        value, grad = value_and_grad_fun(params, state=state)

        # L-BFGS update
        updates, state = opt.update(
            grad, state, params,
            value=value, grad=grad, value_fn=fun
        )
        params = optax.apply_updates(params, updates)

        # iteration count
        iter_num = otu.tree_get(state, 'count')

        # log objective value
        loss_log = loss_log.at[iter_num].set(-value)

        # optional debug print
        cond = verbose & (iter_num % log_every == 0)
        def _print(args):
            i, v = args
            jax.debug.print("[iter {i}] value={v:.6e}", i=i, v=v)
        _ = jax.lax.cond(cond, _print, lambda x: None,
                         operand=(iter_num, value))

        return params, state, loss_log

    # 6) Loop condition
    def cont(carry):
        _, state, _ = carry
        it = otu.tree_get(state, 'count')
        grad = otu.tree_get(state, 'grad')
        err = otu.tree_l2_norm(grad)
        # return (it == 0) | ((it < max_iter) & (err >= tol))
        return (it < 2) | ((it < max_iter) & (err >= tol))

    # 7) Run loop
    init_carry = (init_duals, state0, loss_log0)
    final_params, final_state, loss_log = jax.lax.while_loop(
        cont, step, init_carry
    )

    # 8) Slice to actual iterations
    n_iters = otu.tree_get(final_state, 'count')
    tol_reached = jnp.where(n_iters < max_iter, True, False)
    return {
        'params': final_params,
        'n_iters': n_iters,
        'state': final_state[0],
        'loss_history': loss_log,
        "tol_reached": tol_reached
    }


def subgrad_tol3(cost_matrix, ptraces, dims, system_parts):
    start = time.time()
    state = run_subgrad_polyak(
        cost_matrix,
        ptraces,
        dims,
        system_parts,
        max_iter = 500000,
        tol = 1e-3,
        lanczos_max_iter = 2,
        lanczos_tol = 1e-6,
        verbose = False,
        log_every = 50
    )
    end = time.time()
    n_iters, loss_history, tol_reached = state['n_iters'], state['loss_history'], state['tol_reached']
    return {
        'loss_history': [float(v) for v in loss_history[:n_iters]],
        'tol_reached': bool(tol_reached),
        'fun_eval': int(n_iters),
        'time': end - start
    }

def subgrad_tol4(cost_matrix, ptraces, dims, system_parts):
    start = time.time()
    state = run_subgrad_polyak(
        cost_matrix,
        ptraces,
        dims,
        system_parts,
        max_iter = 500000,
        tol = 1e-4,
        lanczos_max_iter = 2,
        lanczos_tol = 1e-6,
        verbose = False,
        log_every = 50
    )
    end = time.time()
    n_iters, loss_history, tol_reached = state['n_iters'], state['loss_history'], state['tol_reached']
    return {
        'loss_history': [float(v) for v in loss_history[:n_iters]],
        'tol_reached': bool(tol_reached),
        'fun_eval': int(n_iters),
        'time': end - start
    }

def subgrad_tol5(cost_matrix, ptraces, dims, system_parts):
    start = time.time()
    state = run_subgrad_polyak(
        cost_matrix,
        ptraces,
        dims,
        system_parts,
        max_iter = 500000,
        tol = 1e-5,
        lanczos_max_iter = 2,
        lanczos_tol = 1e-6,
        verbose = False,
        log_every = 50
    )
    end = time.time()
    n_iters, loss_history, tol_reached = state['n_iters'], state['loss_history'], state['tol_reached']
    return {
        'loss_history': [float(v) for v in loss_history[:n_iters]],
        'tol_reached': bool(tol_reached),
        'fun_eval': int(n_iters),
        'time': end - start
    }

def subgrad_tol6(cost_matrix, ptraces, dims, system_parts):
    start = time.time()
    state = run_subgrad_polyak(
        cost_matrix,
        ptraces,
        dims,
        system_parts,
        max_iter = 500000,
        tol = 1e-6,
        lanczos_max_iter = 2,
        lanczos_tol = 1e-6,
        verbose = False,
        log_every = 50
    )
    end = time.time()
    n_iters, loss_history, tol_reached = state['n_iters'], state['loss_history'], state['tol_reached']
    return {
        'loss_history': [float(v) for v in loss_history[:n_iters]],
        'tol_reached': bool(tol_reached),
        'fun_eval': int(n_iters),
        'time': end - start
    }


def lbfgs_tol3(cost_matrix, ptraces, dims, system_parts):
    start = time.time()
    state = run_lbfgs(
        cost_matrix,
        ptraces,
        VAREPSILON,
        max_iter = 10000,
        tol = 1e-3,
        verbose = False,
        log_every = 50
    )
    end = time.time()
    n_iters, loss_history, tol_reached = state['n_iters'], state['loss_history'], state['tol_reached']
    return {
        'loss_history': [float(v) for v in loss_history[:n_iters]],
        'tol_reached': bool(tol_reached),
        'fun_eval': int(n_iters),
        'time': end - start
    }

def lbfgs_tol4(cost_matrix, ptraces, dims, system_parts):
    start = time.time()
    state = run_lbfgs(
        cost_matrix,
        ptraces,
        VAREPSILON,
        max_iter = 10000,
        tol = 1e-4,
        verbose = False,
        log_every = 50
    )
    end = time.time()
    n_iters, loss_history, tol_reached = state['n_iters'], state['loss_history'], state['tol_reached']
    return {
        'loss_history': [float(v) for v in loss_history[:n_iters]],
        'tol_reached': bool(tol_reached),
        'fun_eval': int(n_iters),
        'time': end - start
    }

def lbfgs_tol5(cost_matrix, ptraces, dims, system_parts):
    start = time.time()
    state = run_lbfgs(
        cost_matrix,
        ptraces,
        VAREPSILON,
        max_iter = 10000,
        tol = 1e-5,
        verbose = False,
        log_every = 50
    )
    end = time.time()
    n_iters, loss_history, tol_reached = state['n_iters'], state['loss_history'], state['tol_reached']
    return {
        'loss_history': [float(v) for v in loss_history[:n_iters]],
        'tol_reached': bool(tol_reached),
        'fun_eval': int(n_iters),
        'time': end - start
    }

def lbfgs_tol6(cost_matrix, ptraces, dims, system_parts):
    start = time.time()
    state = run_lbfgs(
        cost_matrix,
        ptraces,
        VAREPSILON,
        max_iter = 10000,
        tol = 1e-6,
        verbose = False,
        log_every = 50
    )
    end = time.time()
    n_iters, loss_history, tol_reached = state['n_iters'], state['loss_history'], state['tol_reached']
    return {
        'loss_history': [float(v) for v in loss_history[:n_iters]],
        'tol_reached': bool(tol_reached),
        'fun_eval': int(n_iters),
        'time': end - start
    }