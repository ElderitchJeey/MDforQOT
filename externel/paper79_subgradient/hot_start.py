import json
import argparse
from functools import partial
import optax
import optax.tree_utils as otu
import time

from examples import *
from solvers import run_subgrad_polyak
from test_instances import tests
from utils import get_entropy_reg_qot


SUBGRAD_ITER = 5000
SUBGRAD_TOL = 1e-3
VAREPSILON = 1e-12
LBFGS_ITER = 10000
LBFGS_TOL = 1e-6


def _subgrad_f(cost_matrix, ptraces, dims, system_parts):
    return run_subgrad_polyak(
            cost_matrix,
            ptraces,
            dims,
            system_parts,
            max_iter = SUBGRAD_ITER,
            tol = SUBGRAD_TOL,
            lanczos_max_iter = 2,
            lanczos_tol = 1e-6,
            verbose = False,
            log_every = 50
    )


@partial(jax.jit, static_argnums=(4, 5, 6, 7))
def run_lbfgs_hot(
    cost_matrix: jnp.ndarray,
    ptraces: tuple[jnp.ndarray, ...],
    init_duals: tuple[jnp.ndarray, ...],
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
    # key = jax.random.PRNGKey(0)
    # init_duals = tuple(jax.random.normal(key, p.shape) for p in ptraces)
    # init_duals = jax.tree_util.tree_map(make_herm, init_duals)
    # init_duals = jax.tree_util.tree_map(make_traceless, init_duals)
    # init_duals = jax.tree_util.tree_map(jnp.zeros_like, ptraces)

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run a subset of tests and dump to JSON"
    )
    parser.add_argument(
        "--index", type=int,
        help="(0-based) index of the test in all_tests to run"
    )
    args = parser.parse_args()

    test = tests[args.index]
    cost_matrix, ptraces, dims, system_parts = test

    test_logs = dict()

    subgrad_f = jax.jit(lambda cm, ptrs: _subgrad_f(cm, ptrs, dims, system_parts))
    subgrad_lowered = subgrad_f.lower(cost_matrix, ptraces)
    subgrad_executable = subgrad_lowered.compile()
    start = time.time()
    subgrad_state = subgrad_executable(cost_matrix, ptraces)
    end = time.time()

    n_iters, loss_history, tol_reached, hot_start = subgrad_state['n_iters'], subgrad_state['loss_history'], subgrad_state['tol_reached'], subgrad_state[
        'duals']
    subgrad_log = {
        'loss_history': [float(v) for v in loss_history[:n_iters]],
        'tol_reached': bool(tol_reached),
        'fun_eval': int(n_iters),
        'time': end - start
    }
    test_logs['subgrad_log'] = subgrad_log

    hot_start = jax.tree.map(lambda hs: hs + subgrad_state['eigval'] / len(hot_start) * jnp.eye(hs.shape[0]), hot_start)

    lbfgs_f = jax.jit(lambda cm, ptrs, hs: run_lbfgs_hot(cm, ptrs, hs, VAREPSILON, LBFGS_ITER, LBFGS_TOL))
    lbfgs_lowered = lbfgs_f.lower(cost_matrix, ptraces, hot_start)
    lbfgs_executable = lbfgs_lowered.compile()

    start = time.time()
    lbfgs_state = lbfgs_executable(cost_matrix, ptraces, hot_start)
    end = time.time()

    n_iters, loss_history, tol_reached = lbfgs_state['n_iters'], lbfgs_state['loss_history'], lbfgs_state['tol_reached']
    lbfgs_log = {
        'loss_history': [float(v) for v in loss_history[:n_iters]],
        'tol_reached': bool(tol_reached),
        'fun_eval': int(n_iters),
        'time': end - start
    }
    test_logs['lbfgs_log'] = lbfgs_log

    with open(f'hot_start_data/test{args.index}_hot.json', 'w') as file:
        json.dump(test_logs, file)
