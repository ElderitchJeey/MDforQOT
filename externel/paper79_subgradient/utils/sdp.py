import cvxpy as cp
import jax.numpy as jnp


def solve_dual_mosek(
        cost_matrix: jnp.array,
        ptraces: tuple[jnp.array, ...],
        max_iters: int = 3000,
        tol_rel_gap: float = 1e-6,
        verbose: bool = False
) -> tuple[tuple[jnp.array, ...], jnp.ndarray]:
    """
    Solves a dual semidefinite programming (SDP) problem using MOSEK solver.
    Args:
        :param cost_matrix: A `jnp.array` object representing the cost matrix.
        :param ptraces: A tuple of `jnp.array` objects representing the partial traces.
        :param max_iters: The maximum number of iterations for the MOSEK (default: 3000).
        :param tol_rel_gap: Relative gap tolerance for the MOSEK (default: 1e-6)
        :param verbose: A boolean indicating whether to display solver progress (default: False).
    Returns:
        Two tuples of `jnp.array` objects representing the primal and dual variables of the SDP problem.
    """
    fom = 0
    direct_sum = 0
    for i, ptr in enumerate(ptraces):
        var = cp.Variable(ptr.shape, symmetric=True)
        fom += cp.trace(var @ ptr)

        if i != 0:
            matrix = jnp.eye(ptraces[0].shape[0])
        else:
            matrix = var
        for j, ptr2 in enumerate(ptraces[1:], start=1):
            if i != j:
                matrix = cp.kron(matrix, jnp.eye(ptr2.shape[0]))
            else:
                matrix = cp.kron(matrix, var)
        direct_sum += matrix

    # Define the objective function
    objective = cp.Maximize(fom)

    # Define constraints
    constraints = [
        cost_matrix >> direct_sum,
    ]

    # Formulate the SDP problem
    problem = cp.Problem(objective, constraints)

    mosek_params = {
        'MSK_IPAR_INTPNT_MAX_ITERATIONS': max_iters,
        'MSK_DPAR_INTPNT_TOL_REL_GAP': tol_rel_gap,
        'MSK_IPAR_LOG': 1}

    # Solve the SDP problem
    problem.solve(solver='MOSEK', mosek_params=mosek_params, verbose=verbose)

    return tuple(jnp.array(v) for v in problem.solution.primal_vars.values()), \
    tuple(jnp.array(v) for v in problem.solution.dual_vars.values())[0]
