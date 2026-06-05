import jax
import jax.numpy as jnp
import jax.nn as nn
from functools import partial

from .dual import get_dual_value, cost_minus_kron_sum
from .mixture import mix_vectors_and_return_ptraces


@partial(jax.jit, static_argnums=(4, 5, 6))
def get_value_and_grad_qot(
        duals: tuple[jnp.ndarray, ...],
        ptraces: tuple[jnp.ndarray, ...],
        cost_matrix: jnp.ndarray,
        reg: float,
        eigensolver: callable,
        dims: tuple[int, ...],
        system_parts: tuple[tuple[int, ...], ...],
        *args, **kwargs
) -> tuple[float, tuple[jnp.ndarray, ...]]:
    """Compute the value and gradient of the entropy regularized quantum optimal transport (EQOT) dual problem.

    This function evaluates the dual objective function and its gradient for regularized
    quantum optimal transport.

    Parameters
    ----------
    duals : tuple[jnp.ndarray, ...]
        Dual variables (Lagrange multipliers) for each marginal constraint.
        Each array should match the shape of corresponding partial traces.
    ptraces : tuple[jnp.ndarray, ...]
        Target partial traces (marginals) for each subsystem combination.
        Should have same structure as what `system_parts` would produce.
    cost_matrix : jnp.ndarray
        The cost matrix defining the transport problem.
        Shape: (d, d) where d is the total Hilbert space dimension.
    reg : float
        Regularization strength (entropic regularization parameter).
        Must be positive.
    eigensolver : callable
        Function that solves the eigenvalue problem for (cost_matrix - dual_operator).
        Signature: (cost_matrix, duals, *args, **kwargs) -> (eigenvalues, eigenvectors)
    dims : tuple[int, ...]
        Dimensions of individual quantum subsystems (static argument).
    system_parts : tuple[tuple[int, ...], ...]
        Subsystem combinations specifying the marginals (static argument).
    *args
        Additional positional arguments passed to eigensolver.
    **kwargs
        Additional keyword arguments passed to eigensolver.

    Returns
    -------
    tuple[float, tuple[jnp.ndarray, ...]]
        A tuple containing:
        - The dual objective value (float)
        - The gradient with respect to dual variables (same structure as input duals)

    Examples
    --------
    >>> # Two-qubit system with trivial cost matrix
    >>> dims = (2, 2)
    >>> system_parts = ((0,), (1,))  # Margin for each qubit
    >>> cost_matrix = jnp.eye(4)
    >>> reg = 0.1
    >>>
    >>> # Define marginals (maximally mixed states here)
    >>> ptraces = (jnp.eye(2)/2, jnp.eye(2)/2)
    >>> duals = (jnp.zeros((2,2)), jnp.zeros((2,2)))
    >>>
    >>> def eigensolver(cost, duals):
    ...     # Simple eigensolver for demonstration
    ...     H = cost - sum(jnp.kron(d, jnp.eye(2)) for d in duals)  # Approximate
    ...     return jnp.linalg.eigh(H)
    ...
    >>> value, grad = get_value_and_grad_qot(duals, ptraces, cost_matrix,
    ...                                     reg, eigensolver, dims, system_parts)

    Notes
    -----
    1. The regularization parameter 'reg' controls the strength of entropic smoothing.
    2. The gradient is computed as the difference between target marginals (ptraces)
       and the current predicted marginals from the optimal coupling.
    3. For large systems, consider using approximate eigensolvers and GPU acceleration.
    4. The function is compatible with JAX transformations (grad, jit, etc.).
    """
    dual_loss = get_dual_value(duals, ptraces)
    eigvals, eigvecs = eigensolver(cost_matrix, duals, *args, **kwargs)

    log_sum_exp = reg * nn.logsumexp(-eigvals / reg)
    value = dual_loss - log_sum_exp

    exp_eigvals = nn.softmax(-eigvals / reg)
    eigvec_mixture = mix_vectors_and_return_ptraces(eigvecs, exp_eigvals, dims, system_parts)
    grad = jax.tree.map(lambda p, m: p - m, ptraces, eigvec_mixture)

    return value, grad


def reconstruct_primal(cost_matrix: jnp.ndarray,
                       duals: tuple[jnp.ndarray, ...],
                       reg: float
) -> jnp.ndarray:
    h_eff = cost_minus_kron_sum(cost_matrix, duals)
    eigvals, eigvecs = jnp.linalg.eigh(h_eff)
    eigvals = nn.softmax(-eigvals / reg)
    return eigvecs @ jnp.diag(eigvals) @ eigvecs.T


def get_entropy_reg_qot(
        duals: tuple[jnp.ndarray, ...],
        ptraces: tuple[jnp.ndarray, ...],
        cost_matrix: jnp.ndarray,
        reg: float,
) -> float:
    dual_loss = get_dual_value(duals, ptraces)
    h_eff = cost_minus_kron_sum(cost_matrix, duals)
    eigvals, eigvecs = jnp.linalg.eigh(h_eff)
    log_sum_exp = reg * jax.scipy.special.logsumexp(-eigvals / reg)
    value = dual_loss - log_sum_exp
    return value
