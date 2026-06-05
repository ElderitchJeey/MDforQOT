import jax
import jax.numpy as jnp
from functools import partial


@jax.jit
def get_dual_value(duals: tuple[jnp.ndarray, ...], ptraces: tuple[jnp.ndarray, ...]) -> float:
    """
    Computes the dual value of quantum optimal transport (QOT).
    Args:
        duals: A tuple of `jnp.ndarray` objects representing the dual matrices.
        ptraces: A tuple of `jnp.ndarray` objects representing the partial traces.
    Returns:
        A float representing the real part of dual loss.
    """
    # loss = 0
    # for var, ptr in zip(duals, ptraces):
    #     loss += jnp.trace(var @ ptr)

    loss = sum(jax.tree.map(lambda d, p: jnp.trace(d @ p), duals, ptraces))

    return loss.real


@jax.jit
def get_kronecker_sum(duals: tuple[jnp.ndarray, ...]) -> jnp.ndarray:
    """
    Computes the Kronecker sum of matrices using JAX library.
    Args:
        duals: A tuple of `jnp.ndarray` objects representing the matrices.
    Returns:
        A `jnp.ndarray` object representing the direct sum of the matrices.
    """

    kron_sum = 0
    for i in range(len(duals)):
        if i == 0:
            matrix = duals[0]
        else:
            matrix = jnp.eye(duals[0].shape[0])
        for j, var in enumerate(duals[1:], start=1):
            if i == j:
                matrix = jnp.kron(matrix, var)
            else:
                matrix = jnp.kron(matrix, jnp.eye(var.shape[0]))
        kron_sum = kron_sum + matrix

    return kron_sum


@jax.jit
def cost_minus_kron_sum(cost_matrix: jnp.ndarray, duals: tuple[jnp.ndarray, ...]) -> jnp.ndarray:
    kron_sum = get_kronecker_sum(duals)

    return cost_matrix - kron_sum


@partial(jax.jit, static_argnums=(2,))
def kron_sum_mvp(
        duals: tuple[jnp.ndarray, ...],
        vector: jnp.ndarray,
        dims: tuple[int, ...] = None,
) -> jnp.ndarray:
    """
    Optimized MVP for Kronecker sum with batch support.

    Args:
        duals: Tuple of (d_i, d_i) matrices
        vector: Input (prod(d_i),) or (prod(d_i), k)
        dims: Optional precomputed dimensions (static for JIT)

    Returns:
        Result of (∑ᵢ I⊗...⊗Uᵢ⊗...⊗I) @ vector
    """
    if dims is None:
        dims = tuple(U.shape[0] for U in duals)

    if vector.ndim == 1:
        X = vector.reshape(dims)
        result_nd = jnp.zeros_like(X)
        for i, dual in enumerate(duals):
            X_perm = jnp.moveaxis(X, i, -1)
            result_perm = jnp.einsum("...m,mn->...n", X_perm, dual, optimize='optimal')
            result_nd += jnp.moveaxis(result_perm, -1, i)
        return result_nd.reshape(-1)

    else:  # Batch mode (n, k)
        X = vector.reshape(dims + (vector.shape[-1],))
        result_nd = jnp.zeros_like(X)
        for i, dual in enumerate(duals):
            X_perm = jnp.moveaxis(X, i, -2)
            result_perm = jnp.einsum("...mk,mn->...nk", X_perm, dual, optimize='optimal')
            result_nd += jnp.moveaxis(result_perm, -2, i)
        return result_nd.reshape(vector.shape)
