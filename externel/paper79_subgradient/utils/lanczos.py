import jax
import jax.numpy as jnp
from functools import partial
from .dual import kron_sum_mvp


@jax.jit
def get_mvp(cost_matrix, duals, vector):
    v1 = cost_matrix @ vector
    v2 = kron_sum_mvp(duals, vector)

    return v1 - v2


@partial(jax.jit, static_argnums=(3, 4))
def stochastic_lanczos(cost_matrix, duals, initial_vector, max_iter=100, tol=1e-6):
    """
    Approximates the smallest eigenvalue and its eigenvector of a symmetric matrix A
    using the Lanczos algorithm. The function is jittable by avoiding dynamic slicing.

    Instead of slicing with a dynamic index, we compute dot products and projections
    over the entire preallocated arrays and mask out rows beyond the current iteration.

    Parameters:
      initial_vector (jnp.ndarray): starting vector of shape (n,)
      max_iter (int): maximum number of Lanczos iterations (static)
      tol (float): tolerance for the residual norm (static)

    Returns:
      smallest_eigenvalue (float): approximated smallest eigenvalue
      approx_eigenvector (jnp.ndarray): corresponding approximate eigenvector (shape (n,))
    """
    n = initial_vector.shape[0]
    # Pre-allocate arrays for the Lanczos basis and tridiagonal coefficients.
    V = jnp.zeros((max_iter + 1, n))
    # Set the initial vector (normalized).
    V = V.at[0].set(initial_vector / jnp.linalg.norm(initial_vector))
    alphas = jnp.zeros((max_iter,))
    betas = jnp.zeros((max_iter + 1,))

    beta0 = 1.0  # dummy initialization for entering the loop
    i = 0  # iteration counter

    def cond_fun(state):
        i, V, alphas, betas, beta = state
        return jnp.logical_and(i < max_iter, jnp.logical_or(i == 0, beta >= tol))

    def body_fun(state):
        i, V, alphas, betas, beta = state
        v_i = V[i]
        w = get_mvp(cost_matrix, duals, v_i)
        alpha = jnp.dot(v_i, w)
        alphas = alphas.at[i].set(alpha)
        # Subtract the contribution along v_i (and, if i > 0, along the previous Lanczos vector).
        w = jax.lax.cond(
            i == 0,
            lambda w: w - alpha * v_i,
            lambda w: w - alpha * v_i - betas[i] * V[i - 1],
            operand=w,
        )
        # --- Reorthogonalize ---
        # Instead of slicing V[:i+1], compute the dot products for all rows and mask out unused ones.
        full_indices = jnp.arange(max_iter + 1)  # indices 0, 1, ..., max_iter
        # Create a mask that is 1 for valid indices (j < i+1) and 0 for others.
        mask = (full_indices < (i + 1)).astype(w.dtype)  # shape: (max_iter+1,)
        # Compute dot products for each row of V with w (shape: (max_iter+1,)).
        coeffs_full = jnp.einsum('jn,n->j', V, w)
        # Mask out contributions from invalid rows.
        coeffs_valid = coeffs_full * mask
        # Compute the projection using all rows of V weighted by the valid coefficients.
        proj = jnp.sum(coeffs_valid[:, None] * V, axis=0)
        w = w - proj

        beta_new = jnp.linalg.norm(w)
        betas = betas.at[i + 1].set(beta_new)
        # If beta_new is above the tolerance, store the new Lanczos vector.
        V = jax.lax.cond(
            beta_new >= tol,
            lambda V: V.at[i + 1].set(w / beta_new),
            lambda V: V,
            operand=V,
        )
        return (i + 1, V, alphas, betas, beta_new)

    # Run the while loop.
    state_init = (i, V, alphas, betas, beta0)
    i_final, V, alphas, betas, beta_final = jax.lax.while_loop(cond_fun, body_fun, state_init)
    m = i_final  # number of iterations performed (m is dynamic)

    # --- Pad the unused parts of alphas and betas ---
    idx = jnp.arange(max_iter)
    mask_alpha = idx < m  # True for indices < m
    # For padded entries, set alpha to a very large number (so their eigenvalues are huge).
    alphas_full = jnp.where(mask_alpha, alphas, 1e10)

    idx_betas = jnp.arange(max_iter + 1)
    # Set the off-diagonal coupling at the boundary to zero.
    betas_full = jnp.where(idx_betas == m, 0.0, betas)

    # Build the fixed–size tridiagonal matrix.
    T_full = jnp.diag(alphas_full) \
             + jnp.diag(betas_full[1:max_iter], k=1) \
             + jnp.diag(betas_full[1:max_iter], k=-1)

    # Compute the eigen–decomposition.
    eigvals, eigvecs = jnp.linalg.eigh(T_full)
    smallest_eigenvalue = eigvals[0]
    y_full = eigvecs[:, 0]  # eigenvector in the Lanczos subspace (length max_iter)
    # Zero out the padded parts.
    mask_y = (jnp.arange(max_iter) < m).astype(y_full.dtype)
    y_valid = y_full * mask_y

    # Reconstruct the approximate eigenvector in the original space.
    # (V has shape (max_iter+1, n); we use the first max_iter rows.)
    V_reduced = V[:max_iter, :]
    approx_eigenvector = jnp.dot(y_valid, V_reduced)

    return smallest_eigenvalue, approx_eigenvector
