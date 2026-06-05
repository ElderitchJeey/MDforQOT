import jax
import jax.numpy as jnp


def make_projector(vec):
    """Compute the outer product projector for a given vector.

    Given a vector, this function computes its corresponding projector,
    defined as the outer product of the vector with its own complex conjugate:
    P = |v⟩⟨v| = v ⊗ vᴴ.

    Parameters
    ----------
    vec : jax.numpy.ndarray
        A 1-D array representing an eigenvector (or any vector).
        Shape: (n,), where n is the dimension of the vector space.

    Returns
    -------
    jax.numpy.ndarray
        The resulting projector matrix (Hermitian if `evec` is real).
        Shape: (n, n), where n is the length of the input vector.

    Examples
    --------
    >>> v = jnp.array([1, 0])
    >>> make_projector(v)
    Array([[1., 0.],
           [0., 0.]], dtype=float32)

    >>> v = jnp.array([1j, 1])
    >>> make_projector(v)
    Array([[ 1.+0.j,  0.-1.j],
           [ 0.+1.j,  1.+0.j]], dtype=complex64)
    """
    return jnp.outer(vec, vec.conj())


@jax.jit
def make_herm(matrix: jnp.ndarray) -> jnp.ndarray:
    """
    Creates a Hermitian matrix using JAX library from the given one.
     Args:
        matrix: A `jnp.ndarray` object representing the input matrix.
     Returns:
        A `jnp.ndarray` object representing the Hermitian matrix.
    """

    # Compute the Hermitian matrix by adding the matrix with its conjugate transpose and dividing by 2
    return (matrix + matrix.conj().T) / 2


def make_traceless(matrix: jnp.ndarray) -> jnp.ndarray:
    """
    Returns a traceless version of the input square matrix.

    The function subtracts (trace(matrix) / n) * I from the input matrix,
    where I is the identity matrix and n is the matrix dimension, ensuring
    that the resulting matrix has a trace of zero.

    Parameters:
    -----------
    matrix : jnp.ndarray
        A square matrix of shape (n, n).

    Returns:
    --------
    jnp.ndarray
        A traceless matrix of the same shape and dtype as the input matrix.
    """
    dim = matrix.shape[0]  # Extract matrix dimension
    identity = jnp.eye(dim, dtype=matrix.dtype)  # Create identity matrix of same dtype
    trace_avg = jnp.trace(matrix) / dim  # Compute average trace contribution per diagonal element
    return matrix - trace_avg * identity  # Subtract trace component to make the matrix traceless