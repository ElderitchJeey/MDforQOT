import jax
from jax import lax
from functools import partial

from .ptraces import get_all_ptraces_pure
from .misc import make_projector


@jax.jit
def mix_vectors(vectors, coeffs):
    """Compute a weighted mixture of projectors constructed from input vectors.

    Given a set of vectors and corresponding coefficients, this function computes:
    Σ_i (coeffs[i] * |v_i⟩⟨v_i|)
    where |v_i⟩⟨v_i| is the projector formed from the i-th vector.

    The computation is performed efficiently using JAX's fori_loop for automatic
    differentiation and JIT compilation compatibility.

    Parameters
    ----------
    vectors : jax.numpy.ndarray
        A 2-D array containing column vectors to be mixed.
        Shape: (dim, n_vectors) where dim is the vector dimension and
        n_vectors is the number of vectors.
    coeffs : jax.numpy.ndarray
        A 1-D array of mixing coefficients corresponding to each vector.
        Shape: (n_vectors,). Must satisfy len(coeffs) == vectors.shape[1].

    Returns
    -------
    jax.numpy.ndarray
        The resulting mixed matrix formed by the weighted sum of projectors.
        Shape: (dim, dim) where dim is the dimension of the input vectors.

    Examples
    --------
    >>> vectors = jnp.array([[1., 0.], [0., 1.]]).T  # Two basis vectors
    >>> coeffs = jnp.array([0.7, 0.3])
    >>> mix_vectors(vectors, coeffs)
    Array([[0.7, 0. ],
           [0. , 0.3]], dtype=float32)

    >>> # Mixing non-orthogonal vectors
    >>> vectors = jnp.array([[1., 1.], [1., -1.]]).T / jnp.sqrt(2)
    >>> coeffs = jnp.array([1.5, -0.5])
    >>> mix_vectors(vectors, coeffs)
    Array([[ 0.5, 1. ],
           [ 1. , 0.5]], dtype=float32)
    """
    def body_fn(i, carry):
        curr_coeff = coeffs[i]
        curr_vector = vectors[:, i]
        curr_projector = make_projector(curr_vector)
        updated_primal = carry + curr_coeff * curr_projector
        return updated_primal

    initial_carry = coeffs[0] * make_projector(vectors[:, 0])
    primal = lax.fori_loop(1, vectors.shape[1], body_fn, initial_carry)
    return primal


@partial(jax.jit, static_argnums=(2, 3))
def mix_vectors_and_return_ptraces(vectors, coeffs, dims, system_parts):
    """Compute weighted mixture of partial traces for a set of pure state vectors.

    For each vector in the input, computes the partial traces over specified subsystems,
    then returns the weighted sum of these partial traces according to the coefficients.

    Mathematically, computes:
    Σ_i (coeffs[i] * Tr_{not system_parts}[|v_i⟩⟨v_i|])
    where |v_i⟩⟨v_i| is the projector from the i-th vector.

    Parameters
    ----------
    vectors : jax.numpy.ndarray
        A 2D array of column vectors representing pure states.
        Shape: (total_dim, n_vectors) where total_dim = product(dims)
    coeffs : jax.numpy.ndarray
        A 1D array of mixing coefficients for each vector.
        Shape: (n_vectors,). Must satisfy len(coeffs) == vectors.shape[1].
    dims : Tuple[int]
        Dimensions of the individual subsystems (static argument).
        For example, (2, 2) for a two-qubit system.
    system_parts : Tuple[int]
        Indices of subsystems to keep after partial trace (static argument).
        For example, (0,) to trace out all but the first subsystem.

    Returns
    -------
    Dict[Tuple[int], jax.numpy.ndarray]
        A dictionary mapping subsystem combinations to their weighted partial traces.
        Each value is a density matrix for the specified subsystems.
        Keys are tuples of the kept subsystem indices.

    Examples
    --------
    >>> # Two-qubit system, mixing |00⟩ and |11⟩
    >>> vectors = jnp.array([[1,0,0,0], [0,0,0,1]]).T  # Shape (4, 2)
    >>> coeffs = jnp.array([0.6, 0.4])
    >>> dims = (2, 2)
    >>> # Get partial traces for first qubit
    >>> result = mix_vectors_and_return_ptraces(vectors, coeffs, dims, (0,))
    >>> result[(0,)]  # Density matrix of first qubit
    Array([[0.6, 0. ],
           [0. , 0.4]], dtype=float32)

    Notes
    -----
    - The function is JIT-compiled with dims and system_parts as static arguments.
    - Uses tree_map for efficient parallel computation across different partial traces.
    - For large systems, memory usage grows with the number of subsystem combinations.
    """
    def body_fn(i, carry):
        curr_coeff = coeffs[i]
        curr_vector = vectors[:, i]
        curr_ptraces = get_all_ptraces_pure(curr_vector, dims, system_parts)
        updated_primal = jax.tree.map(lambda c, p: c + curr_coeff * p, carry, curr_ptraces)
        return updated_primal

    initial_carry = get_all_ptraces_pure(vectors[:, 0], dims, system_parts)
    initial_carry = jax.tree.map(lambda c: coeffs[0] * c, initial_carry)
    primal = lax.fori_loop(1, vectors.shape[1], body_fn, initial_carry)
    return primal