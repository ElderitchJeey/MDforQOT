import jax
import jax.numpy as jnp
from jax import lax
from functools import partial
import numpy as np

@partial(jax.jit, static_argnums=[1, 2])
def get_ptrace(matrix: jnp.ndarray, dims: tuple[int, ...], qkeep: tuple[int, ...]) -> jnp.ndarray:
    """
    Calculates the partial trace of a matrix using JAX library.
     Args:
        matrix: A 2D `jnp.ndarray` representing the input matrix.
        dims: A tuple of integers representing the dimensions of the system.
        qkeep: A tuple of integers representing the indexes of dimensions to keep.
     Returns:
        A `jnp.ndarray` representing the result of the partial trace.
     Description:
        This function calculates the partial trace of a matrix by keeping specific particles (dimensions) and tracing
        over the remaining particles. It is a JAX version of the Qutip function ptrace.pyx.
         If the indices cover all dimensions, the function returns the original matrix. Otherwise, it reshapes the input matrix
        and calculates the trace by flattening the tensor along the first two indices and reshaping it as a square matrix.
        The tensor is then transposed so that the dimensions to trace over come first, followed by the dimensions to keep.
        Finally, the matrix is reshaped as a tensor of rank 4, where the first two indices are the dimensions to trace over,
        and the last two indices are the dimensions to keep. The trace of the resulting square matrix is calculated and returned.
         For more information, refer to the Qutip function ptrace.pyx: https://github.com/qutip/qutip/blob/master/qutip/core/data/ptrace.pyx#L145
    """

    # If the indices cover all dimensions, return the original matrix
    if len(dims) == len(qkeep):
        return matrix.copy()

    # Get the number of dimensions and the dimensions to keep and trace over
    nd = len(dims)  # Number of dimensions
    dkeep = []  # Dimensions to keep
    for i in qkeep:
        dkeep.append(dims[i])
    qtrace = tuple(i for i in range(nd) if i not in qkeep)  # Particles (indexes of dimensions) to trace
    dtrace = []  # Dimensions to trace
    for i in qtrace:
        dtrace.append(dims[i])

    # Reshape the input matrix and calculate the trace:
    # Flatten the tensor along the first two indices, and then reshape it as a square matrix.

    # Transpose the tensor so that the dimensions to trace over come first, followed by the
    # dimensions to keep.

    # Reshape the matrix as a tensor of rank 4, where the first two indices are the dimensions
    # to trace over, and the last two indices are the dimensions to keep.

    # Calculate the trace of the resulting square matrix.
    aux_shape = dims + dims
    transpose_map = qtrace + tuple(el + nd for el in qtrace) + qkeep + tuple(el + nd for el in qkeep)

    keep_shape = 1
    for i in qkeep:
        keep_shape *= dims[i]
    trace_shape = 1
    for i in qtrace:
        trace_shape *= dims[i]
    new_shape = (trace_shape, trace_shape, keep_shape, keep_shape)
    return jnp.trace(matrix
                     .reshape(aux_shape)
                     .transpose(transpose_map)
                     .reshape(new_shape)
                     )


@partial(jax.jit, static_argnums=[1, 2])
def get_ptrace_pure(v: jnp.ndarray, dims: tuple[int, ...], qkeep: tuple[int, ...]) -> jnp.ndarray:
    """
    Computes the partial trace for a pure state vector, i.e. for a density matrix
    of the form |v><v|, without ever constructing the full density matrix.

    Args:
        v: A 1D jnp.ndarray representing the state vector |v⟩.
        dims: A tuple of integers giving the Hilbert space dimensions of the subsystems.
              The product of these dimensions must equal the length of v.
        qkeep: A tuple of integers specifying the indices of subsystems to keep.
               The remaining subsystems will be traced out.

    Returns:
        A jnp.ndarray representing the reduced density matrix on the subsystems specified by qkeep.
        The output is returned as a matrix of shape
            (∏[dims[i] for i in qkeep], ∏[dims[i] for i in qkeep]).

    Raises:
        ValueError: If the product of dims does not match the size of v, or if qkeep is not a valid subset.

    Examples:
        >>> # For a bipartite state |v⟩ in H_A ⊗ H_B, to trace out subsystem B:
        >>> reduced = get_ptrace_pure(v, dims=(dA, dB), qkeep=(0,))
    """
    # Check that the vector size matches the total Hilbert space dimension.
    total_dim = np.prod(dims)
    if v.size != total_dim:
        raise ValueError(f"Product of dims {total_dim} does not match the size of the state vector {v.size}")

    nd = len(dims)
    # Ensure that the kept subsystem indices are in sorted order.
    qkeep = tuple(sorted(qkeep))
    if any(i < 0 or i >= nd for i in qkeep):
        raise ValueError("qkeep contains an index outside the allowed range")

    # If all subsystems are kept, simply return the full density matrix.
    if len(qkeep) == nd:
        return jnp.outer(v, jnp.conjugate(v))

    # Determine which subsystems to trace out.
    qtrace = tuple(i for i in range(nd) if i not in qkeep)

    # Reshape the vector into a tensor with shape equal to dims.
    v_tensor = v.reshape(dims)

    # Compute the reduced density matrix by contracting (i.e. tracing) over the qtrace axes.
    #
    # The tensordot below contracts the axes corresponding to qtrace in both
    # v_tensor and its conjugate, leaving the indices for the kept subsystems free.
    reduced_tensor = jnp.tensordot(v_tensor, jnp.conjugate(v_tensor), axes=(qtrace, qtrace))

    # After contraction, reduced_tensor has shape:
    #    (∏[dims[i] for i in qkeep]) × (∏[dims[i] for i in qkeep])
    # (more precisely, it has 2*len(qkeep) indices, the first len(qkeep) coming from v_tensor,
    #  and the second from the conjugate; we flatten these to a matrix.)
    dkeep = int(np.prod([dims[i] for i in qkeep]))
    reduced_matrix = reduced_tensor.reshape((dkeep, dkeep))

    return reduced_matrix


@partial(jax.jit, static_argnums=(1, 2))
def get_all_ptraces(
        matrix: jnp.ndarray,
        dims: tuple[int, ...],
        system_parts: tuple[tuple[int, ...], ...]
) -> tuple[jnp.ndarray]:
    return jax.tree_util.tree_map(lambda sys_part: get_ptrace(matrix, dims, sys_part), system_parts, is_leaf=lambda t: type(t[0]) == int)


@partial(jax.jit, static_argnums=(1, 2))
def get_all_ptraces_pure(
        vector: jnp.ndarray,
        dims: tuple[int, ...],
        system_parts: tuple[tuple[int, ...], ...]
) -> tuple[jnp.ndarray]:
    return jax.tree_util.tree_map(lambda sys_part: get_ptrace_pure(vector, dims, sys_part), system_parts, is_leaf=lambda t: type(t[0]) == int)


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