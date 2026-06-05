import jax
import netket as nk

from .__qwass import *
from utils import make_herm


def ising_model(subsystems):
    # Whole system size
    N = sum(subsystems)

    # Hilbert space associated with the system
    hi = nk.hilbert.Spin(1 / 2, N)

    # Hamiltonian of the system
    hamiltonian = nk.operator.Ising(hi, graph=nk.graph.Chain(N), h=1.5)
    cost_matrix = hamiltonian.to_dense()

    # Original dimensions
    dims = tuple(2 for _ in range(N))

    # Define subsystem partitions
    system_parts = []
    start = 0
    for size in subsystems:
        subsys = tuple(range(start, start + size))
        system_parts.append(subsys)
        start += size
    system_parts = tuple(system_parts)

    return cost_matrix, dims, system_parts


def quantum_wasserstein(dim):
    cost_op = cost_operator(dim)

    cost_matrix = qobj_to_jnp(cost_op)

    dims = (dim, dim)
    subsystems = [1, 1]

    # Define subsystem partitions
    system_parts = []
    start = 0
    for size in subsystems:
        subsys = tuple(range(start, start + size))
        system_parts.append(subsys)
        start += size
    system_parts = tuple(system_parts)

    return cost_matrix, dims, system_parts


def generate_qot_example(
        n: int,
        subsystems: tuple[int, ...],
        key: jax.random.PRNGKey,
) -> tuple[jnp.ndarray, tuple[int, ...], tuple[tuple[int, ...], ...]]:
    N = sum(subsystems)
    dims = tuple(n for _ in range(N))

    # Generate random Hermitian cost matrix
    cost_matrix = jax.random.normal(key, shape=(n ** N, n ** N))
    cost_matrix = make_herm(cost_matrix)

    # Define subsystem partitions
    system_parts = []
    start = 0
    for size in subsystems:
        subsys = tuple(range(start, start + size))
        system_parts.append(subsys)
        start += size
    system_parts = tuple(system_parts)

    return cost_matrix, dims, system_parts
