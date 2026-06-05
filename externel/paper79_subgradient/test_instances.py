from utils import get_all_ptraces
from examples import *


subsystems = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
cost_matrix, dims, system_parts = ising_model(subsystems)

# Ground state
proportions = [1.]
evals, evecs = jax.scipy.linalg.eigh(cost_matrix)
gs_matrix = sum(p * (vec[:, None] @ vec[None, :]) for p, vec in zip(proportions, evecs.T[:len(proportions)]))
ptraces = get_all_ptraces(gs_matrix, dims, system_parts)
test1 = (cost_matrix, ptraces, dims, system_parts)

# Exited state
proportions = [0., 1.]
evals, evecs = jax.scipy.linalg.eigh(cost_matrix)
gs_matrix = sum(p * (vec[:, None] @ vec[None, :]) for p, vec in zip(proportions, evecs.T[:len(proportions)]))
ptraces = get_all_ptraces(gs_matrix, dims, system_parts)
test2 = (cost_matrix, ptraces, dims, system_parts)

# Thermal ptraces
ptraces = tuple([qobj_to_jnp(qt.thermal_dm(2 ** s, i)) for i, s in enumerate(subsystems)])
test3 = (cost_matrix, ptraces, dims, system_parts)

# Mixed state
proportions = [.5, .5]
evals, evecs = jax.scipy.linalg.eigh(cost_matrix)
gs_matrix = sum(p * (vec[:, None] @ vec[None, :]) for p, vec in zip(proportions, evecs.T[:len(proportions)]))
ptraces = get_all_ptraces(gs_matrix, dims, system_parts)
test4 = (cost_matrix, ptraces, dims, system_parts)


subsystems = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
cost_matrix, dims, system_parts = generate_qot_example(2, subsystems, jax.random.PRNGKey(0))

# Ground state
proportions = [1.]
evals, evecs = jax.scipy.linalg.eigh(cost_matrix)
gs_matrix = sum(p * (vec[:, None] @ vec[None, :]) for p, vec in zip(proportions, evecs.T[:len(proportions)]))
ptraces = get_all_ptraces(gs_matrix, dims, system_parts)
test5 = (cost_matrix, ptraces, dims, system_parts)

# Exited state
proportions = [0., 1.]
evals, evecs = jax.scipy.linalg.eigh(cost_matrix)
gs_matrix = sum(p * (vec[:, None] @ vec[None, :]) for p, vec in zip(proportions, evecs.T[:len(proportions)]))
ptraces = get_all_ptraces(gs_matrix, dims, system_parts)
test6 = (cost_matrix, ptraces, dims, system_parts)

# Thermal ptraces
ptraces = tuple([qobj_to_jnp(qt.thermal_dm(2 ** s, i)) for i, s in enumerate(subsystems)])
test7 = (cost_matrix, ptraces, dims, system_parts)

# Mixed state
proportions = [.5, .5]
evals, evecs = jax.scipy.linalg.eigh(cost_matrix)
gs_matrix = sum(p * (vec[:, None] @ vec[None, :]) for p, vec in zip(proportions, evecs.T[:len(proportions)]))
ptraces = get_all_ptraces(gs_matrix, dims, system_parts)
test8 = (cost_matrix, ptraces, dims, system_parts)


dimension = 50
cost_matrix, dims, system_parts = quantum_wasserstein(dimension)

# Pure to pure
rho = thermal_gaussian_state(dimension, temperature=0.)
sigma = thermal_gaussian_state(dimension, temperature=0.)
ptraces = (qobj_to_jnp(rho), qobj_to_jnp(sigma))
test9 = (cost_matrix, ptraces, dims, system_parts)

# Pure to mixed
rho = thermal_gaussian_state(dimension, temperature=0.)
sigma = qt.maximally_mixed_dm(dimension)
ptraces = (qobj_to_jnp(rho), qobj_to_jnp(sigma))
test10 = (cost_matrix, ptraces, dims, system_parts)

# Cat states
# Create tensor products of |alpha> and |-alpha>
alpha = 2.
alpha_state = qt.tensor([qt.coherent(dimension, alpha) for _ in range(2)])
minus_alpha_state = qt.tensor([qt.coherent(dimension, -alpha) for _ in range(2)])
# Superposition and normalization
cat_state = (alpha_state + minus_alpha_state).unit()
rho = cat_state.ptrace(0)
sigma = cat_state.ptrace(1)
ptraces = (qobj_to_jnp(rho), qobj_to_jnp(sigma))
test11 = (cost_matrix, ptraces, dims, system_parts)

### GHZ states
# Parameters
N = 2       # number of modes
n = 0       # first Fock number
m = 1       # second Fock number
# Build tensor product states
state_n = qt.tensor([qt.basis(dimension, n) for _ in range(2)])
state_m = qt.tensor([qt.basis(dimension, m) for _ in range(2)])
# GHZ state and normalization
ghz_state = (state_n + state_m).unit()
rho = ghz_state.ptrace(0)
sigma = ghz_state.ptrace(1)
ptraces = (qobj_to_jnp(rho), qobj_to_jnp(sigma))
test12 = (cost_matrix, ptraces, dims, system_parts)


tests = [
    test1, test2, test3, test4,
    test5, test6, test7, test8,
    test9, test10, test11, test12
]
