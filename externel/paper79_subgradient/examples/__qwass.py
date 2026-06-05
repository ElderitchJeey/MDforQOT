import numpy as np
import qutip as qt
import jax.numpy as jnp


def covariance_matrix_2mode(rho, N):
    """
    Compute the covariance matrix of a 2-mode Gaussian state.

    Parameters:
    - rho (Qobj): 2-mode density matrix (e.g., from tensor product or evolution).
    - N (int): Truncation level (number of Fock states per mode).

    Returns:
    - cov (np.ndarray): 4x4 covariance matrix.
    """
    # Operators for a single mode
    a = qt.destroy(N)
    adag = a.dag()
    x = (a + adag) / np.sqrt(2)
    p = -1j * (a - adag) / np.sqrt(2)

    # Tensor operators for two modes
    x1 = qt.tensor(x, qt.qeye(N))
    p1 = qt.tensor(p, qt.qeye(N))
    x2 = qt.tensor(qt.qeye(N), x)
    p2 = qt.tensor(qt.qeye(N), p)

    # Quadrature vector R = [x1, p1, x2, p2]
    R = [x1, p1, x2, p2]

    # Initialize covariance matrix
    cov = np.zeros((4, 4), dtype=complex)

    # Compute covariance matrix: sigma_{ij} = 1/2 * <{R_i, R_j}> - <R_i><R_j>
    for i in range(4):
        for j in range(4):
            Rij = (R[i] * R[j] + R[j] * R[i]) / 2
            cov[i, j] = qt.expect(Rij, rho) - qt.expect(R[i], rho) * qt.expect(R[j], rho)

    return cov


def covariance_matrix_1mode(rho, N):
    """
    Compute the covariance matrix of a 1-mode Gaussian state.

    Parameters:
    - rho (Qobj): 1-mode density matrix.
    - N (int): Truncation level (number of Fock states).

    Returns:
    - cov (np.ndarray): 2x2 covariance matrix.
    """
    a = qt.destroy(N)
    adag = a.dag()
    x = (a + adag) / np.sqrt(2)
    p = -1j * (a - adag) / np.sqrt(2)

    R = [x, p]  # Quadrature operators: x, p
    cov = np.zeros((2, 2), dtype=np.float64)

    for i in range(2):
        for j in range(2):
            Rij = (R[i] * R[j] + R[j] * R[i]) / 2
            cov[i, j] = qt.expect(Rij, rho) - qt.expect(R[i], rho) * qt.expect(R[j], rho)

    return cov

def qobj_to_jnp(qobj):
    array = qobj.full().real
    return jnp.array(array, dtype=jnp.float64)

# Ensure U and V are Hermitian
def make_hermitian(qobj):
    return (qobj + qobj.trans().conj()) / 2  # Use QuTiP's trans() method for transpose

# Define position and momentum operators
def position_operator(N):
    return qt.position(N)

def momentum_operator(N):
    return qt.momentum(N)

# Define the cost operator C (as in your previous description)
def cost_operator(N):
    X = position_operator(N)
    P = momentum_operator(N)
    I = qt.qeye(N)

    C = 0.5 * ((qt.tensor(X, I) - qt.tensor(I, X)) ** 2 + (qt.tensor(P, I) - qt.tensor(I, P)) ** 2)
    return C

# Generate a thermal Gaussian state (with covariance matrix proportional to the identity)
def thermal_gaussian_state(N, temperature=1.0):
    # Create a thermal state with a specific temperature using thermal_dm
    # Here we create a thermal state with the identity covariance matrix by using the `thermal_dm` function.
    rho = qt.thermal_dm(N, temperature)
    return rho