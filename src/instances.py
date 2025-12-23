# src/instances.py

from __future__ import annotations

from typing import List
import numpy as np

from src.linalg import hermitianize, proj_to_density
from src.tensor import dims_prod, Q_i_lift


def random_unitary(D: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a (approximately) Haar-random unitary via complex Ginibre + QR."""
    A = (rng.normal(size=(D, D)) + 1j * rng.normal(size=(D, D))) / np.sqrt(2.0)
    Q, R = np.linalg.qr(A)
    diag = np.diag(R)
    phases = diag / (np.abs(diag) + 1e-12)
    return Q * phases.conj()


def gen_H_random(dims: List[int], rng: np.random.Generator, scale: float = 1.0) -> np.ndarray:
    """Random Hermitian Hamiltonian with operator-norm roughly scaled to 'scale'."""
    D = dims_prod(dims)
    A = (rng.normal(size=(D, D)) + 1j * rng.normal(size=(D, D))) / np.sqrt(2.0)
    H = hermitianize(A)
    w = float(np.max(np.abs(np.linalg.eigvalsh(H))))
    if w > 0:
        H = (scale / w) * H
    return hermitianize(H)


def gen_H_commuting(dims: List[int], rng: np.random.Generator, scale: float = 1.0) -> np.ndarray:
    """Diagonal (commuting) Hamiltonian with entries scaled to 'scale'."""
    D = dims_prod(dims)
    diag = rng.normal(size=D)
    diag = diag - np.mean(diag)
    diag = diag / (np.max(np.abs(diag)) + 1e-12) * scale
    return np.diag(diag).astype(complex)


def gen_H_conjugated_from_diagonal(dims: List[int], rng: np.random.Generator, scale: float = 1.0) -> np.ndarray:
    """Hamiltonian unitarily conjugated from a diagonal one: H = U H0 U^*."""
    H0 = gen_H_commuting(dims, rng, scale=scale)
    D = H0.shape[0]
    U = random_unitary(D, rng)
    return hermitianize(U @ H0 @ U.conj().T)

def gen_marginal(
    d: int,
    rng: np.random.Generator,
    kind: str = "medium",
    hard_delta: float = 1e-4
) -> np.ndarray:
    """
    Generate a full-rank density operator on C^d.

    kind:
      easy   : near maximally mixed
      medium : Wishart-like random state
      hard   : ill-conditioned spectrum but still full-rank
    """
    if kind == "easy":
        v = rng.normal(size=(d,)) + 1j * rng.normal(size=(d,))
        v = v / (np.linalg.norm(v) + 1e-12)
        rho = 0.9 * np.eye(d, dtype=complex) / d + 0.1 * np.outer(v, v.conj())
        return proj_to_density(rho, jitter=1e-12)

    if kind == "medium":
        G = (rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))) / np.sqrt(2.0)
        rho = G @ G.conj().T
        return proj_to_density(rho, jitter=1e-12)

    if kind == "hard":
        delta = float(hard_delta)
        if d <= 1:
            raise ValueError(f"'hard' kind requires d >= 2, got d={d}.")
        if not (0.0 < delta < 1.0 / (d - 1)):
            raise ValueError(
                f"Invalid hard_delta={delta} for d={d}. "
                f"Require 0 < hard_delta < 1/(d-1) = {1.0/(d-1):.6g}."
            )

        lam = np.array([1 - (d - 1) * delta] + [delta] * (d - 1), dtype=float)
        U = random_unitary(d, rng)
        rho = U @ np.diag(lam) @ U.conj().T
        return proj_to_density(rho, jitter=1e-12)

# ---- Qubit Ising-type Hamiltonians (d=2) ----

PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)


def embed_two_body(opA: np.ndarray, opB: np.ndarray, dims: List[int], i: int, j: int) -> np.ndarray:
    """Dense embedding of a two-body operator opA (site i) ⊗ opB (site j) into full space."""
    out = np.array([[1.0]], dtype=complex)
    for k, dk in enumerate(dims):
        if k == i:
            out = np.kron(out, opA)
        elif k == j:
            out = np.kron(out, opB)
        else:
            out = np.kron(out, np.eye(int(dk), dtype=complex))
    return hermitianize(out)


def gen_H_ising_qubits(
    N: int,
    rng: np.random.Generator,
    scale: float = 1.0,
    noncommuting: bool = True
) -> np.ndarray:
    """
    1D Ising-type qubit Hamiltonian on N sites:
      sum_i h_i Z_i + sum_i J_i Z_i Z_{i+1} + (optional) sum_i K_i X_i X_{i+1}
    The last term makes it generally non-commuting.
    """
    if N < 2:
        raise ValueError(f"gen_H_ising_qubits requires N >= 2, got N={N}.")

    dims = [2] * N
    D = 2 ** N
    H = np.zeros((D, D), dtype=complex)

    h = rng.normal(size=N)
    J = rng.normal(size=N - 1)
    K = rng.normal(size=N - 1) if noncommuting else np.zeros(N - 1)

    for i in range(N):
        H += h[i] * Q_i_lift(PAULI_Z, dims, i)

    for i in range(N - 1):
        H += J[i] * embed_two_body(PAULI_Z, PAULI_Z, dims, i, i + 1)
        if noncommuting:
            H += K[i] * embed_two_body(PAULI_X, PAULI_X, dims, i, i + 1)

    w = float(np.max(np.abs(np.linalg.eigvalsh(hermitianize(H)))))
    if w > 0:
        H = (scale / w) * H
    return hermitianize(H)

