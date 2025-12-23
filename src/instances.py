# src/instances.py

from __future__ import annotations

from typing import List
import numpy as np

from src.linalg import hermitianize, proj_to_density

from src.tensor import dims_prod


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

