# src/linalg.py

from __future__ import annotations

import numpy as np
import numpy.linalg as npl


def hermitianize(A: np.ndarray) -> np.ndarray:
    """Return the Hermitian part (A + A^*) / 2."""
    return (A + A.conj().T) / 2


def proj_to_density(A: np.ndarray, jitter: float = 0.0) -> np.ndarray:
    """
    Project a Hermitian matrix A to a density operator:
    - Hermitianize
    - Clip eigenvalues below 'jitter'
    - Reconstruct and normalize trace to 1
    """
    A = hermitianize(A)
    w, V = npl.eigh(A)
    w = np.maximum(w, jitter)
    rho = (V * w) @ V.conj().T
    tr = float(np.trace(rho).real)
    if tr <= 0:
        raise ValueError("Non-positive trace after projection.")
    return hermitianize(rho / tr)


def herm_exp(H: np.ndarray) -> np.ndarray:
    """
    exp(H) for Hermitian H via eigendecomposition.
    Stabilized by shifting by max eigenvalue (cancels as a scalar factor).
    Note: this returns exp(H - maxEig(H) I), i.e. scaled exp(H).
    This is often intended for numerical stability; if you need exact exp(H),
    do not shift (or multiply back by exp(maxEig)).
    """
    H = hermitianize(H)
    w, V = npl.eigh(H)
    w_shift = w - np.max(w)
    ew = np.exp(w_shift)
    X = (V * ew) @ V.conj().T
    return hermitianize(X)


def herm_log(rho: np.ndarray, jitter: float = 1e-12) -> np.ndarray:
    """log(rho) for PSD Hermitian rho via eigendecomposition, with eigenvalue floor."""
    rho = hermitianize(rho)
    w, V = npl.eigh(rho)
    w = np.maximum(w, jitter)
    lw = np.log(w)
    X = (V * lw) @ V.conj().T
    return hermitianize(X)


def quantum_KL(rho: np.ndarray, sigma: np.ndarray, jitter: float = 1e-12) -> float:
    """KL(rho||sigma) = Tr rho (log rho - log sigma) for density operators."""
    lr = herm_log(rho, jitter=jitter)
    ls = herm_log(sigma, jitter=jitter)
    return float(np.trace(rho @ (lr - ls)).real)


def trace_norm(A: np.ndarray) -> float:
    """Trace norm ||A||_1. For Hermitian A, equals sum |eigenvalues|."""
    A = hermitianize(A)
    w = npl.eigvalsh(A)
    return float(np.sum(np.abs(w)))
