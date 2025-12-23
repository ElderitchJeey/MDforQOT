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


def herm_expm_scaled(H: np.ndarray) -> np.ndarray:
    """
    Stable expm for Hermitian H via eigendecomposition with spectral shift.

    Returns:
      exp(H - s I) where s = max eigenvalue of H.
      This is exp(H) scaled by a positive scalar exp(-s).

    Use-case:
      - For Gibbs normalization exp(H)/Tr exp(H), this scaling cancels out,
        so this is the numerically preferred form.
    """
    H = hermitianize(H)
    w, V = npl.eigh(H)
    s = float(np.max(w))
    ew = np.exp(w - s)
    X_scaled = (V * ew) @ V.conj().T
    return hermitianize(X_scaled)

def herm_expm(H: np.ndarray) -> np.ndarray:
    """
    True expm(H) for Hermitian H via eigendecomposition.

    This matches the interface where herm_expm_scaled(H) returns only the scaled matrix:
      herm_expm_scaled(H) = exp(H - s I),  s = max eig(H)
    so we recompute s here and rescale back:
      exp(H) = exp(s) * exp(H - s I).
    """
    Hh = hermitianize(H)
    # Recompute the shift (max eigenvalue) to undo the scaling
    s = float(np.max(npl.eigvalsh(Hh)))
    X_scaled = herm_expm_scaled(Hh)
    return hermitianize(np.exp(s) * X_scaled)

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
