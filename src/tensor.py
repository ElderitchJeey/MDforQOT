# src/tensor.py

from __future__ import annotations

from typing import List
import string
import numpy as np


from src.linalg import hermitianize


def dims_prod(dims: List[int]) -> int:
    p = 1
    for d in dims:
        p *= int(d)
    return int(p)


def partial_trace_except_i(rho: np.ndarray, dims: List[int], i: int) -> np.ndarray:
    """
    Return Tr_{≠ i}(rho), i.e., the i-th marginal of rho on subsystem i.

    rho: (D, D) operator on ⊗_k C^{dims[k]}
    dims: list of local dimensions (length N)
    i: subsystem index to keep, 0 <= i < N
    """
    dims = [int(d) for d in dims]
    N = len(dims)
    if not (0 <= i < N):
        raise IndexError(f"i must be in [0, {N-1}], got {i}.")

    D = int(np.prod(dims))
    if rho.shape != (D, D):
        raise ValueError(f"rho must have shape {(D, D)}, got {rho.shape}.")

    # reshape into 2N-tensor: (d0,...,d_{N-1}, d0,...,d_{N-1})
    T = rho.reshape(*dims, *dims)

    labels = list(string.ascii_letters)  # 52 labels
    if 2 * N > len(labels):
        raise ValueError("N too large for einsum labels; extend label set.")

    in_labels = labels[: 2 * N]

    # Contract traced subsystems j != i by equating bra/ket labels.
    for j in range(N):
        if j != i:
            in_labels[N + j] = in_labels[j]

    # Output keeps only subsystem i: (bra_i, ket_i)
    out_labels = [labels[i], labels[N + i]]
    eins = "".join(in_labels) + "->" + "".join(out_labels)

    rho_i = np.einsum(eins, T).reshape(dims[i], dims[i])
    return hermitianize(rho_i)


def Q_i_lift(Ui: np.ndarray, dims: List[int], i: int) -> np.ndarray:
    """Lift Ui to full space: I ⊗ ... ⊗ Ui ⊗ ... ⊗ I (dense Kronecker)."""
    dims = [int(d) for d in dims]
    if Ui.shape != (dims[i], dims[i]):
        raise ValueError(f"Ui must have shape {(dims[i], dims[i])}, got {Ui.shape}.")

    out = np.array([[1.0]], dtype=complex)
    for j, dj in enumerate(dims):
        if j == i:
            out = np.kron(out, Ui)
        else:
            out = np.kron(out, np.eye(dj, dtype=complex))
    return hermitianize(out)


def L_of_U(U_list: List[np.ndarray], dims: List[int]) -> np.ndarray:
    """Kronecker-sum lift L(U) = sum_i Q_i(U_i)."""
    dims = [int(d) for d in dims]
    D = dims_prod(dims)
    L = np.zeros((D, D), dtype=complex)
    if len(U_list) != len(dims):
        raise ValueError(f"U_list length {len(U_list)} must match len(dims) {len(dims)}.")

    for i, Ui in enumerate(U_list):
        L += Q_i_lift(Ui, dims, i)
    return hermitianize(L)
