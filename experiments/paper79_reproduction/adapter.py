"""Adapters for paper79 benchmark instances.

The external benchmark stores instances as

    (cost_matrix, ptraces, dims, system_parts)

where ``system_parts`` describes which elementary tensor factors each marginal
keeps. The solvers in ``src.SolverofEQOT`` currently support the common case
where each marginal is a single elementary subsystem, so this adapter validates
that convention and converts JAX arrays to NumPy arrays.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER79_ROOT = REPO_ROOT / "externel" / "paper79_subgradient"


PAPER79_LABELS = {
    0: "IG",
    1: "IE",
    2: "IT",
    3: "IM",
    4: "RG",
    5: "RE",
    6: "RT",
    7: "RM",
    8: "WP",
    9: "WM",
    10: "WC",
    11: "WG",
}


@dataclass(frozen=True)
class Paper79Instance:
    """Repo-native view of one paper79 benchmark instance."""

    index: int
    label: str
    H: np.ndarray
    gammas: Tuple[np.ndarray, ...]
    dims: Tuple[int, ...]
    system_parts: Tuple[Tuple[int, ...], ...]


def ensure_paper79_import_path() -> None:
    path = os.fspath(PAPER79_ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)


def _as_complex_numpy(x) -> np.ndarray:
    return np.asarray(x, dtype=complex)


def validate_single_site_marginals(
    dims: Sequence[int],
    system_parts: Sequence[Sequence[int]],
) -> None:
    """Ensure system_parts matches the current repo-native solver interface."""

    if len(system_parts) != len(dims):
        raise ValueError(
            "Only one marginal per elementary subsystem is supported: "
            f"len(system_parts)={len(system_parts)}, len(dims)={len(dims)}."
        )

    expected = tuple((i,) for i in range(len(dims)))
    got = tuple(tuple(int(j) for j in part) for part in system_parts)
    if got != expected:
        raise ValueError(
            "Current repo-native solvers expect single-site marginals "
            f"{expected}, got {got}."
        )


def load_paper79_instance(index: int) -> Paper79Instance:
    """Load one external benchmark instance and convert it to NumPy arrays."""

    ensure_paper79_import_path()
    from test_instances import tests  # type: ignore

    if not (0 <= int(index) < len(tests)):
        raise IndexError(f"paper79 instance index must be in [0, {len(tests) - 1}].")

    cost_matrix, ptraces, dims, system_parts = tests[int(index)]
    dims_tuple = tuple(int(d) for d in dims)
    parts_tuple = tuple(tuple(int(j) for j in part) for part in system_parts)
    validate_single_site_marginals(dims_tuple, parts_tuple)

    return Paper79Instance(
        index=int(index),
        label=PAPER79_LABELS.get(int(index), f"test{int(index)}"),
        H=_as_complex_numpy(cost_matrix),
        gammas=tuple(_as_complex_numpy(gamma) for gamma in ptraces),
        dims=dims_tuple,
        system_parts=parts_tuple,
    )


def make_tiny_smoke_instance(seed: int = 0, d: int = 2, N: int = 2) -> Paper79Instance:
    """Generate a tiny paper79-shaped instance for local smoke tests."""

    from src.instances import gen_H_random, gen_marginal

    rng = np.random.default_rng(seed)
    dims = tuple([int(d)] * int(N))
    system_parts = tuple((i,) for i in range(int(N)))
    H = gen_H_random(list(dims), rng, scale=1.0)
    gammas = tuple(gen_marginal(int(d), rng, kind="medium") for _ in range(int(N)))
    return Paper79Instance(
        index=-1,
        label="SMOKE",
        H=H,
        gammas=gammas,
        dims=dims,
        system_parts=system_parts,
    )
