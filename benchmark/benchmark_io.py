# benchmark/benchmark_io.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from src.instances import (
    gen_H_random,
    gen_H_commuting,
    gen_H_conjugated_from_diagonal,
    gen_marginal,
)


def make_sample_id(
    N: int,
    d: int,
    log_eps: float,
    seed: int,
    H_type: str = "random",
    gamma_kind: str = "medium",
) -> str:
    raw = f"N{N}_d{d}_logeps{log_eps:+.2f}_seed{seed}_{H_type}_{gamma_kind}"
    return raw.replace("+", "p").replace("-", "m").replace(".", "p")


def save_eqot_instance(
    out_dir: str | Path,
    *,
    N: int,
    d: int,
    log_eps: float,
    seed: int,
    H_type: str = "random",
    gamma_kind: str = "medium",
    H_scale: float = 1.0,
    hard_delta: float = 1e-4,
) -> Dict[str, Any]:
    """
    Generate and save one package-independent EQOT benchmark instance.

    Saves:
        H, gammas, eps, log_eps, dims, N, d, seed, and metadata.

    Returns:
        A row dictionary suitable for samples.csv.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(seed))
    dims: List[int] = [int(d)] * int(N)
    eps = float(np.exp(float(log_eps)))

    if H_type == "random":
        H = gen_H_random(dims, rng, scale=H_scale)
    elif H_type == "commuting":
        H = gen_H_commuting(dims, rng, scale=H_scale)
    elif H_type == "conjugated_diagonal":
        H = gen_H_conjugated_from_diagonal(dims, rng, scale=H_scale)
    else:
        raise ValueError(f"Unknown H_type={H_type!r}.")

    gammas = [
        gen_marginal(
            int(d),
            rng,
            kind=gamma_kind,
            hard_delta=hard_delta,
        )
        for _ in range(int(N))
    ]

    sample_id = make_sample_id(
        N=int(N),
        d=int(d),
        log_eps=float(log_eps),
        seed=int(seed),
        H_type=H_type,
        gamma_kind=gamma_kind,
    )

    path = out_dir / f"{sample_id}.npz"

    np.savez_compressed(
        path,
        format_version=np.array("eqot_benchmark_v1"),
        sample_id=np.array(sample_id),
        seed=np.array(int(seed), dtype=np.int64),
        N=np.array(int(N), dtype=np.int64),
        d=np.array(int(d), dtype=np.int64),
        dims=np.asarray(dims, dtype=np.int64),
        eps=np.array(eps, dtype=np.float64),
        log_eps=np.array(float(log_eps), dtype=np.float64),
        H=np.asarray(H, dtype=np.complex128),
        gammas=np.stack(gammas, axis=0).astype(np.complex128),
        H_type=np.array(H_type),
        gamma_kind=np.array(gamma_kind),
        H_scale=np.array(float(H_scale), dtype=np.float64),
        hard_delta=np.array(float(hard_delta), dtype=np.float64),
        basis_order=np.array("np.kron_left_to_right"),
        problem_convention=np.array("normalized_gibbs_trace_one"),
    )

    return {
        "sample_id": sample_id,
        "seed": int(seed),
        "N": int(N),
        "d": int(d),
        "log_eps": float(log_eps),
        "eps": eps,
        "H_type": H_type,
        "gamma_kind": gamma_kind,
        "instance_file": path.as_posix(),
    }


def load_eqot_instance(path: str | Path) -> Dict[str, Any]:
    """
    Load one EQOT benchmark instance saved by save_eqot_instance.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Instance file not found: {path}")

    data = np.load(path, allow_pickle=False)

    H = np.asarray(data["H"], dtype=np.complex128)
    gammas_arr = np.asarray(data["gammas"], dtype=np.complex128)
    gammas = [gammas_arr[i] for i in range(gammas_arr.shape[0])]

    dims = [int(x) for x in np.asarray(data["dims"])]

    return {
        "format_version": str(data["format_version"]),
        "sample_id": str(data["sample_id"]),
        "seed": int(data["seed"]),
        "N": int(data["N"]),
        "d": int(data["d"]),
        "dims": dims,
        "eps": float(data["eps"]),
        "log_eps": float(data["log_eps"]),
        "H": H,
        "gammas": gammas,
        "H_type": str(data["H_type"]),
        "gamma_kind": str(data["gamma_kind"]),
        "basis_order": str(data["basis_order"]),
        "problem_convention": str(data["problem_convention"]),
    }