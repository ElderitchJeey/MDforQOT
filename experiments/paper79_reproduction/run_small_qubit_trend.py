"""Small paper79-like qubit trend experiments.

This script is meant for quick discussion plots/tables before running the full
paper79 benchmark. It creates small qubit instances inspired by paper79:

- Ising Hamiltonian with marginals from the ground state.
- Random Hermitian Hamiltonian with marginals from the ground state.

Then it compares paper79 entropy L-BFGS against KL descent and MD-Sinkhorn
with M_inner in {1, 2, 5}.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from src.instances import gen_H_ising_qubits
from src.experiment_utils import (
    append_checkpoint,
    default_checkpoint_path,
    fieldnames_union,
    parse_csv_floats,
    parse_csv_ints,
    parse_csv_strings,
    reset_checkpoint,
)
from src.linalg import proj_to_density
from src.tensor import partial_trace_except_i

from .run_lbgfs_vs_ours import (
    DEFAULT_M_LIST,
    run_instance_for_eps,
)


def marginals_from_low_energy_state(H: np.ndarray, dims: List[int], which: str) -> List[np.ndarray]:
    """Generate paper79-like single-site marginals from low-energy states."""

    evals, evecs = np.linalg.eigh(H)
    order = np.argsort(evals)
    v0 = evecs[:, order[0]]
    rho0 = np.outer(v0, v0.conj())

    if which == "ground":
        rho = rho0
    elif which == "excited":
        v1 = evecs[:, order[1]]
        rho = np.outer(v1, v1.conj())
    elif which == "mixed":
        v1 = evecs[:, order[1]]
        rho1 = np.outer(v1, v1.conj())
        rho = 0.5 * rho0 + 0.5 * rho1
    else:
        raise ValueError("which must be ground, excited, or mixed")

    return [proj_to_density(partial_trace_except_i(rho, dims, i), jitter=1e-12) for i in range(len(dims))]


def make_small_instance(
    *,
    kind: str,
    N: int,
    seed: int,
    case: str,
) -> Tuple[np.ndarray, List[np.ndarray], List[int], str]:
    rng = np.random.default_rng(seed)
    dims = [2] * int(N)

    if kind == "ising":
        H = gen_H_ising_qubits(N=int(N), rng=rng, scale=1.0, noncommuting=True)
        label = f"small_ising_{case}_N{N}_seed{seed}"
    elif kind == "random":
        D = int(np.prod(dims))
        A = rng.normal(size=(D, D))
        H = 0.5 * (A + A.T)
        op_norm = float(np.max(np.abs(np.linalg.eigvalsh(H))))
        if op_norm > 0:
            H = H / op_norm
        H = H.astype(complex)
        label = f"small_random_{case}_N{N}_seed{seed}"
    else:
        raise ValueError("kind must be ising or random")

    gammas = marginals_from_low_energy_state(H, dims, which=case)
    return H, gammas, dims, label


class SmallInstance:
    def __init__(self, H, gammas, dims, label):
        self.H = H
        self.gammas = tuple(gammas)
        self.dims = tuple(dims)
        self.index = -2
        self.label = label


def run_small_case(args: argparse.Namespace, *, kind: str, seed: int) -> List[Dict[str, Any]]:
    H, gammas, dims, label = make_small_instance(
        kind=kind,
        N=args.N,
        seed=seed,
        case=args.case,
    )

    # Reuse the main comparison function by temporarily passing a tiny-smoke-like
    # custom instance through a small local monkey patch.
    import experiments.paper79_reproduction.run_lbgfs_vs_ours as runner

    old_make = runner.make_tiny_smoke_instance

    def _make_custom_instance(seed: int = 0, d: int = 2, N: int = 2):
        return SmallInstance(H, gammas, dims, label)

    runner.make_tiny_smoke_instance = _make_custom_instance
    try:
        args.tiny_smoke = True
        args.seed = seed
        args.smoke_d = 2
        args.smoke_N = args.N
        rows = run_instance_for_eps(args, experiment="small", index=0, eps=args.eps)
    finally:
        runner.make_tiny_smoke_instance = old_make

    for row in rows:
        row["small_kind"] = kind
        row["small_case"] = args.case
        row["seed"] = seed
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Small qubit trend: L-BFGS vs KL/MD")
    parser.add_argument("--kinds", default="ising,random")
    parser.add_argument("--N", type=int, default=4)
    parser.add_argument("--case", choices=["ground", "excited", "mixed"], default="ground")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--methods", default="lbfgs,kl,md", help="comma list from lbfgs,kl,md")
    parser.add_argument("--M_list", default=",".join(str(x) for x in DEFAULT_M_LIST))
    parser.add_argument("--T_kl", type=int, default=300)
    parser.add_argument("--T_md", type=int, default=300)
    parser.add_argument("--max_gibbs_calls", type=int, default=None)
    parser.add_argument("--eta_kl", type=float, default=None)
    parser.add_argument("--eta_kl_rule", choices=["eps_over_N", "eps"], default="eps_over_N")
    parser.add_argument("--eta_kl_rules", default="eps_over_N,eps")
    parser.add_argument("--lbfgs_max_iter", type=int, default=300)
    parser.add_argument("--lbfgs_tol", type=float, default=1e-6)
    parser.add_argument("--tol_f", type=float, default=1e-8)
    parser.add_argument("--tol_tr", type=float, default=1e-8)
    parser.add_argument("--tol_pi", type=float, default=1e-6)
    parser.add_argument("--tol_f_grid", default="1e-3,1e-4,1e-5")
    parser.add_argument("--tol_tr_grid", default="1e-3,1e-4,1e-5")
    parser.add_argument("--tol_inner", type=float, default=1e-4)
    parser.add_argument("--jitter", type=float, default=1e-10)
    parser.add_argument("--out", type=Path, default=Path("results") / "small_qubit_trend_eps1e-3.csv")
    parser.add_argument("--checkpoint_jsonl", type=Path, default=None)
    parser.add_argument("--no_checkpoint", action="store_true")
    parser.add_argument(
        "--save_final_state",
        action="store_true",
        help="Save final pi and potentials for each successful method as .npz files.",
    )
    parser.add_argument("--state_dir", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.methods = [x.strip().lower() for x in args.methods.split(",") if x.strip()]
    for method in args.methods:
        if method not in {"lbfgs", "kl", "md"}:
            raise ValueError("--methods entries must be lbfgs, kl, or md")
    args.M_list = parse_csv_ints(args.M_list)
    args.eta_kl_rules = parse_csv_strings(args.eta_kl_rules)
    for eta_rule in args.eta_kl_rules:
        if eta_rule not in {"eps_over_N", "eps"}:
            raise ValueError("--eta_kl_rules entries must be eps_over_N or eps")
    args.tol_f_grid = parse_csv_floats(args.tol_f_grid)
    args.tol_tr_grid = parse_csv_floats(args.tol_tr_grid)
    args.same_limit_require_converged = False

    kinds = [x.strip().lower() for x in args.kinds.split(",") if x.strip()]
    seeds = parse_csv_ints(args.seeds)
    checkpoint_path = None if args.no_checkpoint else (args.checkpoint_jsonl or default_checkpoint_path(args.out))
    reset_checkpoint(checkpoint_path)

    rows: List[Dict[str, Any]] = []
    for kind in kinds:
        for seed in seeds:
            print(f"[Run] kind={kind} N={args.N} case={args.case} seed={seed} eps={args.eps}", flush=True)
            batch = run_small_case(args, kind=kind, seed=seed)
            rows.extend(batch)
            append_checkpoint(checkpoint_path, batch)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = fieldnames_union(rows)
    for extra in ["small_kind", "small_case", "seed"]:
        if extra not in fields:
            fields.append(extra)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        import csv

        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {args.out}")
    if checkpoint_path is not None:
        print(f"Wrote partial checkpoint {checkpoint_path}")


if __name__ == "__main__":
    main()
