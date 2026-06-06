"""Quantum Wasserstein/channel-style benchmarks without QuTiP.

Paper79 uses a Fock-basis discretization for the quantum Wasserstein/channel
examples WP, WM, WC, and WG. This script recreates those instances with NumPy
so the benchmark does not require QuTiP.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from src.linalg import proj_to_density
from src.tensor import partial_trace_except_i

from .run_lbgfs_vs_ours import DEFAULT_M_LIST, fieldnames_union, run_instance_for_eps, write_csv


def annihilation_operator(d: int) -> np.ndarray:
    a = np.zeros((d, d), dtype=complex)
    for n in range(1, d):
        a[n - 1, n] = np.sqrt(n)
    return a


def wasserstein_cost(d: int) -> np.ndarray:
    a = annihilation_operator(d)
    adag = a.conj().T
    X = (a + adag) / np.sqrt(2.0)
    P = -1j * (a - adag) / np.sqrt(2.0)
    I = np.eye(d, dtype=complex)
    DX = np.kron(X, I) - np.kron(I, X)
    DP = np.kron(P, I) - np.kron(I, P)
    H = 0.5 * (DX @ DX + DP @ DP)
    return 0.5 * (H + H.conj().T)


def basis_density(d: int, n: int) -> np.ndarray:
    v = np.zeros(d, dtype=complex)
    v[int(n)] = 1.0
    return np.outer(v, v.conj())


def coherent_vector(d: int, alpha: float) -> np.ndarray:
    coeffs = np.zeros(d, dtype=complex)
    coeffs[0] = np.exp(-0.5 * abs(alpha) ** 2)
    for n in range(1, d):
        coeffs[n] = coeffs[n - 1] * alpha / np.sqrt(n)
    norm = np.linalg.norm(coeffs)
    return coeffs / max(norm, 1e-300)


def cat_marginals(d: int, alpha: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    plus = coherent_vector(d, alpha)
    minus = coherent_vector(d, -alpha)
    psi = np.kron(plus, plus) + np.kron(minus, minus)
    psi = psi / np.linalg.norm(psi)
    rho = np.outer(psi, psi.conj())
    dims = [d, d]
    return partial_trace_except_i(rho, dims, 0), partial_trace_except_i(rho, dims, 1)


def ghz_marginals(d: int) -> Tuple[np.ndarray, np.ndarray]:
    psi = np.zeros(d * d, dtype=complex)
    psi[0 * d + 0] = 1.0
    psi[1 * d + 1] = 1.0
    psi = psi / np.linalg.norm(psi)
    rho = np.outer(psi, psi.conj())
    dims = [d, d]
    return partial_trace_except_i(rho, dims, 0), partial_trace_except_i(rho, dims, 1)


def make_wasserstein_instance(d: int, case: str, jitter: float) -> Tuple[np.ndarray, List[np.ndarray], List[int], str]:
    H = wasserstein_cost(d)
    dims = [int(d), int(d)]
    case = case.upper()

    vacuum = basis_density(d, 0)
    if case == "WP":
        gammas = [vacuum, vacuum]
    elif case == "WM":
        gammas = [vacuum, np.eye(d, dtype=complex) / d]
    elif case == "WC":
        gammas = list(cat_marginals(d))
    elif case == "WG":
        gammas = list(ghz_marginals(d))
    else:
        raise ValueError("case must be one of WP, WM, WC, WG")

    gammas = [proj_to_density(g, jitter=jitter) for g in gammas]
    label = f"wasserstein_{case}_d{d}"
    return H, gammas, dims, label


class WassersteinInstance:
    def __init__(self, H, gammas, dims, label):
        self.H = H
        self.gammas = tuple(gammas)
        self.dims = tuple(dims)
        self.index = -3
        self.label = label


def run_case(args: argparse.Namespace, case: str) -> List[Dict[str, Any]]:
    H, gammas, dims, label = make_wasserstein_instance(args.d, case, args.marginal_jitter)

    import experiments.paper79_reproduction.run_lbgfs_vs_ours as runner

    old_make = runner.make_tiny_smoke_instance

    def _make_custom_instance(seed: int = 0, d: int = 2, N: int = 2):
        return WassersteinInstance(H, gammas, dims, label)

    runner.make_tiny_smoke_instance = _make_custom_instance
    try:
        args.tiny_smoke = True
        args.seed = 0
        args.smoke_d = args.d
        args.smoke_N = 2
        rows = run_instance_for_eps(args, experiment="wasserstein", index=0, eps=args.eps)
    finally:
        runner.make_tiny_smoke_instance = old_make

    for row in rows:
        row["wasserstein_case"] = case.upper()
        row["d"] = int(args.d)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Quantum Wasserstein trend: L-BFGS vs KL/MD")
    parser.add_argument("--d", type=int, default=20)
    parser.add_argument("--cases", default="WP,WM,WC,WG")
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--methods", default="lbfgs,kl,md")
    parser.add_argument("--M_list", default=",".join(str(x) for x in DEFAULT_M_LIST))
    parser.add_argument("--T_kl", type=int, default=300)
    parser.add_argument("--T_md", type=int, default=300)
    parser.add_argument("--eta_kl", type=float, default=None)
    parser.add_argument("--eta_kl_rule", choices=["eps_over_N", "eps"], default="eps_over_N")
    parser.add_argument("--eta_kl_rules", default="eps_over_N,eps")
    parser.add_argument("--lbfgs_max_iter", type=int, default=300)
    parser.add_argument("--lbfgs_tol", type=float, default=1e-6)
    parser.add_argument("--tol_f", type=float, default=1e-8)
    parser.add_argument("--tol_tr", type=float, default=1e-8)
    parser.add_argument("--tol_f_grid", default="1e-3,1e-4,1e-5")
    parser.add_argument("--tol_tr_grid", default="1e-3,1e-4,1e-5")
    parser.add_argument("--tol_pi", type=float, default=1e-6)
    parser.add_argument("--tol_inner", type=float, default=1e-4)
    parser.add_argument("--jitter", type=float, default=1e-10)
    parser.add_argument("--marginal_jitter", type=float, default=1e-12)
    parser.add_argument("--out", type=Path, default=Path("results") / "wasserstein_d20_eps1e-3_summary.csv")
    parser.add_argument("--checkpoint_jsonl", type=Path, default=None)
    parser.add_argument("--no_checkpoint", action="store_true")
    return parser


def main() -> None:
    from .run_lbgfs_vs_ours import (
        append_checkpoint,
        default_checkpoint_path,
        parse_csv_floats,
        parse_csv_ints,
        parse_csv_strings,
        reset_checkpoint,
    )

    args = build_parser().parse_args()
    args.methods = [x.strip().lower() for x in args.methods.split(",") if x.strip()]
    args.M_list = parse_csv_ints(args.M_list)
    args.eta_kl_rules = parse_csv_strings(args.eta_kl_rules)
    for eta_rule in args.eta_kl_rules:
        if eta_rule not in {"eps_over_N", "eps"}:
            raise ValueError("--eta_kl_rules entries must be eps_over_N or eps")
    args.tol_f_grid = parse_csv_floats(args.tol_f_grid)
    args.tol_tr_grid = parse_csv_floats(args.tol_tr_grid)
    args.same_limit_require_converged = False

    cases = [x.strip().upper() for x in args.cases.split(",") if x.strip()]
    checkpoint_path = None if args.no_checkpoint else (args.checkpoint_jsonl or default_checkpoint_path(args.out))
    reset_checkpoint(checkpoint_path)

    rows: List[Dict[str, Any]] = []
    for case in cases:
        print(f"[Run] Wasserstein case={case} d={args.d} eps={args.eps}", flush=True)
        batch = run_case(args, case)
        rows.extend(batch)
        append_checkpoint(checkpoint_path, batch)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = fieldnames_union(rows)
    for extra in ["wasserstein_case", "d"]:
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
