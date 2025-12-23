# experiments/run_sanity.py
#
# Quick sanity checks for the three algorithms:
#  (1) potential_marginal_kl_descent
#  (2) block_gradient_ascent (BGDA baseline)
#  (3) md_type_sinkhorn_potential (MD-Sinkhorn)
#
# Usage (from repo root):
#   python -m experiments.run_sanity
#   python -m experiments.run_sanity --dims 2 2 --eps 0.5 --tol 1e-8 --T 80
#
# Notes:
# - This script generates a small random Hermitian H and full-rank marginals gammas.
# - It verifies basic invariants: pi is Hermitian, trace~1, PSD-ish, marginals trace~1.
# - It prints final errors and gibbs call counts to help debug performance/correctness.

import argparse
import numpy as np
import numpy.linalg as npl

from src.linalg import hermitianize, proj_to_density
from src.tensor import dims_prod, partial_trace_except_i

# Adjust this import to your actual module filename in src/
# If you named it differently, change `SolverofEQOT` accordingly.
from src.SolverofEQOT import (
    potential_marginal_kl_descent,
    block_gradient_ascent,
    md_type_sinkhorn_potential,
)


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def gen_random_hermitian(D: int, rng: np.random.Generator, scale: float = 1.0) -> np.ndarray:
    A = (rng.normal(size=(D, D)) + 1j * rng.normal(size=(D, D))) / np.sqrt(2.0)
    H = hermitianize(A)
    wmax = float(np.max(np.abs(npl.eigvalsh(H))))
    if wmax > 0:
        H = (scale / wmax) * H
    return hermitianize(H)


def gen_random_density(d: int, rng: np.random.Generator, jitter: float = 1e-12) -> np.ndarray:
    G = (rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))) / np.sqrt(2.0)
    rho = G @ G.conj().T
    return proj_to_density(rho, jitter=jitter)


def check_density(pi: np.ndarray, name: str, atol_trace: float = 1e-8, atol_herm: float = 1e-8) -> None:
    # Hermitian
    herm_err = float(npl.norm(pi - pi.conj().T))
    if herm_err > atol_herm:
        raise AssertionError(f"{name}: not Hermitian enough, ||pi-pi*||_F={herm_err:.3e}")

    # Trace
    tr = float(np.real(np.trace(pi)))
    if abs(tr - 1.0) > atol_trace:
        raise AssertionError(f"{name}: trace not ~1, Tr={tr:.12f}")

    # PSD-ish (allow tiny negative due to numerics)
    lam_min = float(np.min(npl.eigvalsh(hermitianize(pi))))
    if lam_min < -1e-7:
        raise AssertionError(f"{name}: not PSD-ish, min eig={lam_min:.3e}")


def check_marginals_trace(pi: np.ndarray, dims: list[int], name: str, atol_trace: float = 1e-8) -> None:
    for i, d in enumerate(dims):
        rho_i = partial_trace_except_i(pi, dims, i)
        tr_i = float(np.real(np.trace(rho_i)))
        if abs(tr_i - 1.0) > atol_trace:
            raise AssertionError(f"{name}: marginal {i} trace not ~1, Tr={tr_i:.12f} (d={d})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", nargs="+", type=int, default=[2, 2], help="Local dimensions, e.g., --dims 2 2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eps", type=float, default=0.5)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--T", type=int, default=80, help="Iteration budget (used for all methods)")
    ap.add_argument("--eta_bgda", type=float, default=1.0, help="BGDA step size")
    ap.add_argument("--eta_kl", type=float, default=None, help="KL-descent step size (None => default 1/N)")
    ap.add_argument("--no_project", action="store_true", help="Disable proj_to_density inside Gibbs mapping (if supported)")
    args = ap.parse_args()

    dims = [int(x) for x in args.dims]
    N = len(dims)
    D = dims_prod(dims)
    rng = _rng(args.seed)

    print("=" * 72)
    print("Sanity run")
    print(f"dims={dims}, D={D}, seed={args.seed}, eps={args.eps}, tol={args.tol}, T={args.T}")
    print("=" * 72)

    # Instance
    H = gen_random_hermitian(D, rng, scale=1.0)
    gammas = [gen_random_density(d, rng, jitter=1e-12) for d in dims]

    # Whether algorithms accept project_pi argument depends on your implementation.
    # We attempt to pass it; if your functions don't support it, just remove it here.
    project_pi = (not args.no_project)

    # --- (1) Potential marginal KL descent
    print("\n[1] potential_marginal_kl_descent")
    res_kl = potential_marginal_kl_descent(
        H=H,
        gammas=gammas,
        eps=float(args.eps),
        dims=dims,
        T=int(args.T),
        eta=(None if args.eta_kl is None else float(args.eta_kl)),
        tol_tr=float(args.tol),
        store_hist=False,
        project_pi=project_pi,
    )
    check_density(res_kl.pi, "KL.pi")
    check_marginals_trace(res_kl.pi, dims, "KL.pi")
    print(f"  converged={res_kl.converged}")
    print(f"  e_tr: init={res_kl.e_tr_list[0]:.3e}  final={res_kl.e_tr_list[-1]:.3e}")
    print(f"  F:    init={res_kl.F_list[0]:.3e}  final={res_kl.F_list[-1]:.3e}")
    print(f"  gibbs_calls={res_kl.gibbs_calls}")

    # --- (2) BGDA baseline
    print("\n[2] block_gradient_ascent (BGDA)")
    res_bgda = block_gradient_ascent(
        H=H,
        gammas=gammas,
        eps=float(args.eps),
        dims=dims,
        T=int(args.T),
        eta=float(args.eta_bgda),
        tol_tr=float(args.tol),
        gauge_trace0=True,
        store_hist=False,
        project_pi=project_pi,
    )
    check_density(res_bgda.pi, "BGDA.pi")
    check_marginals_trace(res_bgda.pi, dims, "BGDA.pi")
    print(f"  converged={res_bgda.converged}")
    print(f"  e_tr: init={res_bgda.e_tr_list[0]:.3e}  final={res_bgda.e_tr_list[-1]:.3e}")
    print(f"  F:    init={res_bgda.F_list[0]:.3e}  final={res_bgda.F_list[-1]:.3e}")
    print(f"  gibbs_calls={res_bgda.gibbs_calls}")

    # --- (3) MD-Sinkhorn
    print("\n[3] md_type_sinkhorn_potential (MD-Sinkhorn)")
    res_mds = md_type_sinkhorn_potential(
        H=H,
        gammas=gammas,
        eps=float(args.eps),
        dims=dims,
        T_outer=int(args.T),
        tol_tr=float(args.tol),
        M_inner=1,
        eta_inner=1.0,
        keep_U_hist=False,
        keep_pi_hist=False,
        tol_inner=float(args.tol),
        project_pi=project_pi,
    )
    check_density(res_mds.pi, "MDS.pi")
    check_marginals_trace(res_mds.pi, dims, "MDS.pi")
    print(f"  converged={res_mds.converged}")
    print(f"  e_tr: init={res_mds.e_tr_list[0]:.3e}  final={res_mds.e_tr_list[-1]:.3e}")
    print(f"  F:    init={res_mds.F_list[0]:.3e}  final={res_mds.F_list[-1]:.3e}")
    print(f"  gibbs_calls={res_mds.gibbs_calls}")

    # Basic consistency check: all final errors should be finite
    finals = [res_kl.e_tr_list[-1], res_bgda.e_tr_list[-1], res_mds.e_tr_list[-1]]
    if not all(np.isfinite(finals)):
        raise AssertionError(f"Non-finite final e_tr detected: {finals}")

    print("\nOK: sanity checks passed.")


if __name__ == "__main__":
    main()
