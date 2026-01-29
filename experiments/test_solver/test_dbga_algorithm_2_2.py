"""
experiments/test_dbga_algorithm_2_2.py

Smoke test / comparison experiment for Randig–von Renesse Algorithm 2.2 (DBGA)
implemented as `dbga_algorithm_2_2` in src.SolverofEQOT.

We generate a simple 2-marginal instance using src.instances:
  - dims = [d, d]
  - cost/Hamiltonian H from gen_H_random (or other variants)
  - marginals rho, sigma from gen_marginal

Then compare:
  (A) Paper DBGA Algorithm 2.2 (alternating, unnormalised Γ, fixed steps)
  (B) Existing BGDA baseline with tuned eta (fallback to fixed eta if tuning fails)

Metrics are reported through src.metrics so Chapter 5 comparisons remain consistent.
"""

import argparse
import time
import numpy as np

from src.instances import (
    gen_H_random,
    gen_H_commuting,
    gen_H_conjugated_from_diagonal,
    gen_marginal,
)
from src.tensor import partial_trace_except_i
from src.metrics import final_scalar_metrics, same_limit

from src.SolverofEQOT import (
    dbga_algorithm_2_2,
    block_gradient_ascent,
    tune_bgda_eta,
)


def make_two_marginal_instance(
    d: int,
    seed: int,
    *,
    H_kind: str = "random",
    H_scale: float = 1.0,
    marginal_kind: str = "medium",
    hard_delta: float = 1e-4,
):
    """
    Construct (H, gammas, dims) for N=2 using src.instances building blocks.
    """
    rng = np.random.default_rng(seed)
    dims = [int(d), int(d)]

    hk = H_kind.lower().strip()
    if hk == "random":
        H = gen_H_random(dims, rng, scale=H_scale)
    elif hk == "commuting":
        H = gen_H_commuting(dims, rng, scale=H_scale)
    elif hk in {"conjugated", "conjugated_diagonal", "conjugated_from_diagonal"}:
        H = gen_H_conjugated_from_diagonal(dims, rng, scale=H_scale)
    else:
        raise ValueError(f"Unknown H_kind='{H_kind}'. Use: random / commuting / conjugated_from_diagonal")

    rho = gen_marginal(d, rng, kind=marginal_kind, hard_delta=hard_delta)
    sigma = gen_marginal(d, rng, kind=marginal_kind, hard_delta=hard_delta)

    gammas = [rho, sigma]
    return H, gammas, dims


def sanity_checks(res, gammas, dims, tol=1e-10):
    """
    Basic physical sanity checks on the output coupling pi.
    Note: we do NOT attempt a full PSD check (can be numerically delicate),
    but we enforce hermiticity + trace + marginal trace.
    """
    pi = np.asarray(res.pi)

    # Hermitian
    if not np.allclose(pi, pi.conj().T, atol=tol):
        raise AssertionError("pi is not Hermitian")

    # Trace = 1
    tr = float(np.real(np.trace(pi)))
    if abs(tr - 1.0) > tol:
        raise AssertionError(f"Tr(pi) = {tr}, not 1")

    # Marginal traces
    for i in range(2):
        marg = partial_trace_except_i(pi, dims, i)
        tr_m = float(np.real(np.trace(marg)))
        if abs(tr_m - 1.0) > tol:
            raise AssertionError(f"Tr(marginal {i}) = {tr_m}, not 1")


def run_experiment(
    *,
    d: int,
    eps: float,
    T_paper: int,
    T_tune: int,
    delta: float,
    tol_tr: float,
    seed: int,
    H_kind: str,
    H_scale: float,
    marginal_kind: str,
    hard_delta: float,
    gauge_trace0: bool,
):
    print("=" * 72)
    print("DBGA Algorithm 2.2 test experiment (repo-native instance builder)")
    print(f"d={d}, eps={eps}, seed={seed}")
    print(f"H_kind={H_kind}, H_scale={H_scale}, marginal_kind={marginal_kind}, hard_delta={hard_delta}")
    print(f"gauge_trace0={gauge_trace0}")
    print("=" * 72)

    # ------------------------------------------------------------------
    # Generate 2-marginal instance
    # ------------------------------------------------------------------
    H, gammas, dims = make_two_marginal_instance(
        d=d,
        seed=seed,
        H_kind=H_kind,
        H_scale=H_scale,
        marginal_kind=marginal_kind,
        hard_delta=hard_delta,
    )
    print("Instance generated.")

    # ------------------------------------------------------------------
    # Run paper Algorithm 2.2
    # ------------------------------------------------------------------
    print("\n[Paper] Running DBGA Algorithm 2.2 ...")
    t0 = time.time()
    res_paper = dbga_algorithm_2_2(
        H,
        gammas,
        eps,
        dims,
        T=T_paper,
        delta=delta,
        gauge_trace0=gauge_trace0,
    )
    t_paper = time.time() - t0
    print(f"Done. Time: {t_paper:.4f} sec")
    sanity_checks(res_paper, gammas, dims)

    print("\n[Paper] Final metrics:")
    print(final_scalar_metrics(res_paper))

    # ------------------------------------------------------------------
    # Run BGDA baseline (tuned eta), fallback to fixed eta on failure
    # ------------------------------------------------------------------
    print("\n[Baseline] Running tuned BGDA ...")
    try:
        res_bgda = tune_bgda_eta(
            H,
            gammas,
            eps,
            dims,
            T=T_tune,
            tol_tr=tol_tr,
        )
        print("Tuned BGDA succeeded.")
    except Exception as e:
        print("Tuned BGDA failed; falling back to fixed eta=1.0.")
        print("Reason:", repr(e))
        res_bgda = block_gradient_ascent(
            H,
            gammas,
            eps,
            dims,
            T=T_tune,
            eta=1.0,
        )

    sanity_checks(res_bgda, gammas, dims)

    print("\n[Baseline] Final metrics:")
    print(final_scalar_metrics(res_bgda))

    # ------------------------------------------------------------------
    # Compare final couplings (gauge-invariant notion of same limit)
    # ------------------------------------------------------------------
    print("\n[Comparison] same_limit check:")
    print(same_limit(res_paper, res_bgda))

    print("=" * 72)
    print("Experiment finished.")
    print("=" * 72)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test DBGA Algorithm 2.2 against BGDA baseline using repo-native instance generation"
    )
    parser.add_argument("--d", type=int, default=2, help="Local dimension (d x d)")
    parser.add_argument("--eps", type=float, default=1, help="Entropic regularisation epsilon")
    parser.add_argument("--T_paper", type=int, default=100, help="Iterations for paper DBGA Algorithm 2.2")
    parser.add_argument("--T_tune", type=int, default=100, help="Iterations for BGDA baseline")
    parser.add_argument("--delta", type=float, default=1e-6, help="Stopping tolerance for DBGA (max Fro norm of E1/E2)")
    parser.add_argument("--tol_tr", type=float, default=1e-6, help="Trace error tolerance used by BGDA tuning")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    parser.add_argument("--H_kind", type=str, default="random",
                        help="H generator: random / commuting / conjugated_from_diagonal")
    parser.add_argument("--H_scale", type=float, default=1.0, help="Scale for H operator norm (roughly)")
    parser.add_argument("--marginal_kind", type=str, default="medium",
                        help="Marginal generator kind: easy / medium / hard")
    parser.add_argument("--hard_delta", type=float, default=1e-4,
                        help="For marginal_kind=hard, smallest eigenvalue level (see src.instances.gen_marginal)")
    parser.add_argument("--gauge_trace0", action="store_true",
                        help="Enable gauge fix in Algorithm 2.2 (keeps U⊕V invariant, sets Tr(U)=0)")

    args = parser.parse_args()

    run_experiment(
        d=args.d,
        eps=args.eps,
        T_paper=args.T_paper,
        T_tune=args.T_tune,
        delta=args.delta,
        tol_tr=args.tol_tr,
        seed=args.seed,
        H_kind=args.H_kind,
        H_scale=args.H_scale,
        marginal_kind=args.marginal_kind,
        hard_delta=args.hard_delta,
        gauge_trace0=args.gauge_trace0,
    )
