"""
experiments/diagnose_dbga_algorithm_2_2.py

Detailed diagnostic experiment for Randig–von Renesse Algorithm 2.2 (DBGA).

Goal: Determine whether Algorithm 2.2 is:
  (i) not moving due to an implementation bug, or
  (ii) moving but with step sizes so small that progress is imperceptible,
  (iii) converging but slowly.

We do this by:
  - Printing the paper step-size quantities: y, beta, eta1, eta2
  - Running dbga_algorithm_2_2 with store_hist=True
  - Measuring per-iteration changes in U,V and pi
  - Printing summaries and optional baseline BGDA run

This file is designed to fit MDforQOT repo conventions and reuse src.instances/src.metrics.
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
from src.metrics import final_scalar_metrics

from src.SolverofEQOT import (
    dbga_algorithm_2_2,
    block_gradient_ascent,
    tune_bgda_eta,
    gamma_from_potentials,
    _nu2_inv_exp_minus_x_minus_1,  # internal helper used by Algorithm 2.2
    hermitianize,
)


def normalize_cost_operator(H: np.ndarray) -> np.ndarray:
    """
    Shift + scale Hermitian H so that its spectrum is roughly in [0,1].
    This often matches "cost >= 0 with controlled scale" assumptions used in experiments.
    """
    evals = np.linalg.eigvalsh(hermitianize(H))
    lam_min = float(evals.min())
    lam_max = float(evals.max())
    H2 = hermitianize(H - lam_min * np.eye(H.shape[0], dtype=complex))
    denom = max(lam_max - lam_min, 1e-12)
    H2 = hermitianize(H2 / denom)
    return H2


def make_two_marginal_instance(
    d: int,
    seed: int,
    *,
    H_kind: str = "random",
    H_scale: float = 1.0,
    marginal_kind: str = "medium",
    hard_delta: float = 1e-4,
    normalize_cost: bool = False,
):
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

    if normalize_cost:
        H = normalize_cost_operator(H)

    return H, gammas, dims


def trace_distance(A: np.ndarray, B: np.ndarray) -> float:
    """0.5 * ||A-B||_1 (trace norm)."""
    X = hermitianize(A - B)
    s = np.linalg.svd(X, compute_uv=False)
    return 0.5 * float(np.sum(np.abs(s)))


def compute_paper_stepsizes(
    H: np.ndarray,
    gammas,
    eps: float,
    dims,
    U0: np.ndarray,
    V0: np.ndarray,
):
    """
    Recompute Algorithm 2.2 step-size quantities:
      y = (tr((rho⊗sigma)H) - D(U0,V0)) / eps
      beta = nu2^{-1}(y)
      eta1 = (eps/d2)*exp(-beta), eta2 = (eps/d1)*exp(-beta)

    We implement D(U,V) exactly as in our solver implementation:
      D(U,V) = tr(U rho) + tr(V sigma) - eps * tr(Gamma(U,V)) + eps
    with Gamma = exp((U⊕V - H)/eps).
    """
    rho, sigma = gammas
    d1, d2 = int(dims[0]), int(dims[1])

    Gamma0, Z0 = gamma_from_potentials([U0, V0], H, eps, dims)
    # NOTE: Z0 is Tr(Gamma0); in our code gamma_from_potentials returns (Gamma, Z)
    D0 = float(np.real(np.trace(U0 @ rho))) + float(np.real(np.trace(V0 @ sigma))) - eps * float(np.real(Z0)) + eps

    tr_rhosig_H = float(np.real(np.trace(np.kron(rho, sigma) @ H)))
    y = (tr_rhosig_H - D0) / eps
    y = max(float(y), 0.0)

    beta = float(_nu2_inv_exp_minus_x_minus_1(y))
    eta1 = (eps / float(d2)) * np.exp(-beta)
    eta2 = (eps / float(d1)) * np.exp(-beta)
    return y, beta, float(eta1), float(eta2)


def sanity_checks_pi(pi: np.ndarray, gammas, dims, tol=1e-10):
    pi = np.asarray(pi)
    if not np.allclose(pi, pi.conj().T, atol=tol):
        raise AssertionError("pi is not Hermitian")
    tr = float(np.real(np.trace(pi)))
    if abs(tr - 1.0) > tol:
        raise AssertionError(f"Tr(pi) = {tr}, not 1")
    for i in range(2):
        marg = partial_trace_except_i(pi, dims, i)
        tr_m = float(np.real(np.trace(marg)))
        if abs(tr_m - 1.0) > tol:
            raise AssertionError(f"Tr(marginal {i}) = {tr_m}, not 1")


def summarize_progress(res, *, max_lines: int = 12):
    """
    Print a compact progress summary based on e_tr_list and step norms.
    """
    e = np.array(res.e_tr_list, dtype=float)
    F = np.array(res.F_list, dtype=float)

    print("\n[DBGA] Progress summary")
    print(f"  iters={len(e)-1}, converged={res.converged}")
    print(f"  e_tr[0]={e[0]:.6e}, e_tr[end]={e[-1]:.6e}, ratio={e[-1]/max(e[0],1e-300):.3e}")
    print(f"  F[0]={F[0]:.6e}, F[end]={F[-1]:.6e}")

    # show a few samples
    if len(e) > 1:
        idxs = np.unique(np.round(np.linspace(0, len(e)-1, min(max_lines, len(e)))).astype(int))
        print("  samples: k : e_tr[k], F[k]")
        for k in idxs:
            print(f"    {k:5d} : {e[k]:.6e}, {F[k]:.6e}")


def analyze_hist(res, *, metric: str = "trace"):
    """
    Analyze U_hist/pi_list to see whether the algorithm is actually moving.
    """
    if res.U_hist is None or res.pi_list is None:
        print("\n[DBGA] No history stored (store_hist=False). Skipping movement diagnostics.")
        return

    U_hist = res.U_hist
    pi_list = res.pi_list
    K = len(pi_list) - 1
    if K <= 0:
        print("\n[DBGA] History too short.")
        return

    dU = np.zeros(K)
    dV = np.zeros(K)
    dpi = np.zeros(K)

    for k in range(K):
        U0, V0 = U_hist[k]
        U1, V1 = U_hist[k + 1]
        dU[k] = np.linalg.norm(U1 - U0, ord="fro")
        dV[k] = np.linalg.norm(V1 - V0, ord="fro")
        if metric == "trace":
            dpi[k] = trace_distance(pi_list[k + 1], pi_list[k])
        else:
            dpi[k] = np.linalg.norm(pi_list[k + 1] - pi_list[k], ord="fro")

    print("\n[DBGA] Movement diagnostics (per-iter)")
    print(f"  ||U_{'{k+1}'}-U_k||_F: min={dU.min():.3e}, median={np.median(dU):.3e}, max={dU.max():.3e}")
    print(f"  ||V_{'{k+1}'}-V_k||_F: min={dV.min():.3e}, median={np.median(dV):.3e}, max={dV.max():.3e}")
    print(f"  dist(pi_{'{k+1}'} , pi_k): min={dpi.min():.3e}, median={np.median(dpi):.3e}, max={dpi.max():.3e}")

    # detect "not moving" pattern
    if np.max(dU) < 1e-14 and np.max(dV) < 1e-14:
        print("  WARNING: U,V updates are below ~1e-14 (likely step sizes too small or numerical stagnation).")
    if np.max(dpi) < 1e-12:
        print("  WARNING: pi changes are below ~1e-12 (coupling essentially not changing).")


def run_dbga_diagnosis(args):
    print("=" * 88)
    print("DIAGNOSE: DBGA Algorithm 2.2 convergence / movement")
    print(f"d={args.d}, eps={args.eps}, seed={args.seed}, T={args.T}")
    print(f"H_kind={args.H_kind}, H_scale={args.H_scale}, marginal_kind={args.marginal_kind}, hard_delta={args.hard_delta}")
    print(f"normalize_cost={args.normalize_cost}, gauge_trace0={args.gauge_trace0}, project_pi={args.project_pi}")
    print("=" * 88)

    H, gammas, dims = make_two_marginal_instance(
        d=args.d,
        seed=args.seed,
        H_kind=args.H_kind,
        H_scale=args.H_scale,
        marginal_kind=args.marginal_kind,
        hard_delta=args.hard_delta,
        normalize_cost=args.normalize_cost,
    )
    print("Instance generated.")

    d1, d2 = dims
    U0 = np.zeros((d1, d1), dtype=complex)
    V0 = np.zeros((d2, d2), dtype=complex)

    # Step-size diagnostics (paper quantities)
    y, beta, eta1, eta2 = compute_paper_stepsizes(H, gammas, args.eps, dims, U0, V0)
    print("\n[DBGA] Paper step-size diagnostics")
    print(f"  y={y:.6e}, beta={beta:.6e}, eta1={eta1:.6e}, eta2={eta2:.6e}")
    if eta1 < 1e-12 or eta2 < 1e-12:
        print("  WARNING: eta is extremely small (<1e-12). Expect stagnation unless cost is re-scaled.")

    # Run Algorithm 2.2
    print("\n[DBGA] Running Algorithm 2.2 ...")
    t0 = time.time()
    res = dbga_algorithm_2_2(
        H,
        gammas,
        args.eps,
        dims,
        T=args.T,
        tol_tr=args.tol_tr,
        delta=args.delta,
        gauge_trace0=args.gauge_trace0,
        store_hist=True,
        project_pi=args.project_pi,
        jitter=args.jitter,
        U0=U0,
        V0=V0,
    )
    dt = time.time() - t0
    print(f"Done. wall_time={dt:.4f} sec")
    sanity_checks_pi(res.pi, gammas, dims)

    print("\n[DBGA] Final scalar metrics:")
    print(final_scalar_metrics(res))

    summarize_progress(res, max_lines=12)
    analyze_hist(res, metric="trace")

    # Optional baseline BGDA sanity check
    if args.run_bgda:
        print("\n[BGDA] Running baseline BGDA ...")
        try:
            eta_grid = [float(x) for x in args.eta_grid.split(",")]
            res_bgda = tune_bgda_eta(
                H, gammas, args.eps, dims,
                eta_grid=eta_grid,
                T=args.T_bgda,
                tol_tr=args.tol_tr if args.tol_tr is not None else 1e-6,
            )
            print("[BGDA] tuned BGDA succeeded.")
        except Exception as e:
            print("[BGDA] tuned BGDA failed; fallback to eta=1.0")
            print("Reason:", repr(e))
            res_bgda = block_gradient_ascent(
                H, gammas, args.eps, dims,
                T=args.T_bgda,
                eta=1.0,
            )

        print("\n[BGDA] Final scalar metrics:")
        print(final_scalar_metrics(res_bgda))

    print("=" * 88)
    print("Diagnosis finished.")
    print("=" * 88)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Diagnose DBGA Algorithm 2.2 convergence / movement")
    parser.add_argument("--d", type=int, default=3)
    parser.add_argument("--eps", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--T", type=int, default=5000, help="Iterations for DBGA Algorithm 2.2")
    parser.add_argument("--tol_tr", type=float, default=None, help="Stop when e_tr <= tol_tr (optional)")
    parser.add_argument("--delta", type=float, default=1e-6, help="Paper stopping threshold on ||E1||_F,||E2||_F")

    parser.add_argument("--H_kind", type=str, default="random",
                        help="random / commuting / conjugated_from_diagonal")
    parser.add_argument("--H_scale", type=float, default=1.0)
    parser.add_argument("--marginal_kind", type=str, default="medium",
                        help="easy / medium / hard (per src.instances.gen_marginal)")
    parser.add_argument("--hard_delta", type=float, default=1e-4)

    parser.add_argument("--normalize_cost", action="store_true",
                        help="Shift+scale H so spectrum ~ [0,1]. Often crucial for Alg 2.2.")
    parser.add_argument("--gauge_trace0", action="store_true",
                        help="Enable gauge fix (keeps U⊕V invariant, sets Tr(U)=0).")
    parser.add_argument("--project_pi", action="store_true",
                        help="Project pi to density during logging for numerical robustness.")
    parser.add_argument("--jitter", type=float, default=1e-12)

    parser.add_argument("--run_bgda", action="store_true",
                        help="Also run BGDA baseline for sanity check.")
    parser.add_argument("--T_bgda", type=int, default=500)
    parser.add_argument("--eta_grid", type=str, default="0.1,0.5,1.0,2.0,5.0",
                        help="Comma-separated eta grid for tune_bgda_eta")

    args = parser.parse_args()
    run_dbga_diagnosis(args)
