# experiments/test_metrics.py
#
# Comprehensive sanity tests for src/metrics.py across:
#   - BGDA: block_gradient_ascent (+ tune_bgda_eta)
#   - Potential KL descent: potential_marginal_kl_descent
#   - MD-type Sinkhorn: md_type_sinkhorn_potential
#
# Run from repo root:
#   python -m experiments.test_metrics
#
# Notes:
# - Assumes your repo provides:
#     src/instances.py: gen_H_random, gen_marginal
#     src/SolverofEQOT.py: tune_bgda_eta, block_gradient_ascent, potential_marginal_kl_descent, md_type_sinkhorn_potential
#     src/metrics.py: total_time, time_per_gibbs_call, final_scalar_metrics, trajectory_arrays, error_vs_gibbs,
#                     objective_F_marg, marginal_trace_errors
#
# If your instance module is src/instance.py, change the import accordingly.

import argparse
import numpy as np

from src.instances import gen_H_random, gen_marginal

from src.SolverofEQOT import (
    tune_bgda_eta,
    block_gradient_ascent,
    potential_marginal_kl_descent,
    md_type_sinkhorn_potential,
)

from src.metrics import (
    total_time,
    time_per_gibbs_call,
    final_scalar_metrics,
    trajectory_arrays,
    error_vs_gibbs,
    objective_F_marg,
    marginal_trace_errors,
)


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _print_header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def _check_metrics_common(name: str, res, H, gammas, dims, tol_tr: float) -> None:
    """
    Common checks for any ResultLike object returned by your solvers.
    """
    print(f"\n[{name}] Checking metrics...")

    # --- basic trajectories ---
    _assert(hasattr(res, "e_tr_list"), f"{name}: missing e_tr_list")
    _assert(hasattr(res, "F_list"), f"{name}: missing F_list")
    _assert(hasattr(res, "times"), f"{name}: missing times")
    _assert(hasattr(res, "gibbs_calls"), f"{name}: missing gibbs_calls")
    _assert(hasattr(res, "pi"), f"{name}: missing pi")

    e_tr = res.e_tr_list
    F = res.F_list
    t = res.times

    _assert(len(e_tr) == len(F) == len(t), f"{name}: len(e_tr), len(F), len(times) not equal")
    _assert(len(e_tr) >= 1, f"{name}: empty trajectories")

    # --- metrics helpers ---
    tt = total_time(res)
    tpg = time_per_gibbs_call(res)
    sm = final_scalar_metrics(res)
    tr = trajectory_arrays(res)
    evg = error_vs_gibbs(res)

    print(f"  total_time={tt:.6f}s, time_per_gibbs={tpg:.6e}s/call")
    print(f"  final_scalar_metrics={sm}")
    print(f"  traj lens: e_tr={len(tr['e_tr'])}, F={len(tr['F'])}, t={len(tr['t'])}")
    print(f"  error_vs_gibbs lens: gibbs={len(evg['gibbs'])}, e_tr={len(evg['e_tr'])}")

    # --- self-consistency ---
    _assert(np.isfinite(tt) and tt >= 0.0, f"{name}: total_time invalid")
    _assert(np.isfinite(tpg) and tpg >= 0.0, f"{name}: time_per_gibbs invalid")
    _assert(set(["time_sec", "gibbs_calls", "final_e_tr", "final_F"]).issubset(sm.keys()),
            f"{name}: final_scalar_metrics missing keys")

    _assert(len(evg["gibbs"]) == len(evg["e_tr"]), f"{name}: error_vs_gibbs length mismatch")

    # If you implemented gibbs_calls_list in the solver, it should be exact:
    if getattr(res, "gibbs_calls_list", None) is not None:
        _assert(len(res.gibbs_calls_list) == len(res.e_tr_list),
                f"{name}: gibbs_calls_list length != e_tr_list length")
        _assert(int(res.gibbs_calls_list[-1]) == int(res.gibbs_calls),
                f"{name}: gibbs_calls_list[-1] != gibbs_calls total")

    # --- verify forwarded canonical metrics on final pi ---
    pi_final = res.pi
    errs = marginal_trace_errors(pi_final, gammas, dims)
    Fv = objective_F_marg(pi_final, gammas, dims)

    print(f"  marginal_trace_errors(pi_final) = {errs}  (max={float(np.max(errs)):.3e})")
    print(f"  objective_F_marg(pi_final) = {float(Fv):.6e}")

    _assert(np.all(np.isfinite(errs)), f"{name}: marginal_trace_errors returned non-finite")
    _assert(np.isfinite(float(Fv)), f"{name}: objective_F_marg returned non-finite")

    # Optional: if solver claims converged, final error should be near tol_tr
    if bool(getattr(res, "converged", False)):
        final_e = float(res.e_tr_list[-1])
        _assert(final_e <= 1.2 * tol_tr, f"{name}: converged=True but final_e_tr={final_e} > 1.2*tol_tr")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", nargs="+", type=int, default=[2, 2], help="e.g. --dims 2 2 or --dims 2 2 2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eps", type=float, default=0.5)
    ap.add_argument("--tol", type=float, default=1e-8)

    # budgets for each method
    ap.add_argument("--T_bgda", type=int, default=300)
    ap.add_argument("--T_kl", type=int, default=200)
    ap.add_argument("--T_md", type=int, default=100)

    # BGDA eta tuning
    ap.add_argument("--eta_grid", nargs="+", type=float, default=[0.25, 0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--prefer", type=str, default="gibbs_calls", choices=["gibbs_calls", "time_sec"])
    ap.add_argument("--gauge_trace0", action="store_true")
    ap.add_argument("--no_project", action="store_true")

    # KL step size
    ap.add_argument("--eta_kl", type=float, default=0.5, help="Step size for potential_marginal_kl_descent (override if desired)")

    # MD-Sinkhorn inner controls (your implementation enforces eta_inner=1)
    ap.add_argument("--M_inner", type=int, default=1)

    # instance controls
    ap.add_argument("--H_scale", type=float, default=1.0)
    ap.add_argument("--kind", type=str, default="medium", choices=["easy", "medium", "hard"])
    ap.add_argument("--hard_delta", type=float, default=1e-4)

    args = ap.parse_args()

    dims = [int(x) for x in args.dims]
    rng = np.random.default_rng(int(args.seed))

    _print_header("test_metrics.py — metrics sanity across BGDA / KL / MD-Sinkhorn")
    print(f"dims={dims}, seed={args.seed}, eps={args.eps}, tol={args.tol}")
    print(f"T_bgda={args.T_bgda}, T_kl={args.T_kl}, T_md={args.T_md}")
    print(f"BGDA eta_grid={args.eta_grid}, prefer={args.prefer}, gauge_trace0={args.gauge_trace0}, project_pi={not args.no_project}")
    print(f"KL eta={args.eta_kl}")
    print(f"MD M_inner={args.M_inner}")
    print(f"instance: H_scale={args.H_scale}, kind={args.kind}, hard_delta={args.hard_delta}")

    # --- generate one fixed instance used by all methods ---
    H = gen_H_random(dims=dims, rng=rng, scale=float(args.H_scale))
    gammas = [
        gen_marginal(d=int(d), rng=rng, kind=str(args.kind), hard_delta=float(args.hard_delta))
        for d in dims
    ]

    # ============================================================
    # 1) BGDA (tune eta then run)
    # ============================================================
    _print_header("BGDA: tune eta + run block_gradient_ascent")
    tune = tune_bgda_eta(
        H=H,
        gammas=gammas,
        eps=float(args.eps),
        dims=dims,
        eta_grid=[float(x) for x in args.eta_grid],
        T=int(args.T_bgda),
        tol_tr=float(args.tol),
        gauge_trace0=bool(args.gauge_trace0),
        project_pi=(not args.no_project),
        store_hist=False,
        prefer=str(args.prefer),
    )
    best_eta = float(tune.best_eta)
    print(f"tuned best_eta = {best_eta}")

    res_bgda = block_gradient_ascent(
        H=H,
        gammas=gammas,
        eps=float(args.eps),
        dims=dims,
        T=int(args.T_bgda),
        eta=float(best_eta),
        tol_tr=float(args.tol),
        gauge_trace0=bool(args.gauge_trace0),
        project_pi=(not args.no_project),
        store_hist=False,
    )
    _check_metrics_common("BGDA", res_bgda, H, gammas, dims, tol_tr=float(args.tol))

    # ============================================================
    # 2) Potential KL descent
    # ============================================================
    _print_header("KL: run potential_marginal_kl_descent")
    res_kl = potential_marginal_kl_descent(
        H=H,
        gammas=gammas,
        eps=float(args.eps),
        dims=dims,
        T=int(args.T_kl),
        eta=float(args.eta_kl),
        tol_tr=float(args.tol),
        store_hist=False,
    )
    _check_metrics_common("KL", res_kl, H, gammas, dims, tol_tr=float(args.tol))

    # ============================================================
    # 3) MD-type Sinkhorn
    # ============================================================
    _print_header("MD-Sinkhorn: run md_type_sinkhorn_potential")
    res_md = md_type_sinkhorn_potential(
        H=H,
        gammas=gammas,
        eps=float(args.eps),
        dims=dims,
        T_outer=int(args.T_md),
        tol_tr=float(args.tol),
        jitter=1e-12,
        eta_inner=1.0,
        M_inner=int(args.M_inner),
        keep_U_hist=False,
        keep_pi_hist=False,
        tol_inner=None,
    )
    _check_metrics_common("MD-Sinkhorn", res_md, H, gammas, dims, tol_tr=float(args.tol))

    _print_header("OK — metrics.py works across all three solvers")
    print("Done.")


if __name__ == "__main__":
    main()
