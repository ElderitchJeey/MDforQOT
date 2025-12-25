# experiments/sweeps/sweep_speed_vs_d_N2.py
#
# N=2, sweep d, compare 3 algorithms:
#   - BGDA (eta tuned per instance via tune_bgda_eta)
#   - KL-descent (eta default = 1/N)
#   - MD-Sinkhorn (eta_inner=1, M_inner configurable)
#
# Output CSV: results/sweep_speed_vs_d_N2.csv
#
# Run:
#   python -m experiments.sweeps.sweep_speed_vs_d_N2
#
# Optional:
#   python -m experiments.sweeps.sweep_speed_vs_d_N2 --d_list 2 3 4 5 6 --num_seeds 10

import os
import csv
import argparse
import numpy as np

from src.instances import gen_marginal, gen_H_random
from src.SolverofEQOT import (
    tune_bgda_eta,
    block_gradient_ascent,
    potential_marginal_kl_descent,
    md_type_sinkhorn_potential,
)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def final_e_tr(res) -> float:
    if hasattr(res, "e_tr_list") and res.e_tr_list:
        return float(res.e_tr_list[-1])
    if hasattr(res, "final_e_tr"):
        return float(res.final_e_tr)
    return float("nan")


def final_time_sec(res) -> float:
    # In this repo, res.times usually stores elapsed seconds (cumulative).
    if hasattr(res, "times") and res.times:
        return float(res.times[-1])
    if hasattr(res, "time_sec"):
        return float(res.time_sec)
    return float("nan")


def make_instance(seed: int, d: int, N: int, kind: str, H_scale: float):
    rng = np.random.default_rng(seed)
    dims = [d] * N
    gammas = [gen_marginal(d=d, rng=rng, kind=kind) for _ in range(N)]
    H = gen_H_random(dims=dims, rng=rng, scale=H_scale)
    return H, gammas, dims


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d_list", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--num_seeds", type=int, default=5)

    parser.add_argument("--eps", type=float, default=0.5)
    parser.add_argument("--tol_tr", type=float, default=1e-8)

    # iteration budgets
    parser.add_argument("--T_bgda", type=int, default=3000)
    parser.add_argument("--T_kl", type=int, default=2000)
    parser.add_argument("--T_outer_md", type=int, default=2000)

    # BGDA tuning
    parser.add_argument("--eta_grid", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0])
    parser.add_argument("--prefer", type=str, default="gibbs_calls")  # or "time_sec"

    # MD inner config
    parser.add_argument("--eta_inner", type=float, default=1.0)   # md_type_sinkhorn_potential currently enforces eta_inner=1.0
    parser.add_argument("--M_inner", type=int, default=1)

    # instance generation
    parser.add_argument("--kind", type=str, default="medium")
    parser.add_argument("--H_scale", type=float, default=1.0)

    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()

    N = 2
    seeds = list(range(args.num_seeds))

    ensure_dir(args.results_dir)
    out_csv = os.path.join(args.results_dir, "sweep_speed_vs_d_N2.csv")

    print("=" * 96)
    print("Sweep speed vs d (N=2)")
    print(f"d_list={args.d_list}, seeds=0..{args.num_seeds-1}, eps={args.eps}, tol_tr={args.tol_tr}")
    print(f"BGDA: T={args.T_bgda}, eta_grid={args.eta_grid}, prefer={args.prefer}")
    print(f"KL:   T={args.T_kl}, eta=default(=1/N)")
    print(f"MD:   T_outer={args.T_outer_md}, eta_inner={args.eta_inner}, M_inner={args.M_inner}")
    print("=" * 96)

    rows = []

    for d in args.d_list:
        for seed in seeds:
            H, gammas, dims = make_instance(seed=seed, d=d, N=N, kind=args.kind, H_scale=args.H_scale)

            # -----------------------
            # BGDA (tuned eta per instance)
            # -----------------------
            tune = tune_bgda_eta(
                H=H,
                gammas=gammas,
                eps=args.eps,
                dims=dims,
                eta_grid=list(args.eta_grid),
                T=args.T_bgda,
                tol_tr=args.tol_tr,
                gauge_trace0=True,
                project_pi=True,
                store_hist=False,
                prefer=args.prefer,
            )
            bgda_eta = float(tune.best_eta)
            bgda_res = tune.best_run
            bgda_e = final_e_tr(bgda_res)
            bgda_t = final_time_sec(bgda_res)
            bgda_calls = int(getattr(bgda_res, "gibbs_calls", -1))
            bgda_conv = bool(bgda_e <= args.tol_tr)

            # -----------------------
            # KL-descent (eta=1/N by default)
            # -----------------------
            kl_res = potential_marginal_kl_descent(
                H=H,
                gammas=gammas,
                eps=args.eps,
                dims=dims,
                T=args.T_kl,
                eta=None,               # default = 1/N inside
                jitter_log=1e-12,
                tol_tr=args.tol_tr,
                store_hist=False,
                project_pi=True,
            )
            kl_e = final_e_tr(kl_res)
            kl_t = final_time_sec(kl_res)
            kl_calls = int(getattr(kl_res, "gibbs_calls", -1))
            kl_conv = bool(kl_e <= args.tol_tr)

            # -----------------------
            # MD-Sinkhorn
            # -----------------------
            md_res = md_type_sinkhorn_potential(
                H=H,
                gammas=gammas,
                eps=args.eps,
                dims=dims,
                T_outer=args.T_outer_md,
                tol_tr=args.tol_tr,
                jitter=1e-12,
                eta_inner=args.eta_inner,
                M_inner=args.M_inner,
                keep_U_hist=False,
                keep_pi_hist=False,
                tol_inner=None,
                project_pi=True,
            )
            md_e = final_e_tr(md_res)
            md_t = final_time_sec(md_res)
            md_calls = int(getattr(md_res, "gibbs_calls", -1))
            md_conv = bool(md_e <= args.tol_tr)

            rows.append({
                "N": N,
                "d": int(d),
                "seed": int(seed),
                "eps": float(args.eps),
                "tol_tr": float(args.tol_tr),

                "bgda_eta": bgda_eta,
                "bgda_converged": bgda_conv,
                "bgda_final_e_tr": bgda_e,
                "bgda_gibbs_calls": bgda_calls,
                "bgda_time_sec": bgda_t,

                "kl_eta": 1.0 / N,
                "kl_converged": kl_conv,
                "kl_final_e_tr": kl_e,
                "kl_gibbs_calls": kl_calls,
                "kl_time_sec": kl_t,

                "md_eta_inner": float(args.eta_inner),
                "md_M_inner": int(args.M_inner),
                "md_converged": md_conv,
                "md_final_e_tr": md_e,
                "md_gibbs_calls": md_calls,
                "md_time_sec": md_t,
            })

            print(
                f"[d={d}, seed={seed}] "
                f"BGDA(conv={bgda_conv}, calls={bgda_calls}, t={bgda_t:.2f}, eta={bgda_eta}) | "
                f"KL(conv={kl_conv}, calls={kl_calls}, t={kl_t:.2f}) | "
                f"MD(conv={md_conv}, calls={md_calls}, t={md_t:.2f}, M={args.M_inner})"
            )

    # write CSV
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print("\nDone.")
    print(f"Wrote: {out_csv}")


if __name__ == "__main__":
    main()
