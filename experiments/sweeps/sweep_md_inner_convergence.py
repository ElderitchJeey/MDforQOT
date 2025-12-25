# experiments/sweeps/sweep_md_inner_convergence.py
#
# Sweep inner-loop convergence for md_inner_update_i:
# - Fix N,d,eps,eta_inner
# - Sweep seeds
# - For each seed and each block i:
#     run inner up to M_max
#     record whether tol_inner is hit
#     record hit_iter and hit_gibbs_calls
#     record final error and total calls
#
# Run:
#   python -m experiments.sweeps.sweep_md_inner_convergence --N 2 --d 4 --eps 0.5 --num_seeds 10 --M_max 200

import os
import csv
import argparse
import numpy as np

from src.instances import gen_marginal, gen_H_random
from src.SolverofEQOT import (
    md_inner_update_i,
    gibbs_state_from_potentials,
)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def first_hit_index(arr, tol: float):
    """Return first index n such that arr[n] <= tol; else None."""
    for k, v in enumerate(arr):
        if float(v) <= float(tol):
            return k
    return None


def main():
    parser = argparse.ArgumentParser(description="Sweep MD inner-loop convergence.")
    parser.add_argument("--N", type=int, default=2)
    parser.add_argument("--d", type=int, default=4)
    parser.add_argument("--eps", type=float, default=0.5)
    parser.add_argument("--eta_inner", type=float, default=1.0)
    parser.add_argument("--M_max", type=int, default=200)
    parser.add_argument("--tol_inner", type=float, default=1e-8)
    parser.add_argument("--num_seeds", type=int, default=10)
    parser.add_argument("--kind", type=str, default="medium")
    parser.add_argument("--H_scale", type=float, default=1.0)
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()

    N, d = args.N, args.d
    dims = [d] * N

    ensure_dir(args.results_dir)
    out_csv = os.path.join(
        args.results_dir,
        f"sweep_md_inner_conv_N{N}_d{d}_eps{args.eps:g}_seeds{args.num_seeds}.csv",
    )

    seeds = list(range(args.num_seeds))

    print("=" * 96)
    print("Sweep: MD inner-loop convergence")
    print(f"N={N}, d={d}, eps={args.eps}, eta_inner={args.eta_inner}")
    print(f"M_max={args.M_max}, tol_inner={args.tol_inner}, seeds=0..{args.num_seeds-1}")
    print("=" * 96)

    rows = []

    for seed in seeds:
        rng = np.random.default_rng(seed)
        gammas = [gen_marginal(d=d, rng=rng, kind=args.kind) for _ in range(N)]
        H = gen_H_random(dims=dims, rng=rng, scale=args.H_scale)

        # Start from U=0 and its corresponding pi0
        U0 = [np.zeros((d, d), dtype=complex) for _ in range(N)]
        pi0 = gibbs_state_from_potentials(U0, H, args.eps, dims, jitter=1e-12, project=True)

        for i in range(N):
            res = md_inner_update_i(
                i=i,
                U_list=U0,
                H=H,
                gamma_i=gammas[i],
                eps=args.eps,
                dims=dims,
                pi0=pi0,
                eta_inner=args.eta_inner,
                M_inner=args.M_max,
                tol_inner=args.tol_inner,
                jitter=1e-12,
                project_pi=True,
                reset_counter=True,
                keep_history=True,
            )

            e_list = np.array(res.e_i_tr_list, dtype=float)
            hit_k = first_hit_index(e_list, args.tol_inner)

            if hit_k is None:
                hit = False
                hit_iter = -1
                hit_calls = -1
            else:
                hit = True
                hit_iter = int(hit_k)
                hit_calls = int(res.gibbs_calls_list[hit_k]) if res.gibbs_calls_list else -1

            rows.append({
                "seed": seed,
                "N": N,
                "d": d,
                "eps": float(args.eps),
                "eta_inner": float(args.eta_inner),
                "M_max": int(args.M_max),
                "tol_inner": float(args.tol_inner),

                "i": int(i),
                "hit": bool(hit),
                "hit_iter": int(hit_iter),
                "hit_gibbs_calls": int(hit_calls),

                "e0": float(e_list[0]) if len(e_list) > 0 else float("nan"),
                "e_final": float(e_list[-1]) if len(e_list) > 0 else float("nan"),
                "gibbs_calls_total": int(res.gibbs_calls),
            })

        print(f"[seed={seed}] done")

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
