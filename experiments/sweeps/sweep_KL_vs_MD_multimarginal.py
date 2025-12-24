# experiments/sweeps/sweep_KL_vs_MD_multimarginal.py

import os
import csv
import time
from typing import Dict, Any, List, Tuple

import numpy as np

from src.instances import gen_marginal, gen_H_random
from src.SolverofEQOT import (
    potential_marginal_kl_descent,
    md_type_sinkhorn_potential,
)
from src.metrics import same_limit


def make_instance(seed: int, d: int, N: int, eps: float, H_scale: float, kind: str):
    rng = np.random.default_rng(seed)
    dims = [d] * N
    gammas = [gen_marginal(d=d, rng=rng, kind=kind) for _ in range(N)]
    H = gen_H_random(dims=dims, rng=rng, scale=H_scale)
    return H, gammas, dims


def converged_by_final_e_tr(res, tol_tr: float) -> Tuple[bool, float]:
    if hasattr(res, "e_tr_list") and res.e_tr_list:
        e = float(res.e_tr_list[-1])
        return (e <= tol_tr), e
    return bool(getattr(res, "converged", False)), float("nan")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def main():
    # -----------------------
    # Sweep configuration
    # -----------------------
    d = 3
    N_list = [2, 3, 4]
    seeds = list(range(5))

    eps = 0.5
    H_scale = 1.0
    kind = "medium"

    tol_tr = 1e-8
    tol_pi = 1e-6

    T_kl = 3000
    T_md = 3000

    out_dir = "results"
    ensure_dir(out_dir)
    out_csv = os.path.join(out_dir, "sweep_KL_vs_MD_multimarginal_d3.csv")

    print("=" * 96)
    print("Sweep: KL-descent vs MD-Sinkhorn (multi-marginal)")
    print(f"d={d}, N_list={N_list}, seeds={seeds}")
    print("=" * 96)

    rows: List[Dict[str, Any]] = []

    for N in N_list:
        for seed in seeds:
            print(f"\n--- N={N}, seed={seed} ---")
            H, gammas, dims = make_instance(
                seed=seed, d=d, N=N, eps=eps, H_scale=H_scale, kind=kind
            )

            # -------- KL-descent --------
            res_kl = potential_marginal_kl_descent(
                H=H,
                gammas=gammas,
                eps=eps,
                dims=dims,
                T=T_kl,
                eta=None,              # default eta = 1/N
                jitter_log=1e-12,
                tol_tr=tol_tr,
                store_hist=False,
                project_pi=True,
            )
            kl_conv, kl_e = converged_by_final_e_tr(res_kl, tol_tr)

            # -------- MD-Sinkhorn --------
            res_md = md_type_sinkhorn_potential(
                H=H,
                gammas=gammas,
                eps=eps,
                dims=dims,
                T_outer=T_md,
                tol_tr=tol_tr,
                jitter=1e-12,
                eta_inner=1.0,
                M_inner=1,
                project_pi=True,
                keep_U_hist=False,
                keep_pi_hist=False,
            )
            md_conv, md_e = converged_by_final_e_tr(res_md, tol_tr)

            # -------- same-limit --------
            out = same_limit(
                res_md,
                res_kl,
                tol_pi=tol_pi,
                metric="trace",
                require_converged=True,
                tol_e_tr=tol_tr,
            )

            row = {
                "d": d,
                "N": N,
                "seed": seed,
                "eps": eps,
                "H_scale": H_scale,
                "tol_tr": tol_tr,
                "tol_pi": tol_pi,

                "kl_converged": kl_conv,
                "kl_final_e_tr": kl_e,
                "kl_gibbs_calls": int(getattr(res_kl, "gibbs_calls", -1)),

                "md_converged": md_conv,
                "md_final_e_tr": md_e,
                "md_gibbs_calls": int(getattr(res_md, "gibbs_calls", -1)),

                "both_converged": (kl_conv and md_conv),
                "same_limit": out["same_limit"],
                "dist_pi": out["dist_pi"],
            }

            rows.append(row)

            print(
                f"KL conv={kl_conv}  MD conv={md_conv}  "
                f"same={out['same_limit']}  dist={out['dist_pi']:.3e}"
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
