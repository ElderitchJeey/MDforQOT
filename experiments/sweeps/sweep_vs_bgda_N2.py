# experiments/sweeps/sweep_vs_bgda_N2.py

import os
import csv
import time
from dataclasses import asdict
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

from src.instances import gen_marginal, gen_H_random
from src.SolverofEQOT import (
    tune_bgda_eta,
    potential_marginal_kl_descent,
    md_type_sinkhorn_potential,
)
from src.metrics import same_limit


# -----------------------
# Instance generation
# -----------------------
def make_instance(seed: int, d: int, N: int = 2, eps: float = 0.5, H_scale: float = 1.0, kind: str = "medium"):
    rng = np.random.default_rng(seed)
    dims = [d] * N
    gammas = [gen_marginal(d=d, rng=rng, kind=kind) for _ in range(N)]
    H = gen_H_random(dims=dims, rng=rng, scale=H_scale)
    return H, gammas, dims


def converged_by_final_e_tr(res: Any, tol_tr: float) -> Tuple[bool, float]:
    if hasattr(res, "e_tr_list") and res.e_tr_list is not None and len(res.e_tr_list) > 0:
        e = float(res.e_tr_list[-1])
        return (e <= tol_tr), e
    return bool(getattr(res, "converged", False)), float("nan")


def safe_int(x, default=-1) -> int:
    try:
        return int(x)
    except Exception:
        return default


def run_one_case(
    *,
    seed: int,
    d: int,
    eps: float,
    H_scale: float,
    kind: str,
    tol_tr: float,
    tol_pi: float,
    # budgets
    T_tune: int,
    T_bgda: int,
    T_kl: int,
    T_md: int,
    # bgda tuning
    eta_grid: List[float],
    prefer: str = "gibbs_calls",
) -> Dict[str, Any]:
    """
    One (seed, d) case:
      - tune BGDA eta
      - run BGDA(best)
      - run KLdescent (eta default=1/N)
      - run MD sinkhorn
      - compare limits to BGDA by final pi
    """
    N = 2
    H, gammas, dims = make_instance(seed=seed, d=d, N=N, eps=eps, H_scale=H_scale, kind=kind)

    # -----------------------
    # BGDA tuning
    # -----------------------
    t0 = time.time()
    tune = tune_bgda_eta(
        H=H,
        gammas=gammas,
        eps=eps,
        dims=dims,
        eta_grid=eta_grid,
        T=T_tune,
        tol_tr=tol_tr,
        gauge_trace0=True,
        project_pi=True,
        store_hist=False,
        prefer=prefer,
    )
    t_tune = time.time() - t0

    best_eta = float(tune.best_eta)
    res_bgda = tune.best_run  # reuse best run

    bgda_conv, bgda_e = converged_by_final_e_tr(res_bgda, tol_tr)
    bgda_gibbs = safe_int(getattr(res_bgda, "gibbs_calls", -1))
    bgda_time = float(sum(getattr(res_bgda, "times", []))) if hasattr(res_bgda, "times") else float("nan")

    # If you prefer rerun BGDA(best_eta) with T_bgda, do it here.
    # In your repo, tune.best_run already used T_tune; often enough on laptop.
    # If you want the "final baseline" to always have T_bgda, uncomment:
    #
    # from src.SolverofEQOT import block_gradient_ascent
    # res_bgda = block_gradient_ascent(... T=T_bgda, eta=best_eta, ...)
    # bgda_conv, bgda_e = converged_by_final_e_tr(res_bgda, tol_tr)
    # bgda_gibbs = safe_int(getattr(res_bgda, "gibbs_calls", -1))

    # -----------------------
    # KLdescent (eta = 1/N default)
    # -----------------------
    t0 = time.time()
    res_kl = potential_marginal_kl_descent(
        H=H,
        gammas=gammas,
        eps=eps,
        dims=dims,
        T=T_kl,
        eta=None,            # default 1/N
        jitter_log=1e-12,
        tol_tr=tol_tr,
        store_hist=False,
        project_pi=True,
    )
    t_kl = time.time() - t0

    kl_conv, kl_e = converged_by_final_e_tr(res_kl, tol_tr)
    kl_gibbs = safe_int(getattr(res_kl, "gibbs_calls", -1))
    kl_time = float(sum(getattr(res_kl, "times", []))) if hasattr(res_kl, "times") else float("nan")

    out_kl = same_limit(
        res_kl,
        res_bgda,
        tol_pi=tol_pi,
        metric="trace",
        require_converged=True,
        tol_e_tr=tol_tr,
    )

    # -----------------------
    # MD-type Sinkhorn
    # -----------------------
    t0 = time.time()
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
        name="MD-Sinkhorn",
        keep_U_hist=False,
        keep_pi_hist=False,
        tol_inner=None,
        project_pi=True,
    )
    t_md = time.time() - t0

    md_conv, md_e = converged_by_final_e_tr(res_md, tol_tr)
    md_gibbs = safe_int(getattr(res_md, "gibbs_calls", -1))
    md_time = float(sum(getattr(res_md, "times", []))) if hasattr(res_md, "times") else float("nan")

    out_md = same_limit(
        res_md,
        res_bgda,
        tol_pi=tol_pi,
        metric="trace",
        require_converged=True,
        tol_e_tr=tol_tr,
    )
    # MD vs KL same-limit (only meaningful when both converged)
    out_md_vs_kl = same_limit(
        res_md,
        res_kl,
        tol_pi=tol_pi,
        metric="trace",
        require_converged=True,
        tol_e_tr=tol_tr,
    )


    # -----------------------
    # Pack one CSV row
    # -----------------------
    row = {
        # instance
        "N": 2,
        "d": d,
        "seed": seed,
        "eps": eps,
        "H_scale": H_scale,
        "kind": kind,
        "tol_tr": tol_tr,
        "tol_pi": tol_pi,

        # bgda tuning
        "bgda_eta_grid": "|".join(str(x) for x in eta_grid),
        "bgda_best_eta": best_eta,
        "bgda_tune_time_sec": t_tune,

        # BGDA outcomes
        "bgda_converged": bgda_conv,
        "bgda_final_e_tr": bgda_e,
        "bgda_gibbs_calls": bgda_gibbs,
        "bgda_wall_time_sec": bgda_time,

        # KL outcomes
        "kl_converged": kl_conv,
        "kl_final_e_tr": kl_e,
        "kl_gibbs_calls": kl_gibbs,
        "kl_wall_time_sec": kl_time,
        "kl_same_limit": out_kl["same_limit"],
        "kl_dist_pi_to_bgda": out_kl["dist_pi"],

        # MD outcomes
        "md_converged": md_conv,
        "md_final_e_tr": md_e,
        "md_gibbs_calls": md_gibbs,
        "md_wall_time_sec": md_time,
        "md_same_limit": out_md["same_limit"],
        "md_dist_pi_to_bgda": out_md["dist_pi"],
        
        #KL vs MD
        "md_vs_kl_same_limit": out_md_vs_kl["same_limit"],
        "md_vs_kl_dist_pi": out_md_vs_kl["dist_pi"],
        "md_vs_kl_both_converged": (md_conv and kl_conv),

    }
    return row


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def main():
    # -----------------------
    # User-facing sweep config
    # -----------------------
    N = 2
    d_list = [2, 3, 4]
    seeds = list(range(3))  # laptop-friendly; increase on lab machine

    eps = 0.5
    H_scale = 1.0
    kind = "medium"

    # tolerances for "converged" and "same limit"
    tol_tr = 1e-8
    tol_pi = 1e-6

    # budgets (laptop)
    T_tune = 1500
    T_bgda = 1500
    T_kl = 2000
    T_md = 2000

    # BGDA tuning grid
    eta_grid = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    prefer = "gibbs_calls"

    out_dir = "results"
    out_csv = os.path.join(out_dir, "sweep_vs_bgda_N2.csv")
    ensure_dir(out_dir)

    print("=" * 96)
    print("SWEEP vs BGDA (N=2)")
    print(f"d_list={d_list}, seeds={seeds}")
    print(f"eps={eps}, H_scale={H_scale}, kind={kind}")
    print(f"tol_tr={tol_tr}, tol_pi={tol_pi}")
    print(f"T_tune={T_tune}, T_kl={T_kl}, T_md={T_md}")
    print(f"eta_grid={eta_grid}, prefer={prefer}")
    print(f"Output: {out_csv}")
    print("=" * 96)

    # -----------------------
    # Run sweep + write CSV
    # -----------------------
    rows: List[Dict[str, Any]] = []
    for d in d_list:
        for seed in seeds:
            print(f"\n--- running case: d={d}, seed={seed} ---")
            row = run_one_case(
                seed=seed,
                d=d,
                eps=eps,
                H_scale=H_scale,
                kind=kind,
                tol_tr=tol_tr,
                tol_pi=tol_pi,
                T_tune=T_tune,
                T_bgda=T_bgda,
                T_kl=T_kl,
                T_md=T_md,
                eta_grid=eta_grid,
                prefer=prefer,
            )
            rows.append(row)
            print(f"BGDA: conv={row['bgda_converged']} e_tr={row['bgda_final_e_tr']:.3e} eta*={row['bgda_best_eta']}")
            print(f" KL : conv={row['kl_converged']}  same={row['kl_same_limit']} dist={row['kl_dist_pi_to_bgda']:.3e}")
            print(f" MD : conv={row['md_converged']}  same={row['md_same_limit']} dist={row['md_dist_pi_to_bgda']:.3e}")

    # write CSV
    fieldnames = list(rows[0].keys()) if rows else []
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print("\nDone.")
    print(f"Wrote: {out_csv}")


if __name__ == "__main__":
    main()
