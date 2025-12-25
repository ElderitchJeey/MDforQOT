# experiments/sweeps/sweep_md_minner_sensitivity.py
#
# Sweep MD-Sinkhorn inner iterations M_inner for fixed (N=2, d=4) and 5 seeds.
# Compare:
#   - convergence rate
#   - mean Gibbs calls among converged runs
#   - same-limit rate vs reference M_inner=1 (conditioned on both converging)

import os
import csv
from typing import Dict, Any, List, Tuple

import numpy as np

from src.instances import gen_marginal, gen_H_random
from src.SolverofEQOT import md_type_sinkhorn_potential
from src.metrics import same_limit


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


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


def run_md(H, gammas, dims, eps, T_outer, tol_tr, eta_inner, M_inner):
    return md_type_sinkhorn_potential(
        H=H,
        gammas=gammas,
        eps=eps,
        dims=dims,
        T_outer=T_outer,
        tol_tr=tol_tr,
        jitter=1e-12,
        eta_inner=eta_inner,
        M_inner=M_inner,
        project_pi=True,
        keep_U_hist=False,
        keep_pi_hist=False,
    )


def main():
    # -----------------------
    # Config
    # -----------------------
    N = 2
    d = 12
    seeds = list(range(5))

    eps = 0.5
    H_scale = 1.0
    kind = "medium"

    tol_tr = 1e-8
    tol_pi = 1e-6

    T_outer = 3000
    eta_inner = 1.0

    M_list = [1, 2, 3]
    M_ref = 1

    out_dir = "results"
    ensure_dir(out_dir)
    out_csv = os.path.join(out_dir, f"sweep_md_Minner_N{N}_d{d}_seed{len(seeds)}.csv")

    print("=" * 96)
    print("Sweep: MD-Sinkhorn sensitivity to M_inner")
    print(f"N={N}, d={d}, seeds={seeds}, eps={eps}, T_outer={T_outer}")
    print(f"M_list={M_list}, reference M_ref={M_ref}")
    print("=" * 96)

    rows: List[Dict[str, Any]] = []

    # Pre-generate instances per seed so every M_inner sees the same instance
    instances = {}
    for seed in seeds:
        H, gammas, dims = make_instance(seed=seed, d=d, N=N, eps=eps, H_scale=H_scale, kind=kind)
        instances[seed] = (H, gammas, dims)

    # Run reference (M_ref) for each seed
    ref_results = {}
    for seed in seeds:
        H, gammas, dims = instances[seed]
        res_ref = run_md(H, gammas, dims, eps=eps, T_outer=T_outer, tol_tr=tol_tr, eta_inner=eta_inner, M_inner=M_ref)
        ref_results[seed] = res_ref
        conv_ref, e_ref = converged_by_final_e_tr(res_ref, tol_tr)
        calls_ref = int(getattr(res_ref, "gibbs_calls", -1))
        print(f"[ref] seed={seed} M={M_ref} conv={conv_ref} final_e_tr={e_ref:.3e} calls={calls_ref}")

    # Sweep M_inner
    for M_inner in M_list:
        for seed in seeds:
            H, gammas, dims = instances[seed]

            res = run_md(H, gammas, dims, eps=eps, T_outer=T_outer, tol_tr=tol_tr, eta_inner=eta_inner, M_inner=M_inner)
            conv, e_tr = converged_by_final_e_tr(res, tol_tr)
            calls = int(getattr(res, "gibbs_calls", -1))

            # same-limit vs reference M_ref
            out = same_limit(
                res,
                ref_results[seed],
                tol_pi=tol_pi,
                metric="trace",
                require_converged=True,   # only meaningful when both converged
                tol_e_tr=tol_tr,
            )

            rows.append({
                "N": N,
                "d": d,
                "seed": seed,
                "eps": eps,
                "T_outer": T_outer,
                "eta_inner": eta_inner,

                "M_inner": int(M_inner),
                "M_ref": int(M_ref),

                "converged": conv,
                "final_e_tr": e_tr,
                "gibbs_calls": calls,

                "ref_converged": converged_by_final_e_tr(ref_results[seed], tol_tr)[0],

                "same_limit_vs_ref": out["same_limit"],
                "dist_pi_vs_ref": out["dist_pi"],
            })

        convs = [r["converged"] for r in rows if r["M_inner"] == int(M_inner)]
        print(f"M_inner={M_inner}  conv_rate={np.mean(convs):.2f}")

    # write csv
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print("\nDone.")
    print(f"Wrote: {out_csv}")


if __name__ == "__main__":
    main()
