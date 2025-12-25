# experiments/sweeps/sweep_kl_eta_sensitivity.py
#
# Sweep KL-descent step size eta for fixed (N=2, d=3) and 5 seeds.
# For each eta:
#   - convergence rate
#   - mean Gibbs calls among converged runs
#   - same-limit rate vs reference eta_ref=1/N (conditioned on both converging)

import os
import csv
from typing import Dict, Any, List, Tuple

import numpy as np

from src.instances import gen_marginal, gen_H_random
from src.SolverofEQOT import potential_marginal_kl_descent
from src.metrics import same_limit


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def to_bool(x) -> bool:
    return str(x).strip().lower() in ("true", "1", "yes")


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


def run_kl(H, gammas, dims, eps, T, eta, tol_tr):
    return potential_marginal_kl_descent(
        H=H,
        gammas=gammas,
        eps=eps,
        dims=dims,
        T=T,
        eta=eta,
        jitter_log=1e-12,
        tol_tr=tol_tr,
        store_hist=False,
        project_pi=True,
    )


def main():
    # -----------------------
    # Config
    # -----------------------
    N = 2
    d = 3
    seeds = list(range(5))

    eps = 0.5
    H_scale = 1.0
    kind = "medium"

    tol_tr = 1e-8
    tol_pi = 1e-6

    T = 4000

    eta_min = 0.5 / N
    eta_max = 1.0
    num_eta = 13
    eta_list = np.linspace(eta_min, eta_max, num_eta)

    eta_ref = 1.0 / N

    out_dir = "results"
    ensure_dir(out_dir)
    out_csv = os.path.join(out_dir, f"sweep_kl_eta_N{N}_d{d}_seed{len(seeds)}.csv")

    print("=" * 96)
    print("Sweep: KL-descent eta sensitivity")
    print(f"N={N}, d={d}, seeds={seeds}, eps={eps}, T={T}")
    print(f"eta range: [{eta_min}, {eta_max}]  num_eta={num_eta}")
    print(f"reference eta_ref={eta_ref}")
    print("=" * 96)

    rows: List[Dict[str, Any]] = []

    # Pre-generate instances per seed so every eta sees the same instance
    instances = {}
    for seed in seeds:
        H, gammas, dims = make_instance(seed=seed, d=d, N=N, eps=eps, H_scale=H_scale, kind=kind)
        instances[seed] = (H, gammas, dims)

    # Run reference for each seed (eta_ref)
    ref_results = {}
    for seed in seeds:
        H, gammas, dims = instances[seed]
        res_ref = run_kl(H, gammas, dims, eps=eps, T=T, eta=eta_ref, tol_tr=tol_tr)
        ref_results[seed] = res_ref
        conv_ref, e_ref = converged_by_final_e_tr(res_ref, tol_tr)
        print(f"[ref] seed={seed} eta={eta_ref:.6f} conv={conv_ref} final_e_tr={e_ref:.3e} calls={getattr(res_ref,'gibbs_calls',-1)}")

    # Sweep eta
    for eta in eta_list:
        for seed in seeds:
            H, gammas, dims = instances[seed]

            res = run_kl(H, gammas, dims, eps=eps, T=T, eta=float(eta), tol_tr=tol_tr)
            conv, e_tr = converged_by_final_e_tr(res, tol_tr)
            calls = int(getattr(res, "gibbs_calls", -1))

            # same-limit vs reference eta_ref
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
                "T": T,

                "eta": float(eta),
                "eta_ref": eta_ref,

                "converged": conv,
                "final_e_tr": e_tr,
                "gibbs_calls": calls,

                "ref_converged": to_bool(getattr(ref_results[seed], "converged", False)) if hasattr(ref_results[seed], "converged") else converged_by_final_e_tr(ref_results[seed], tol_tr)[0],

                "same_limit_vs_ref": out["same_limit"],
                "dist_pi_vs_ref": out["dist_pi"],
            })

        # quick summary line per eta
        convs = [r["converged"] for r in rows if abs(r["eta"] - float(eta)) < 1e-15]
        print(f"eta={eta:.6f}  conv_rate={np.mean(convs):.2f}")

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
