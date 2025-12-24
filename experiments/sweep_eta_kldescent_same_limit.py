# experiments/sweep_eta_kldescent_same_limit.py

import numpy as np

from src.instances import gen_marginal, gen_H_random
from src.SolverofEQOT import potential_marginal_kl_descent
from src.metrics import same_limit, pi_distance, final_pi


def make_instance(seed: int, d: int, N: int, H_scale: float = 1.0, kind: str = "medium"):
    rng = np.random.default_rng(seed)
    dims = [d] * N
    gammas = [gen_marginal(d=d, rng=rng, kind=kind) for _ in range(N)]
    H = gen_H_random(dims=dims, rng=rng, scale=H_scale)
    return H, gammas, dims


def eta_grid_1_over_N_to_1(N: int, num: int = 13) -> np.ndarray:
    """
    Log-spaced grid in [1/N, 1], inclusive.
    """
    eta_min = 1.0 / N
    eta_max = 1.0
    # avoid degenerate if N=1
    if np.isclose(eta_min, eta_max):
        return np.array([eta_max], dtype=float)
    return np.geomspace(eta_min, eta_max, num=num).astype(float)


def converged_by_e_tr(res, tol_tr: float) -> tuple[bool, float]:
    """
    Robust convergence check:
      converged := (final e_tr <= tol_tr), if e_tr_list exists.
    Returns (converged, final_e_tr).
    """
    if hasattr(res, "e_tr_list") and res.e_tr_list is not None and len(res.e_tr_list) > 0:
        final_e = float(res.e_tr_list[-1])
        return (final_e <= tol_tr), final_e
    # fallback
    final_e = float("nan")
    conv = bool(getattr(res, "converged", False))
    return conv, final_e


def main():
    # -----------------------
    # Config (edit as needed)
    # -----------------------
    seed = 0
    d = 3
    N = 2

    eps = 0.5
    H_scale = 1.0
    kind = "medium"

    T = 2000
    tol_tr = 1e-8
    tol_pi = 1e-6

    # KL-descent internal numerics
    jitter_log = 1e-12
    project_pi = True
    store_hist = False

    # eta sweep grid
    num_eta = 13
    etas = eta_grid_1_over_N_to_1(N=N, num=num_eta)

    print("=" * 96)
    print(f"KL eta sweep on one instance")
    print(f"  seed={seed}, d={d}, N={N}, eps={eps}, H_scale={H_scale}, kind={kind}")
    print(f"  T={T}, tol_tr={tol_tr}, tol_pi={tol_pi}, num_eta={num_eta}")
    print(f"  eta range: [{1.0/N:.6g}, 1.0]")
    print("=" * 96)

    # -----------------------
    # Build fixed instance
    # -----------------------
    H, gammas, dims = make_instance(seed=seed, d=d, N=N, H_scale=H_scale, kind=kind)

    # -----------------------
    # Run sweep
    # -----------------------
    results = []
    for eta in etas:
        res = potential_marginal_kl_descent(
            H=H,
            gammas=gammas,
            eps=eps,
            dims=dims,
            T=T,
            eta=float(eta),
            jitter_log=jitter_log,
            tol_tr=tol_tr,
            store_hist=store_hist,
            project_pi=project_pi,
        )
        conv, final_e = converged_by_e_tr(res, tol_tr=tol_tr)
        gibbs_calls = int(getattr(res, "gibbs_calls", -1))
        results.append(
            {
                "eta": float(eta),
                "res": res,
                "converged": bool(conv),
                "final_e_tr": float(final_e),
                "gibbs_calls": gibbs_calls,
            }
        )
        print(f"eta={eta:.6g}  converged={conv}  final_e_tr={final_e:.6g}  gibbs_calls={gibbs_calls}")

    # -----------------------
    # Pick a reference limit among converged runs
    # Strategy: choose the smallest eta (closest to 1/N) that converged;
    # if none converged, stop.
    # -----------------------
    conv_runs = [r for r in results if r["converged"]]
    if len(conv_runs) == 0:
        print("\nNo eta in [1/N,1] converged under current (T, tol_tr, instance scale).")
        print("Try increasing T, loosening tol_tr, or lowering H_scale / increasing eps.")
        return

    ref = sorted(conv_runs, key=lambda x: x["eta"])[0]
    ref_eta = ref["eta"]
    ref_res = ref["res"]
    ref_pi = final_pi(ref_res)

    print("\n" + "-" * 96)
    print(f"Reference run: eta={ref_eta:.6g} (smallest converged eta)")
    print("-" * 96)

    # -----------------------
    # Compare each converged eta to reference by final pi
    # -----------------------
    agree = []
    disagree = []
    for r in conv_runs:
        eta = r["eta"]
        res = r["res"]
        out = same_limit(
            res,
            ref_res,
            tol_pi=tol_pi,
            metric="trace",
            require_converged=True,
            tol_e_tr=tol_tr,
        )
        r["dist_to_ref"] = float(out["dist_pi"])
        r["same_as_ref"] = bool(out["same_limit"])
        if r["same_as_ref"]:
            agree.append(r)
        else:
            disagree.append(r)

    print("Converged etas that match the reference limit (by final pi):")
    for r in sorted(agree, key=lambda x: x["eta"]):
        print(f"  eta={r['eta']:.6g}  dist_to_ref={r['dist_to_ref']:.3e}  e_tr={r['final_e_tr']:.3e}")

    if len(disagree) > 0:
        print("\nConverged etas that DO NOT match the reference limit (unexpected if optimum unique):")
        for r in sorted(disagree, key=lambda x: x["eta"]):
            print(f"  eta={r['eta']:.6g}  dist_to_ref={r['dist_to_ref']:.3e}  e_tr={r['final_e_tr']:.3e}")

        # Optional: show pairwise distances among the "disagree" set (small)
        print("\nPairwise trace distances among disagreeing converged runs:")
        dis = sorted(disagree, key=lambda x: x["eta"])
        for i in range(len(dis)):
            for j in range(i + 1, len(dis)):
                pi_i = final_pi(dis[i]["res"])
                pi_j = final_pi(dis[j]["res"])
                d_ij = pi_distance(pi_i, pi_j, metric="trace")
                print(f"  eta {dis[i]['eta']:.6g} vs {dis[j]['eta']:.6g}: dist={d_ij:.3e}")

    # -----------------------
    # Summary
    # -----------------------
    num_conv = len(conv_runs)
    num_same = sum(1 for r in conv_runs if r.get("same_as_ref", False))
    print("\n" + "=" * 96)
    print("SUMMARY")
    print(f"  converged: {num_conv}/{len(results)}")
    print(f"  same-limit among converged (vs ref eta={ref_eta:.6g}): {num_same}/{num_conv}")
    print("=" * 96)


if __name__ == "__main__":
    main()
