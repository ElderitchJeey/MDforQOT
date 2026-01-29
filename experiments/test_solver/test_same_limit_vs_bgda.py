import numpy as np

from src.instances import gen_marginal, gen_H_random
from src.SolverofEQOT import (
    tune_bgda_eta,
    block_gradient_ascent,
    potential_marginal_kl_descent,
    md_type_sinkhorn_potential,
)
from src.metrics import same_limit


def make_instance(seed: int, d: int, N: int, H_scale: float = 1.0, kind: str = "medium"):
    rng = np.random.default_rng(seed)
    dims = [d] * N
    gammas = [gen_marginal(d=d, rng=rng, kind=kind) for _ in range(N)]
    H = gen_H_random(dims=dims, rng=rng, scale=H_scale)
    return H, gammas, dims


def main():
    seed = 0
    d = 3
    N = 2
    eps = 0.5

    tol_tr = 1e-8
    tol_pi = 1e-6

    # Iteration budgets
    T_tune = 3000     # for tuning BGDA eta
    T_bgda = 3000     # final BGDA run (could reuse best_run)
    T_kl = 300
    T_md_outer = 300

    # BGDA eta grid (you can change)
    eta_grid = [1.0, 2.0, 4.0, 8.0, 16.0]  # typical
    prefer = "gibbs_calls"           # fairness by expensive Gibbs evals

    H, gammas, dims = make_instance(seed=seed, d=d, N=N, H_scale=1.0, kind="medium")

    print("=" * 90)
    print(f"Instance: seed={seed}, d={d}, N={N}, eps={eps}, tol_tr={tol_tr}")
    print("=" * 90)

    # ------------------------------------------------------------
    # Tune BGDA eta (fair baseline)
    # ------------------------------------------------------------
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

    best_eta = float(tune.best_eta)
    res_bgda = tune.best_run  # reuse best run directly

    print("[BGDA tuned]")
    print("  eta_grid:", eta_grid)
    print("  prefer:", prefer)
    print("  best_eta:", best_eta)
    print("  converged:", res_bgda.converged)
    print("  final e_tr:", float(res_bgda.e_tr_list[-1]))
    print("  gibbs_calls:", int(res_bgda.gibbs_calls))
    print("")

    # (Optional) if you prefer rerun BGDA with best_eta and a longer T:
    # res_bgda = block_gradient_ascent(
    #     H=H, gammas=gammas, eps=eps, dims=dims,
    #     T=T_bgda, eta=best_eta, tol_tr=tol_tr,
    #     gauge_trace0=True, store_hist=False, project_pi=True,
    # )

    # ============================================================
    # 1) KLdescent (eta default = 1/N) vs tuned BGDA
    # ============================================================
    res_kl_default = potential_marginal_kl_descent(
        H=H,
        gammas=gammas,
        eps=eps,
        dims=dims,
        T=T_kl,
        eta=None,          # default inside solver (1/N)
        jitter_log=1e-12,
        tol_tr=tol_tr,
        store_hist=False,
        project_pi=True,
    )

    out1 = same_limit(
        res_kl_default,
        res_bgda,
        tol_pi=tol_pi,
        metric="trace",
        require_converged=True,
        tol_e_tr=tol_tr,
    )

    print("[1] KLdescent (eta=1/N default) vs tuned BGDA")
    print("  KL converged:", res_kl_default.converged)
    print("  KL final e_tr:", float(res_kl_default.e_tr_list[-1]))
    print("  same_limit:", out1["same_limit"])
    print("  dist_pi:", out1["dist_pi"])
    print("  KL gibbs_calls:", int(res_kl_default.gibbs_calls))
    print("")

    # ============================================================
    # 2) KLdescent (eta=1) vs tuned BGDA
    # ============================================================
    res_kl_eta1 = potential_marginal_kl_descent(
        H=H,
        gammas=gammas,
        eps=eps,
        dims=dims,
        T=T_kl,
        eta=1.0,
        jitter_log=1e-12,
        tol_tr=tol_tr,
        store_hist=False,
        project_pi=True,
    )

    out2 = same_limit(
        res_kl_eta1,
        res_bgda,
        tol_pi=tol_pi,
        metric="trace",
        require_converged=True,
        tol_e_tr=tol_tr,
    )

    print("[2] KLdescent (eta=1) vs tuned BGDA")
    print("  KL converged:", res_kl_eta1.converged)
    print("  KL final e_tr:", float(res_kl_eta1.e_tr_list[-1]))
    print("  same_limit:", out2["same_limit"])
    print("  dist_pi:", out2["dist_pi"])
    print("  KL gibbs_calls:", int(res_kl_eta1.gibbs_calls))
    print("")

    # ============================================================
    # 3) MD-type Sinkhorn vs tuned BGDA
    # ============================================================
    res_md = md_type_sinkhorn_potential(
        H=H,
        gammas=gammas,
        eps=eps,
        dims=dims,
        T_outer=T_md_outer,
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

    out3 = same_limit(
        res_md,
        res_bgda,
        tol_pi=tol_pi,
        metric="trace",
        require_converged=True,
        tol_e_tr=tol_tr,
    )

    print("[3] MD-type Sinkhorn vs tuned BGDA")
    print("  MD converged:", res_md.converged)
    print("  MD final e_tr:", float(res_md.e_tr_list[-1]))
    print("  same_limit:", out3["same_limit"])
    print("  dist_pi:", out3["dist_pi"])
    print("  MD gibbs_calls:", int(res_md.gibbs_calls))
    print("")

    print("=" * 90)
    print("SUMMARY")
    print(f"  BGDA best_eta={best_eta}, BGDA e_tr={float(res_bgda.e_tr_list[-1])}, BGDA gibbs={int(res_bgda.gibbs_calls)}")
    print(f"  [1] same_limit={out1['same_limit']}  dist_pi={out1['dist_pi']}")
    print(f"  [2] same_limit={out2['same_limit']}  dist_pi={out2['dist_pi']}")
    print(f"  [3] same_limit={out3['same_limit']}  dist_pi={out3['dist_pi']}")
    print("=" * 90)


if __name__ == "__main__":
    main()

