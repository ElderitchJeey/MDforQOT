# experiments/test_md_inner_convergence.py
#
# Small experiment: does the MD inner iteration reduce marginal mismatch and converge?
#
# Run:
#   python -m experiments.sanity.test_md_inner_convergence
#
# Output:
#   figures/inner_md_ei_seed*_i*.png

import os
import time
import numpy as np
import matplotlib.pyplot as plt

from src.instances import gen_marginal, gen_H_random
from src.SolverofEQOT import (
    md_inner_update_i,
    gibbs_state_from_potentials,   # used only to build pi0 consistently
)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def main():
    # -----------------------
    # Config (edit here)
    # -----------------------
    N = 2
    d = 4
    dims = [d] * N

    eps = 0.5
    H_scale = 1.0
    kind = "medium"

    seeds = [0, 1, 2]        # "几个例子"
    M_inner = 180             # inner 迭代次数
    eta_inner = 1.0
    tol_inner = 1e-8         # inner 想达到的 i-th marginal mismatch 阈值

    out_dir = "figures"
    ensure_dir(out_dir)

    print("=" * 96)
    print("Sanity test: MD inner iteration convergence")
    print(f"N={N}, d={d}, eps={eps}, seeds={seeds}")
    print(f"eta_inner={eta_inner}, M_inner={M_inner}, tol_inner={tol_inner}")
    print("=" * 96)

    for seed in seeds:
        rng = np.random.default_rng(seed)

        # Instance
        gammas = [gen_marginal(d=d, rng=rng, kind=kind) for _ in range(N)]
        H = gen_H_random(dims=dims, rng=rng, scale=H_scale)

        # Initial potentials (random but small)
        U_list = [np.zeros((d, d), dtype=complex) for _ in range(N)]

        # Build pi0 (optional but makes the inner test self-contained and reproducible)
        pi0 = gibbs_state_from_potentials(U_list, H, eps, dims, jitter=1e-12, project=True)

        # Test i = 0 and i = 1 separately
        for i in range(N):
            # IMPORTANT: start each i-test from the same initial point (U_list, pi0)
            res = md_inner_update_i(
                i=i,
                U_list=U_list,
                H=H,
                gamma_i=gammas[i],
                eps=eps,
                dims=dims,
                pi0=pi0,
                eta_inner=eta_inner,
                M_inner=M_inner,
                tol_inner=tol_inner,
                jitter=1e-12,
                project_pi=True,
                reset_counter=True,
                keep_history=True,
            )

            e = np.array(res.e_i_tr_list, dtype=float)
            it = np.arange(len(e))

            # Print summary
            e0 = float(e[0]) if len(e) > 0 else float("nan")
            eT = float(e[-1]) if len(e) > 0 else float("nan")
            print(
                f"[seed={seed} i={i}] "
                f"e0={e0:.3e} -> eT={eT:.3e}, "
                f"converged={res.converged}, gibbs_calls={res.gibbs_calls}"
            )

            # Plot e_i vs inner iteration
            plt.figure()
            plt.plot(it, e, marker="o", linewidth=2)
            plt.yscale("log")
            plt.xlabel("inner iteration")
            plt.ylabel(r"$e_i = \|\mathrm{Tr}_{\neq i}(\pi) - \gamma_i\|_1$")
            plt.title(f"MD inner convergence (seed={seed}, i={i}, eps={eps})")
            plt.grid(True, which="both", alpha=0.3)
            plt.tight_layout()

            out_png = os.path.join(out_dir, f"inner_md_ei_seed{seed}_i{i}.png")
            plt.savefig(out_png, dpi=200)
            print(f"  saved: {out_png}")

    print("\nDone.")


if __name__ == "__main__":
    main()
