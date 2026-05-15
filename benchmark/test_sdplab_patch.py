# benchmark/test_sdplab_patch.py

from benchmark.benchmark_io import save_eqot_instance, load_eqot_instance
from benchmark.run_sdplab_methods import build_sdplab_qot_regularized_problem
from benchmark.diagnostics import final_diagnostics

import numpy as np
import optax

from sdplab.solvers import run_optax_solver


def main():
    # 1. Generate one tiny instance
    row = save_eqot_instance(
        "instances",
        N=2,
        d=2,
        log_eps=0.0,
        seed=123,
        H_type="random",
        gamma_kind="medium",
    )

    inst = load_eqot_instance(row["instance_file"])

    H = inst["H"]
    gammas = inst["gammas"]
    eps = inst["eps"]
    dims = inst["dims"]

    # 2. Build SDPLab regularized QOT problem
    reg_sdp = build_sdplab_qot_regularized_problem(
        H=H,
        gammas=gammas,
        eps=eps,
        dims=dims,
    )

    base_sdp = reg_sdp.sdp
    N = len(dims)
    d = dims[0]

    init_y = np.zeros((N, d, d), dtype=np.complex128)
    init_dual = base_sdp.dual_from_array(base_sdp.ctx.asarray(init_y))

    # 3. Run a short Adam test with checkpoint_every=1
    info = run_optax_solver(
        reg_sdp,
        init_dual,
        opt=optax.adam(1e-2),
        max_iter=20,
        tol=1e-12,
        verbose=False,
        log_every=5,
        checkpoint_every=1,
        return_primal=True,
        primal_normalized=True,
    )

    print("iterations:", info.iterations)
    print("gibbs_calls:", info.gibbs_calls)
    print("compile_time:", info.compile_time)
    print("run_time:", info.time)
    print("total_time:", info.total_time)
    print("time_per_gibbs_call:", info.time_per_gibbs_call)

    print("primal is None?", info.primal is None)
    print("dual_checkpoints is None?", info.dual_checkpoints is None)
    print("checkpoint_gibbs_calls is None?", info.checkpoint_gibbs_calls is None)

    if info.dual_checkpoints is not None:
        print("dual_checkpoints shape:", np.asarray(info.dual_checkpoints).shape)
        print("checkpoint_gibbs_calls:", np.asarray(info.checkpoint_gibbs_calls))

    # 4. Check final primal density
    pi = np.asarray(info.primal.X)
    pi = (pi + pi.conj().T) / 2
    pi = pi / np.trace(pi).real

    diag = final_diagnostics(
        pi=pi,
        H=H,
        gammas=gammas,
        eps=eps,
        dims=dims,
        tol_F=1e-8,
    )

    print("diagnostic keys:", diag.keys())
    print("final_F_marg:", diag["final_F_marg"])
    print("final_e_tr_max:", diag["final_e_tr_max"])


if __name__ == "__main__":
    main()