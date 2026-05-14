# benchmark/make_samples.py

from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

from benchmark.benchmark_io import save_eqot_instance


def main():
    rows = []
    out_dir = Path("instances")
    seeds = range(10)

    # A1 / C1: dimension robustness, N=2, eps=1, d=2..10
    for d in range(2, 11):
        for seed in seeds:
            rows.append(save_eqot_instance(
                out_dir,
                N=2,
                d=d,
                log_eps=0.0,
                seed=seed,
                H_type="random",
                gamma_kind="medium",
            ))

    # D1: N robustness, d=3, eps=1, N=2..6
    for N in range(2, 7):
        for seed in seeds:
            rows.append(save_eqot_instance(
                out_dir,
                N=N,
                d=3,
                log_eps=0.0,
                seed=seed,
                H_type="random",
                gamma_kind="medium",
            ))

    # E1: eps robustness, N=2, d=5, log eps in [-2,0]
    log_eps_grid = np.linspace(-2.0, 0.0, 9)
    for log_eps in log_eps_grid:
        for seed in seeds:
            rows.append(save_eqot_instance(
                out_dir,
                N=2,
                d=5,
                log_eps=float(log_eps),
                seed=seed,
                H_type="random",
                gamma_kind="medium",
            ))

    # E2: multi-marginal eps robustness, choose N=4,d=3
    for log_eps in log_eps_grid:
        for seed in seeds:
            rows.append(save_eqot_instance(
                out_dir,
                N=4,
                d=3,
                log_eps=float(log_eps),
                seed=seed,
                H_type="random",
                gamma_kind="medium",
            ))

    Path("manifests").mkdir(exist_ok=True)
    pd.DataFrame(rows).drop_duplicates("sample_id").to_csv(
        "manifests/samples.csv",
        index=False,
    )


if __name__ == "__main__":
    main()