# benchmark/make_samples.py

from pathlib import Path
import pandas as pd

from benchmark.benchmark_io import save_eqot_instance


def main():
    rows = []
    out_dir = Path("instances")

    rows.append(save_eqot_instance(
        out_dir,
        N=2,
        d=3,
        log_eps=0.0,      # eps = 1
        seed=0,
        H_type="random",
        gamma_kind="medium",
    ))

    Path("manifests").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(
        "manifests/samples.csv",
        index=False,
    )


if __name__ == "__main__":
    main()