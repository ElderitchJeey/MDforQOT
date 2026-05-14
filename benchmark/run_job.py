# benchmark/run_job.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from benchmark.benchmark_io import load_eqot_instance
from benchmark.diagnostics import final_diagnostics
from benchmark.result_io import save_summary, save_history
from benchmark.run_my_methods import run_my_method
from benchmark.run_sdplab_methods import run_sdplab_adam


def _none_if_nan(x: Any):
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except TypeError:
        pass
    if x == "":
        return None
    return x


def _float_or_none(x: Any):
    x = _none_if_nan(x)
    if x is None:
        return None
    return float(x)


def _int_or_none(x: Any):
    x = _none_if_nan(x)
    if x is None:
        return None
    return int(float(x))


def _get_time_sec(res) -> float:
    times = getattr(res, "times", None)
    if times is not None and len(times) > 0:
        return float(times[-1])
    return float(getattr(res, "time", 0.0) or 0.0)


def _get_compile_time_sec(res) -> float:
    return float(getattr(res, "compile_time", 0.0) or 0.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", default="manifests/jobs.csv")
    parser.add_argument("--job-index", type=int, required=True)
    args = parser.parse_args()

    jobs = pd.read_csv(args.jobs)
    row = jobs.iloc[int(args.job_index)]

    inst = load_eqot_instance(row["instance_file"])

    H = inst["H"]
    gammas = inst["gammas"]
    eps = float(inst["eps"])
    dims = inst["dims"]

    method = str(row["method"])
    tol_F = float(row["tol_F"])
    max_gibbs_calls = int(row["max_gibbs_calls"])

    M_inner = _int_or_none(row.get("M_inner", None))
    lr = _float_or_none(row.get("lr", None))
    checkpoint_every = _int_or_none(row.get("checkpoint_every", None))

    if method in {"KL", "MD", "BGDA"}:
        res = run_my_method(
            method=method,
            H=H,
            gammas=gammas,
            eps=eps,
            dims=dims,
            tol_F=tol_F,
            max_gibbs_calls=max_gibbs_calls,
            M_inner=M_inner,
            store_hist=True,
        )

    elif method == "SDPLab-Adam":
        res = run_sdplab_adam(
            H=H,
            gammas=gammas,
            eps=eps,
            dims=dims,
            tol_F=tol_F,
            max_gibbs_calls=max_gibbs_calls,
            lr=1e-2 if lr is None else float(lr),
            checkpoint_every=checkpoint_every if checkpoint_every is not None else 10,
        )

    else:
        raise ValueError(f"Unknown method: {method}")

    diag = final_diagnostics(
        pi=res.pi,
        H=H,
        gammas=gammas,
        eps=eps,
        dims=dims,
        tol_F=tol_F,
    )

    output_prefix = Path(str(row["output_prefix"]))
    summary_path = Path(str(output_prefix) + "_summary.json")
    history_path = Path(str(output_prefix) + "_history.csv")

    gibbs_calls = int(getattr(res, "gibbs_calls", 0) or 0)
    time_sec = _get_time_sec(res)
    compile_time_sec = _get_compile_time_sec(res)
    total_time_sec = time_sec + compile_time_sec

    F_list = list(getattr(res, "F_list", []) or [])
    e_tr_list = list(getattr(res, "e_tr_list", []) or [])
    gibbs_calls_list = list(getattr(res, "gibbs_calls_list", []) or [])

    if len(gibbs_calls_list) != len(F_list):
        if len(F_list) == 1:
            gibbs_calls_list = [gibbs_calls]
        else:
            gibbs_calls_list = list(np.round(np.linspace(0, gibbs_calls, len(F_list))).astype(int))

    summary = {
        "job_id": str(row["job_id"]),
        "sample_id": str(inst["sample_id"]),
        "method": method,
        "M_inner": M_inner,
        "optimizer": _none_if_nan(row.get("optimizer", None)),
        "lr": lr,
        "N": int(inst["N"]),
        "d": int(inst["d"]),
        "dims": [int(x) for x in dims],
        "eps": eps,
        "log_eps": float(inst["log_eps"]),
        "seed": int(inst["seed"]),
        "tol_F": tol_F,
        "max_gibbs_calls": max_gibbs_calls,
        "checkpoint_every": checkpoint_every,
        "converged": bool(getattr(res, "converged", False)),
        "hit": bool(diag["hit"]),
        "gibbs_calls": gibbs_calls,
        "iterations": max(0, len(F_list) - 1),
        "time_sec": time_sec,
        "compile_time_sec": compile_time_sec,
        "total_time_sec": total_time_sec,
        "time_per_gibbs": time_sec / float(max(1, gibbs_calls)),
        **diag,
    }

    save_summary(summary, summary_path)

    save_history(
        history_path,
        gibbs_calls=gibbs_calls_list,
        F_list=F_list,
        e_tr_list=e_tr_list,
        times=list(getattr(res, "times", []) or [time_sec]),
        dual_obj=getattr(res, "dual_obj", None),
        grad_norm=getattr(res, "grad_norm", None),
    )

    print(
        f"[{summary['job_id']}] {method} "
        f"M={M_inner} hit={summary['hit']} "
        f"gibbs={summary['gibbs_calls']} "
        f"F={summary['final_F_marg']:.3e} "
        f"e={summary['final_e_tr_max']:.3e} "
        f"time={summary['time_sec']:.3e}s"
    )


if __name__ == "__main__":
    main()