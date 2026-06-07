"""Run one qubit mixed benchmark task for a SLURM array.

Default task grid has the four case families:

    N=4, kind=ising
    N=4, kind=random
    N=8, kind=ising
    N=8, kind=random

Each task runs the full formal method set unless --method_group narrows it.
For very tight wall-clock limits, submit separate arrays with
--method_group lbfgs, kl, md1, md2, or md5.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class QubitTask:
    N: int
    kind: str
    seed: int


def parse_csv_ints(spec: str) -> List[int]:
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def parse_csv_strings(spec: str) -> List[str]:
    return [x.strip().lower() for x in spec.split(",") if x.strip()]


def build_tasks(Ns: List[int], kinds: List[str], seeds: List[int]) -> List[QubitTask]:
    return [QubitTask(N=N, kind=kind, seed=seed) for N in Ns for kind in kinds for seed in seeds]


def method_args(method_group: str) -> List[str]:
    if method_group == "all":
        return ["--methods", "lbfgs,kl,md", "--M_list", "1,2,5", "--eta_kl_rules", "eps_over_N,eps"]
    if method_group == "lbfgs":
        return ["--methods", "lbfgs"]
    if method_group == "kl":
        return ["--methods", "kl", "--eta_kl_rules", "eps_over_N,eps"]
    if method_group in {"md1", "md2", "md5"}:
        return ["--methods", "md", "--M_list", method_group[-1]]
    raise ValueError("--method_group must be all, lbfgs, kl, md1, md2, or md5")


def default_task_id() -> Optional[int]:
    raw = os.environ.get("SLURM_ARRAY_TASK_ID")
    return int(raw) if raw is not None and raw.strip() else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Run one HPC qubit mixed benchmark task")
    parser.add_argument("--task_id", type=int, default=None)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--Ns", default="4,8")
    parser.add_argument("--kinds", default="ising,random")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--method_group", default="all", choices=["all", "lbfgs", "kl", "md1", "md2", "md5"])
    parser.add_argument("--T_kl", type=int, default=3000)
    parser.add_argument("--T_md", type=int, default=3000)
    parser.add_argument("--lbfgs_max_iter", type=int, default=1000)
    parser.add_argument("--max_gibbs_calls", type=int, default=None)
    parser.add_argument("--tol_tr", type=float, default=1e-4)
    parser.add_argument("--tol_f", type=float, default=1e-8)
    parser.add_argument("--tol_inner", type=float, default=1e-4)
    parser.add_argument("--outdir", type=Path, default=Path("results") / "hpc_qubit_mixed")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    Ns = parse_csv_ints(args.Ns)
    kinds = parse_csv_strings(args.kinds)
    seeds = parse_csv_ints(args.seeds)
    tasks = build_tasks(Ns, kinds, seeds)

    if args.list:
        for i, task in enumerate(tasks):
            print(f"{i}: N={task.N} kind={task.kind} seed={task.seed}")
        return

    task_id = args.task_id if args.task_id is not None else default_task_id()
    if task_id is None:
        raise ValueError("Provide --task_id or run under SLURM_ARRAY_TASK_ID.")
    if task_id < 0 or task_id >= len(tasks):
        raise IndexError(f"task_id={task_id} outside [0, {len(tasks) - 1}]")

    task = tasks[task_id]
    eps_tag = f"{args.eps:.0e}".replace("-", "m").replace("+", "p")
    out = args.outdir / f"N{task.N}" / f"{task.kind}_mixed_seed{task.seed}_{args.method_group}_eps{eps_tag}.csv"

    cmd = [
        sys.executable,
        "-m",
        "experiments.paper79_reproduction.run_small_qubit_trend",
        "--kinds",
        task.kind,
        "--N",
        str(task.N),
        "--case",
        "mixed",
        "--seeds",
        str(task.seed),
        "--eps",
        str(args.eps),
        "--T_kl",
        str(args.T_kl),
        "--T_md",
        str(args.T_md),
        "--lbfgs_max_iter",
        str(args.lbfgs_max_iter),
        "--tol_tr",
        str(args.tol_tr),
        "--tol_f",
        str(args.tol_f),
        "--tol_inner",
        str(args.tol_inner),
        "--out",
        str(out),
        *method_args(args.method_group),
    ]
    if args.max_gibbs_calls is not None:
        cmd.extend(["--max_gibbs_calls", str(args.max_gibbs_calls)])

    print(" ".join(cmd), flush=True)
    if args.dry_run:
        return
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
