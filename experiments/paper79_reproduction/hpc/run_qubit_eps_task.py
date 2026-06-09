"""Run one qubit mixed epsilon-stress task for a SLURM array."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .run_qubit_mixed_task import method_args, parse_csv_ints, parse_csv_strings


@dataclass(frozen=True)
class EpsTask:
    kind: str
    eps: float
    seed: int


def parse_csv_floats(spec: str) -> List[float]:
    return [float(x.strip()) for x in spec.split(",") if x.strip()]


def build_tasks(kinds: List[str], eps_values: List[float], seeds: List[int]) -> List[EpsTask]:
    return [EpsTask(kind=kind, eps=eps, seed=seed) for kind in kinds for eps in eps_values for seed in seeds]


def default_task_id() -> Optional[int]:
    raw = os.environ.get("SLURM_ARRAY_TASK_ID")
    return int(raw) if raw is not None and raw.strip() else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Run one HPC qubit mixed epsilon task")
    parser.add_argument("--task_id", type=int, default=None)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--N", type=int, default=4)
    parser.add_argument("--kinds", default="ising,random")
    parser.add_argument("--eps_values", default="1e-4,1e-8,1e-12")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--method_group", default="all", choices=["all", "lbfgs", "kl", "md1", "md2", "md5"])
    parser.add_argument("--T_kl", type=int, default=3000)
    parser.add_argument("--T_md", type=int, default=3000)
    parser.add_argument("--lbfgs_max_iter", type=int, default=1000)
    parser.add_argument("--max_gibbs_calls", type=int, default=None)
    parser.add_argument("--tol_tr", type=float, default=1e-4)
    parser.add_argument("--tol_f", type=float, default=1e-8)
    parser.add_argument("--tol_inner", type=float, default=1e-4)
    parser.add_argument("--outdir", type=Path, default=Path("results") / "hpc_qubit_eps")
    parser.add_argument("--save_final_state", action="store_true")
    parser.add_argument("--state_dir", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    kinds = parse_csv_strings(args.kinds)
    eps_values = parse_csv_floats(args.eps_values)
    seeds = parse_csv_ints(args.seeds)
    tasks = build_tasks(kinds, eps_values, seeds)

    if args.list:
        for i, task in enumerate(tasks):
            print(f"{i}: N={args.N} kind={task.kind} seed={task.seed} eps={task.eps}")
        return

    task_id = args.task_id if args.task_id is not None else default_task_id()
    if task_id is None:
        raise ValueError("Provide --task_id or run under SLURM_ARRAY_TASK_ID.")
    if task_id < 0 or task_id >= len(tasks):
        raise IndexError(f"task_id={task_id} outside [0, {len(tasks) - 1}]")

    task = tasks[task_id]
    eps_tag = f"{task.eps:.0e}".replace("-", "m").replace("+", "p")
    out = args.outdir / f"N{args.N}" / f"{task.kind}_mixed_seed{task.seed}_{args.method_group}_eps{eps_tag}.csv"

    cmd = [
        sys.executable,
        "-m",
        "experiments.paper79_reproduction.run_small_qubit_trend",
        "--kinds",
        task.kind,
        "--N",
        str(args.N),
        "--case",
        "mixed",
        "--seeds",
        str(task.seed),
        "--eps",
        str(task.eps),
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
    if args.save_final_state:
        cmd.append("--save_final_state")
        if args.state_dir is not None:
            cmd.extend(["--state_dir", str(args.state_dir / f"N{args.N}")])
    print(" ".join(cmd), flush=True)
    if args.dry_run:
        return
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
