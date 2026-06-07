"""Run one warm-start qubit mixed task for a SLURM array."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from .run_qubit_mixed_task import build_tasks, parse_csv_ints, parse_csv_strings


def default_task_id() -> Optional[int]:
    raw = os.environ.get("SLURM_ARRAY_TASK_ID")
    return int(raw) if raw is not None and raw.strip() else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Run one warm-start qubit mixed task")
    parser.add_argument("--task_id", type=int, default=None)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--Ns", default="4,8")
    parser.add_argument("--kinds", default="ising,random")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--eps_final", type=float, default=1e-3)
    parser.add_argument("--eps0", type=float, default=1.0)
    parser.add_argument("--q", type=float, default=10.0)
    parser.add_argument("--max_inner", type=int, default=100000)
    parser.add_argument("--max_gibbs_calls", type=int, default=None)
    parser.add_argument("--inner_tol", type=float, default=1e-2)
    parser.add_argument("--final_tol", type=float, default=1e-4)
    parser.add_argument("--tol_tr", type=float, default=1e-4)
    parser.add_argument("--tol_F", type=float, default=1e-8)
    parser.add_argument("--outdir", type=Path, default=Path("results") / "hpc_warm_qubit_mixed")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    tasks = build_tasks(parse_csv_ints(args.Ns), parse_csv_strings(args.kinds), parse_csv_ints(args.seeds))
    if args.list:
        for i, task in enumerate(tasks):
            print(f"{i}: N={task.N} kind={task.kind} seed={task.seed}")
        return
    task_id = args.task_id if args.task_id is not None else default_task_id()
    if task_id is None:
        raise ValueError("Provide --task_id or SLURM_ARRAY_TASK_ID.")
    task = tasks[task_id]
    eps_tag = f"{args.eps_final:.0e}".replace("-", "m").replace("+", "p")
    out = args.outdir / f"N{task.N}" / f"{task.kind}_mixed_seed{task.seed}_warm_eps{eps_tag}.csv"
    cmd = [
        sys.executable,
        "-m",
        "experiments.paper79_reproduction.run_warm_start_comparison",
        "--instance",
        "qubit",
        "--kind",
        task.kind,
        "--N",
        str(task.N),
        "--case",
        "mixed",
        "--seed",
        str(task.seed),
        "--eps_final",
        str(args.eps_final),
        "--eps0",
        str(args.eps0),
        "--q",
        str(args.q),
        "--max_inner",
        str(args.max_inner),
        "--inner_tol",
        str(args.inner_tol),
        "--final_tol",
        str(args.final_tol),
        "--tol_tr",
        str(args.tol_tr),
        "--tol_F",
        str(args.tol_F),
        "--out",
        str(out),
    ]
    if args.max_gibbs_calls is not None:
        cmd.extend(["--max_gibbs_calls", str(args.max_gibbs_calls)])
    print(" ".join(cmd), flush=True)
    if args.dry_run:
        return
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
