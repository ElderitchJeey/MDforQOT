"""Run one warm-start qubit epsilon-stress task for a SLURM array."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from .run_qubit_eps_task import build_tasks, parse_csv_floats
from .run_qubit_mixed_task import parse_csv_ints, parse_csv_strings


def default_task_id() -> Optional[int]:
    raw = os.environ.get("SLURM_ARRAY_TASK_ID")
    return int(raw) if raw is not None and raw.strip() else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Run one warm-start qubit epsilon task")
    parser.add_argument("--task_id", type=int, default=None)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--N", type=int, default=4)
    parser.add_argument("--kinds", default="ising,random")
    parser.add_argument("--eps_values", default="1e-4,1e-8,1e-12")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--eps0", type=float, default=1.0)
    parser.add_argument("--q", type=float, default=10.0)
    parser.add_argument("--max_inner", type=int, default=100000)
    parser.add_argument("--outdir", type=Path, default=Path("results") / "hpc_warm_qubit_eps")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    tasks = build_tasks(parse_csv_strings(args.kinds), parse_csv_floats(args.eps_values), parse_csv_ints(args.seeds))
    if args.list:
        for i, task in enumerate(tasks):
            print(f"{i}: N={args.N} kind={task.kind} seed={task.seed} eps={task.eps}")
        return
    task_id = args.task_id if args.task_id is not None else default_task_id()
    if task_id is None:
        raise ValueError("Provide --task_id or SLURM_ARRAY_TASK_ID.")
    task = tasks[task_id]
    eps_tag = f"{task.eps:.0e}".replace("-", "m").replace("+", "p")
    out = args.outdir / f"N{args.N}" / f"{task.kind}_mixed_seed{task.seed}_warm_eps{eps_tag}.csv"
    cmd = [
        sys.executable,
        "-m",
        "experiments.paper79_reproduction.run_warm_start_comparison",
        "--instance",
        "qubit",
        "--kind",
        task.kind,
        "--N",
        str(args.N),
        "--case",
        "mixed",
        "--seed",
        str(task.seed),
        "--eps_final",
        str(task.eps),
        "--eps0",
        str(args.eps0),
        "--q",
        str(args.q),
        "--max_inner",
        str(args.max_inner),
        "--out",
        str(out),
    ]
    print(" ".join(cmd), flush=True)
    if args.dry_run:
        return
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
