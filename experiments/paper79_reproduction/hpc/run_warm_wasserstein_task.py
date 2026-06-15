"""Run one warm-start Wasserstein task for a SLURM array."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from .run_wasserstein_task import parse_csv_strings


def default_task_id() -> Optional[int]:
    raw = os.environ.get("SLURM_ARRAY_TASK_ID")
    return int(raw) if raw is not None and raw.strip() else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Run one warm-start Wasserstein task")
    parser.add_argument("--task_id", type=int, default=None)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--cases", default="WP,WM,WC,WG")
    parser.add_argument("--d", type=int, default=20)
    parser.add_argument("--eps_final", type=float, default=1e-3)
    parser.add_argument("--eps0", type=float, default=1.0)
    parser.add_argument("--q", type=float, default=10.0)
    parser.add_argument("--max_inner", type=int, default=100000)
    parser.add_argument("--max_gibbs_calls", type=int, default=None)
    parser.add_argument("--inner_tol", type=float, default=1e-2)
    parser.add_argument("--final_tol", type=float, default=1e-4)
    parser.add_argument("--tol_tr", type=float, default=1e-4)
    parser.add_argument("--tol_F", type=float, default=1e-8)
    parser.add_argument("--outdir", type=Path, default=Path("results") / "hpc_warm_wasserstein")
    parser.add_argument("--save_final_state", action="store_true")
    parser.add_argument("--state_dir", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cases = parse_csv_strings(args.cases)
    if args.list:
        for i, case in enumerate(cases):
            print(f"{i}: case={case} d={args.d}")
        return
    task_id = args.task_id if args.task_id is not None else default_task_id()
    if task_id is None:
        raise ValueError("Provide --task_id or SLURM_ARRAY_TASK_ID.")
    case = cases[task_id]
    eps_tag = f"{args.eps_final:.0e}".replace("-", "m").replace("+", "p")
    out = args.outdir / f"d{args.d}" / f"{case}_warm_eps{eps_tag}.csv"
    cmd = [
        sys.executable,
        "-m",
        "experiments.paper79_reproduction.run_warm_start_comparison",
        "--instance",
        "wasserstein",
        "--d",
        str(args.d),
        "--wasserstein_case",
        case,
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
    if args.save_final_state:
        cmd.append("--save_final_state")
    if args.state_dir is not None:
        cmd.extend(["--state_dir", str(args.state_dir)])
    print(" ".join(cmd), flush=True)
    if args.dry_run:
        return
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
