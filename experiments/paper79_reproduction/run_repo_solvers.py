"""Run repo-native solvers on paper79 benchmark instances.

This is intentionally a small bridge script. It lets us test the comparison
pipeline before scaling up to the full paper79 H100/MOSEK experiments.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.SolverofEQOT import md_type_sinkhorn_potential, potential_marginal_kl_descent

from .adapter import PAPER79_ROOT, load_paper79_instance, make_tiny_smoke_instance
from .metrics import summarize_solver_result


def load_ground_truth(index: int) -> Optional[float]:
    path = PAPER79_ROOT / "data" / f"test{int(index)}.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    value = data.get("ground_truth")
    return None if value is None else float(value)


def run_one(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if args.tiny_smoke:
        inst = make_tiny_smoke_instance(seed=args.seed, d=args.smoke_d, N=args.smoke_N)
        ground_truth = None
    else:
        inst = load_paper79_instance(args.index)
        ground_truth = load_ground_truth(args.index)

    rows: List[Dict[str, Any]] = []

    if args.method in ("kl", "all"):
        res = potential_marginal_kl_descent(
            H=inst.H,
            gammas=list(inst.gammas),
            eps=args.eps,
            dims=list(inst.dims),
            T=args.T_kl,
            eta=args.eta_kl,
            eta_rule=args.eta_kl_rule,
            jitter_log=args.jitter,
            tol_tr=args.tol_tr,
            tol_F=args.tol_f,
            store_hist=False,
            project_pi=True,
        )
        rows.append(
            summarize_solver_result(
                label="KL descent",
                res=res,
                H=inst.H,
                gammas=list(inst.gammas),
                dims=list(inst.dims),
                tol_f=args.tol_f,
                tol_tr=args.tol_tr,
                ground_truth=ground_truth,
            )
        )

    if args.method in ("md", "all"):
        res = md_type_sinkhorn_potential(
            H=inst.H,
            gammas=list(inst.gammas),
            eps=args.eps,
            dims=list(inst.dims),
            T_outer=args.T_md,
            tol_tr=args.tol_tr,
            tol_F=args.tol_f,
            jitter=args.jitter,
            eta_inner=1.0,
            M_inner=args.M_inner,
            keep_U_hist=False,
            keep_pi_hist=False,
            tol_inner=args.tol_inner,
            project_pi=True,
        )
        rows.append(
            summarize_solver_result(
                label=f"MD-Sinkhorn (M={args.M_inner})",
                res=res,
                H=inst.H,
                gammas=list(inst.gammas),
                dims=list(inst.dims),
                tol_f=args.tol_f,
                tol_tr=args.tol_tr,
                ground_truth=ground_truth,
            )
        )

    for row in rows:
        row.update(
            {
                "paper79_index": inst.index,
                "paper79_label": inst.label,
                "eps": float(args.eps),
                "dims": "x".join(str(d) for d in inst.dims),
            }
        )

    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser("Run MDforQOT solvers on paper79 instances")
    parser.add_argument("--index", type=int, default=8, help="paper79 test index, 0-based")
    parser.add_argument("--method", choices=["kl", "md", "all"], default="all")
    parser.add_argument("--tiny_smoke", action="store_true", help="run a tiny synthetic paper79-shaped instance")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke_d", type=int, default=2)
    parser.add_argument("--smoke_N", type=int, default=2)
    parser.add_argument("--eps", type=float, default=1e-2)
    parser.add_argument("--T_kl", type=int, default=200)
    parser.add_argument("--T_md", type=int, default=200)
    parser.add_argument("--M_inner", type=int, default=1)
    parser.add_argument("--eta_kl", type=float, default=None)
    parser.add_argument("--eta_kl_rule", choices=["eps_over_N", "eps"], default="eps_over_N")
    parser.add_argument("--tol_f", type=float, default=1e-8)
    parser.add_argument("--tol_tr", type=float, default=1e-8)
    parser.add_argument("--tol_inner", type=float, default=None)
    parser.add_argument("--jitter", type=float, default=1e-10)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results") / "paper79_repo_solvers.csv",
    )
    args = parser.parse_args()

    rows = run_one(args)
    write_csv(args.out, rows)
    for row in rows:
        print(row)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
