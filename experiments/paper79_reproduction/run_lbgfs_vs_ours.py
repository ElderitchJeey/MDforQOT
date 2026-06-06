"""Compare paper79 entropy L-BFGS against MDforQOT solvers.

This runner focuses on the comparison most aligned with this repo's methods:

    L-BFGS on the von Neumann entropy-regularized dual
    vs
    KL descent and MD-Sinkhorn with M_inner in {1, 2, 5}

Two experiment presets are provided:

    main   : moderate entropy parameters suitable for the repo-native solvers
    stress : paper79's near-unregularized epsilon=1e-12 regime

The file name intentionally follows the user's requested spelling
``run_lbgfs_vs_ours.py``; all report labels use the standard spelling L-BFGS.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from src.SolverofEQOT import md_type_sinkhorn_potential, potential_marginal_kl_descent

from .adapter import ensure_paper79_import_path, load_paper79_instance, make_tiny_smoke_instance
from .metrics import summarize_solver_result


PAPER79_INDICES = list(range(12))
MAIN_EPS = [1e-2, 1e-3]
STRESS_EPS = [1e-4, 1e-8, 1e-12]
DEFAULT_M_LIST = [1, 2, 5]


def parse_csv_ints(spec: str) -> List[int]:
    spec = spec.strip().lower()
    if spec == "all":
        return PAPER79_INDICES.copy()
    out: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            start, stop = int(a), int(b)
            step = 1 if start <= stop else -1
            out.extend(range(start, stop + step, step))
        else:
            out.append(int(part))
    return out


def parse_csv_floats(spec: str) -> List[float]:
    return [float(x.strip()) for x in spec.split(",") if x.strip()]


def parse_csv_strings(spec: str) -> List[str]:
    return [x.strip() for x in spec.split(",") if x.strip()]


def kl_method_label(*, eta: Optional[float], eta_rule: str) -> str:
    if eta is not None:
        return f"KL descent (eta={float(eta):.3g})"
    if eta_rule == "eps":
        return "KL descent (eta=eps)"
    return "KL descent (eta=eps/N)"


def tol_label(tol: float) -> str:
    return f"{float(tol):.0e}".replace("-", "m").replace("+", "p")


def first_hit_index(values: Sequence[float], tol: float) -> int:
    arr = np.asarray(values, dtype=float)
    hit = np.where(arr <= float(tol))[0]
    return int(hit[0]) if hit.size else -1


def parse_experiments(spec: str) -> List[str]:
    vals = [x.strip().lower() for x in spec.split(",") if x.strip()]
    if "all" in vals:
        return ["main", "stress"]
    for val in vals:
        if val not in {"main", "stress"}:
            raise ValueError("experiment must be main, stress, or all")
    return vals


def make_result_from_lbfgs(
    *,
    state: Dict[str, Any],
    pi: np.ndarray,
    elapsed: float,
    H: np.ndarray,
    gammas,
    dims,
) -> SimpleNamespace:
    """Convert paper79 L-BFGS output to the repo metric protocol."""

    from src.SolverofEQOT import F_marg, marginal_trace_errors

    f = float(F_marg(pi, gammas, dims))
    e = float(np.max(marginal_trace_errors(pi, gammas, dims)))
    n_iters = int(state["n_iters"])
    tol_reached = bool(state["tol_reached"])

    return SimpleNamespace(
        F_list=[f],
        e_tr_list=[e],
        per_i_tr_list=[marginal_trace_errors(pi, gammas, dims)],
        times=[float(elapsed)],
        pi=pi,
        U_list=list(state["params"]),
        converged=tol_reached,
        n_iters=n_iters,
        gibbs_calls=n_iters,
        gibbs_calls_list=[n_iters],
    )


def run_lbfgs_entropy(
    *,
    H: np.ndarray,
    gammas,
    dims,
    eps: float,
    max_iter: int,
    tol: float,
) -> SimpleNamespace:
    """Run paper79's entropy L-BFGS and reconstruct the primal Gibbs state."""

    ensure_paper79_import_path()
    import jax.numpy as jnp  # type: ignore
    from solvers import run_lbfgs  # type: ignore
    from utils.entropy import reconstruct_primal  # type: ignore

    cost_matrix = jnp.asarray(np.real_if_close(H))
    ptraces = tuple(jnp.asarray(np.real_if_close(g)) for g in gammas)

    start = time.time()
    state = run_lbfgs(
        cost_matrix=cost_matrix,
        ptraces=ptraces,
        reg=float(eps),
        max_iter=int(max_iter),
        tol=float(tol),
        verbose=False,
        log_every=50,
    )
    pi = np.asarray(reconstruct_primal(cost_matrix, state["params"], float(eps)), dtype=complex)
    elapsed = time.time() - start
    return make_result_from_lbfgs(
        state=state,
        pi=pi,
        elapsed=elapsed,
        H=H,
        gammas=gammas,
        dims=dims,
    )


def row_with_context(
    row: Dict[str, Any],
    *,
    experiment: str,
    index: int,
    label: str,
    eps: float,
    dims: Sequence[int],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    prefix = {
        "experiment": experiment,
        "paper79_index": index,
        "paper79_label": label,
        "eps": float(eps),
        "dims": "x".join(str(int(d)) for d in dims),
    }
    if extra:
        prefix.update(extra)
    prefix.update(row)
    return prefix


def run_instance_for_eps(args: argparse.Namespace, *, experiment: str, index: int, eps: float) -> List[Dict[str, Any]]:
    if args.tiny_smoke:
        inst = make_tiny_smoke_instance(seed=args.seed, d=args.smoke_d, N=args.smoke_N)
    else:
        inst = load_paper79_instance(index)

    H = inst.H
    gammas = list(inst.gammas)
    dims = list(inst.dims)
    rows: List[Dict[str, Any]] = []
    result_entries: List[Dict[str, Any]] = []

    if "lbfgs" in args.methods:
        try:
            res = run_lbfgs_entropy(
                H=H,
                gammas=gammas,
                dims=dims,
                eps=eps,
                max_iter=args.lbfgs_max_iter,
                tol=args.lbfgs_tol,
            )
            row = summarize_solver_result(
                label="L-BFGS entropy dual",
                res=res,
                H=H,
                gammas=gammas,
                dims=dims,
                tol_f=args.tol_f,
                tol_tr=args.tol_tr,
            )
            row["status"] = "ok"
        except Exception as exc:
            row = {"method": "L-BFGS entropy dual", "status": f"error: {type(exc).__name__}: {exc}"}
            res = None
        result_entries.append({"row": row, "res": res, "M_inner": None})

    if "kl" in args.methods:
        kl_rules = [args.eta_kl_rule] if args.eta_kl is not None else getattr(args, "eta_kl_rules", [args.eta_kl_rule])
        for eta_rule in kl_rules:
            label = kl_method_label(eta=args.eta_kl, eta_rule=eta_rule)
            try:
                res = potential_marginal_kl_descent(
                    H=H,
                    gammas=gammas,
                    eps=eps,
                    dims=dims,
                    T=args.T_kl,
                    eta=args.eta_kl,
                    eta_rule=eta_rule,
                    jitter_log=args.jitter,
                    tol_tr=args.tol_tr,
                    tol_F=args.tol_f,
                    store_hist=False,
                    project_pi=True,
                )
                row = summarize_solver_result(
                    label=label,
                    res=res,
                    H=H,
                    gammas=gammas,
                    dims=dims,
                    tol_f=args.tol_f,
                    tol_tr=args.tol_tr,
                )
                row["status"] = "ok"
            except Exception as exc:
                row = {"method": label, "status": f"error: {type(exc).__name__}: {exc}"}
                res = None
            result_entries.append({"row": row, "res": res, "M_inner": None})

    if "md" in args.methods:
        for M_inner in args.M_list:
            try:
                res = md_type_sinkhorn_potential(
                    H=H,
                    gammas=gammas,
                    eps=eps,
                    dims=dims,
                    T_outer=args.T_md,
                    tol_tr=args.tol_tr,
                    tol_F=args.tol_f,
                    jitter=args.jitter,
                    eta_inner=1.0,
                    M_inner=int(M_inner),
                    keep_U_hist=False,
                    keep_pi_hist=False,
                    tol_inner=args.tol_inner,
                    project_pi=True,
                )
                row = summarize_solver_result(
                    label=f"MD-Sinkhorn (M={int(M_inner)})",
                    res=res,
                    H=H,
                    gammas=gammas,
                    dims=dims,
                    tol_f=args.tol_f,
                    tol_tr=args.tol_tr,
                )
                row["status"] = "ok"
            except Exception as exc:
                row = {
                    "method": f"MD-Sinkhorn (M={int(M_inner)})",
                    "status": f"error: {type(exc).__name__}: {exc}",
                }
                res = None
            result_entries.append({"row": row, "res": res, "M_inner": int(M_inner)})

    add_tolerance_grid_metrics(result_entries, args)
    add_cross_method_consistency(result_entries, args)
    for entry in result_entries:
        extra = {}
        if entry["M_inner"] is not None:
            extra["M_inner"] = entry["M_inner"]
        rows.append(
            row_with_context(
                entry["row"],
                experiment=experiment,
                index=inst.index,
                label=inst.label,
                eps=eps,
                dims=dims,
                extra=extra,
            )
        )
    return rows


def add_tolerance_grid_metrics(entries: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    """Add hit columns for several reporting tolerances."""

    for entry in entries:
        row = entry["row"]
        res = entry.get("res")
        F = list(getattr(res, "F_list", []) or []) if res is not None else []
        e = list(getattr(res, "e_tr_list", []) or []) if res is not None else []
        gibbs = list(getattr(res, "gibbs_calls_list", []) or []) if res is not None else []

        for tol in args.tol_f_grid:
            lab = tol_label(tol)
            idx = first_hit_index(F, tol) if F else -1
            row[f"hit_F_le_{lab}"] = bool(idx >= 0) if F else ""
            row[f"hit_F_iter_le_{lab}"] = idx if F else ""
            row[f"hit_F_gibbs_le_{lab}"] = int(gibbs[idx]) if idx >= 0 and len(gibbs) == len(F) else ""

        for tol in args.tol_tr_grid:
            lab = tol_label(tol)
            idx = first_hit_index(e, tol) if e else -1
            row[f"hit_tr_le_{lab}"] = bool(idx >= 0) if e else ""
            row[f"hit_tr_iter_le_{lab}"] = idx if e else ""
            row[f"hit_tr_gibbs_le_{lab}"] = int(gibbs[idx]) if idx >= 0 and len(gibbs) == len(e) else ""


def add_cross_method_consistency(entries: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    """Add final-coupling and objective consistency columns in-place."""

    from src.metrics import pi_distance, same_limit

    successful = [entry for entry in entries if entry.get("res") is not None and entry["row"].get("status") == "ok"]
    ref = next((entry for entry in successful if entry["row"].get("method") == "L-BFGS entropy dual"), None)

    if ref is None:
        for entry in entries:
            entry["row"]["same_limit_to_lbfgs"] = ""
            entry["row"]["dist_pi_to_lbfgs"] = ""
            entry["row"]["objective_gap_to_lbfgs"] = ""
        return

    ref_res = ref["res"]
    ref_cost = ref["row"].get("final_cost")

    for entry in entries:
        row = entry["row"]
        res = entry.get("res")
        if res is None or row.get("status") != "ok":
            row["same_limit_to_lbfgs"] = ""
            row["dist_pi_to_lbfgs"] = ""
            row["objective_gap_to_lbfgs"] = ""
            continue

        if res is ref_res:
            row["same_limit_to_lbfgs"] = True
            row["dist_pi_to_lbfgs"] = 0.0
            row["objective_gap_to_lbfgs"] = 0.0
            continue

        try:
            out = same_limit(
                res,
                ref_res,
                tol_pi=args.tol_pi,
                metric="trace",
                require_converged=args.same_limit_require_converged,
                tol_e_tr=args.tol_tr if args.same_limit_require_converged else None,
            )
            row["same_limit_to_lbfgs"] = bool(out["same_limit"])
            row["dist_pi_to_lbfgs"] = float(out["dist_pi"])
        except Exception:
            try:
                from src.metrics import final_pi

                row["dist_pi_to_lbfgs"] = float(pi_distance(final_pi(res), final_pi(ref_res), metric="trace"))
                row["same_limit_to_lbfgs"] = bool(row["dist_pi_to_lbfgs"] <= args.tol_pi)
            except Exception as exc:
                row["same_limit_to_lbfgs"] = f"error: {type(exc).__name__}: {exc}"
                row["dist_pi_to_lbfgs"] = ""

        if ref_cost is not None and row.get("final_cost") is not None:
            row["objective_gap_to_lbfgs"] = abs(float(row["final_cost"]) - float(ref_cost))
        else:
            row["objective_gap_to_lbfgs"] = ""


def fieldnames_union(rows: Iterable[Dict[str, Any]]) -> List[str]:
    names: List[str] = []
    seen = set()
    preferred = [
        "experiment",
        "paper79_index",
        "paper79_label",
        "eps",
        "dims",
        "method",
        "M_inner",
        "status",
        "converged",
        "iters",
        "time_sec",
        "gibbs_calls",
        "final_cost",
        "final_F_marg",
        "final_e_tr",
        "hit_F_iter",
        "hit_tr_iter",
        "hit_F_gibbs",
        "hit_tr_gibbs",
        "same_limit_to_lbfgs",
        "dist_pi_to_lbfgs",
        "objective_gap_to_lbfgs",
    ]
    for name in preferred:
        seen.add(name)
        names.append(name)
    for row in rows:
        for name in row:
            if name not in seen:
                seen.add(name)
                names.append(name)
    return names


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fieldnames_union(rows)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def default_checkpoint_path(out: Path) -> Path:
    return out.with_suffix(out.suffix + ".partial.jsonl")


def reset_checkpoint(path: Optional[Path]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def append_checkpoint(path: Optional[Path], rows: List[Dict[str, Any]]) -> None:
    if path is None or not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        f.flush()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Compare paper79 L-BFGS against MDforQOT methods")
    parser.add_argument("--experiment", default="main", help="main, stress, or all")
    parser.add_argument("--indices", default="all", help="all, comma list, or range like 0-3")
    parser.add_argument("--methods", default="lbfgs,kl,md", help="comma list from lbfgs,kl,md")
    parser.add_argument("--main_eps", default=",".join(str(x) for x in MAIN_EPS))
    parser.add_argument("--stress_eps", default=",".join(str(x) for x in STRESS_EPS))
    parser.add_argument("--M_list", default=",".join(str(x) for x in DEFAULT_M_LIST))

    parser.add_argument("--T_kl", type=int, default=1000)
    parser.add_argument("--T_md", type=int, default=1000)
    parser.add_argument("--eta_kl", type=float, default=None)
    parser.add_argument("--eta_kl_rule", choices=["eps_over_N", "eps"], default="eps_over_N")
    parser.add_argument("--eta_kl_rules", default="eps_over_N,eps")
    parser.add_argument("--lbfgs_max_iter", type=int, default=1000)
    parser.add_argument("--lbfgs_tol", type=float, default=1e-6)
    parser.add_argument("--tol_f", type=float, default=1e-8)
    parser.add_argument("--tol_tr", type=float, default=1e-8)
    parser.add_argument("--tol_f_grid", default="1e-3,1e-4,1e-5")
    parser.add_argument("--tol_tr_grid", default="1e-3,1e-4,1e-5")
    parser.add_argument("--tol_pi", type=float, default=1e-6)
    parser.add_argument("--same_limit_require_converged", action="store_true")
    parser.add_argument("--tol_inner", type=float, default=1e-4)
    parser.add_argument("--jitter", type=float, default=1e-10)

    parser.add_argument("--tiny_smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke_d", type=int, default=2)
    parser.add_argument("--smoke_N", type=int, default=2)

    parser.add_argument("--out", type=Path, default=Path("results") / "paper79_lbgfs_vs_ours.csv")
    parser.add_argument("--checkpoint_jsonl", type=Path, default=None)
    parser.add_argument("--no_checkpoint", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    args.methods = [x.strip().lower() for x in args.methods.split(",") if x.strip()]
    args.M_list = parse_csv_ints(args.M_list)
    args.eta_kl_rules = parse_csv_strings(args.eta_kl_rules)
    for eta_rule in args.eta_kl_rules:
        if eta_rule not in {"eps_over_N", "eps"}:
            raise ValueError("--eta_kl_rules entries must be eps_over_N or eps")
    args.tol_f_grid = parse_csv_floats(args.tol_f_grid)
    args.tol_tr_grid = parse_csv_floats(args.tol_tr_grid)
    experiments = parse_experiments(args.experiment)
    indices = [0] if args.tiny_smoke else parse_csv_ints(args.indices)
    eps_by_experiment = {
        "main": parse_csv_floats(args.main_eps),
        "stress": parse_csv_floats(args.stress_eps),
    }
    checkpoint_path = None if args.no_checkpoint else (args.checkpoint_jsonl or default_checkpoint_path(args.out))
    reset_checkpoint(checkpoint_path)

    rows: List[Dict[str, Any]] = []
    for experiment in experiments:
        for eps in eps_by_experiment[experiment]:
            for index in indices:
                print(f"[Run] experiment={experiment} index={index} eps={eps}", flush=True)
                batch = run_instance_for_eps(args, experiment=experiment, index=index, eps=eps)
                rows.extend(batch)
                append_checkpoint(checkpoint_path, batch)

    write_csv(args.out, rows)
    for row in rows:
        print(row)
    print(f"Wrote {args.out}")
    if checkpoint_path is not None:
        print(f"Wrote partial checkpoint {checkpoint_path}")


if __name__ == "__main__":
    main()
