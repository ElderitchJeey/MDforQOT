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
import time
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.experiment_utils import (
    append_checkpoint,
    default_checkpoint_path,
    first_hit_index,
    parse_csv_floats,
    parse_csv_ints,
    parse_csv_strings,
    reset_checkpoint,
    save_final_state,
    tol_label,
    write_csv,
)
from src.SolverofEQOT import md_type_sinkhorn_potential, potential_marginal_kl_descent

from .adapter import ensure_paper79_import_path, load_paper79_instance, make_tiny_smoke_instance
from .metrics import summarize_solver_result


PAPER79_INDICES = list(range(12))
MAIN_EPS = [1e-2, 1e-3]
STRESS_EPS = [1e-4, 1e-8, 1e-12]
DEFAULT_M_LIST = [1, 2, 5]
RUNNER_PREFERRED_FIELDS = [
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
    "gibbs_calls_optimizer",
    "gibbs_calls_unified",
    "final_cost",
    "final_entropic_primal",
    "final_linear_cost",
    "final_entropy",
    "final_dual_value",
    "final_primal_dual_gap",
    "final_state_path",
    "final_F_marg",
    "final_e_tr",
    "hit_F_iter",
    "hit_tr_iter",
    "hit_F_gibbs",
    "hit_tr_gibbs",
    "hit_tr_gibbs_unified",
    "same_limit_to_lbfgs",
    "dist_pi_to_lbfgs",
    "objective_gap_to_lbfgs",
    "dual_gap_to_lbfgs",
]


def kl_method_label(*, eta: Optional[float], eta_rule: str) -> str:
    if eta is not None:
        return f"KL descent (eta={float(eta):.3g})"
    if eta_rule == "eps":
        return "KL descent (eta=eps)"
    return "KL descent (eta=eps/N)"


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
    e_tr_history: Optional[Sequence[float]] = None,
    gibbs_history: Optional[Sequence[int]] = None,
    gibbs_history_unified: Optional[Sequence[int]] = None,
) -> SimpleNamespace:
    """Convert paper79 L-BFGS output to the repo metric protocol."""

    from src.SolverofEQOT import F_marg, marginal_trace_errors

    f = float(F_marg(pi, gammas, dims))
    e = float(np.max(marginal_trace_errors(pi, gammas, dims)))
    n_iters = int(state["n_iters"])
    tol_reached = bool(state["tol_reached"])
    e_list = [float(x) for x in e_tr_history] if e_tr_history is not None else [e]
    gibbs_list = [int(x) for x in gibbs_history] if gibbs_history is not None else [n_iters]
    gibbs_unified_list = (
        [int(x) for x in gibbs_history_unified]
        if gibbs_history_unified is not None
        else [g + k + 1 for k, g in enumerate(gibbs_list)]
    )
    total_gibbs = int(state.get("gibbs_calls", gibbs_list[-1] if gibbs_list else n_iters))
    total_gibbs_unified = int(
        state.get("gibbs_calls_unified", (gibbs_unified_list[-1] if gibbs_unified_list else total_gibbs))
    )

    return SimpleNamespace(
        F_list=[],
        e_tr_list=e_list,
        per_i_tr_list=[marginal_trace_errors(pi, gammas, dims)],
        times=[float(elapsed)],
        pi=pi,
        U_list=list(state["params"]),
        converged=tol_reached,
        n_iters=n_iters,
        gibbs_calls_optimizer=total_gibbs,
        gibbs_calls=total_gibbs,
        gibbs_calls_list=gibbs_list,
        gibbs_calls_unified=total_gibbs_unified,
        gibbs_calls_list_unified=gibbs_unified_list,
    )


def run_lbfgs_with_error_history(
    *,
    cost_matrix,
    ptraces,
    reg: float,
    max_iter: int,
    tol: float,
    verbose: bool = False,
    log_every: int = 50,
) -> Dict[str, Any]:
    """Run paper79 L-BFGS while recording marginal trace error every Gibbs call.

    For the entropy dual, the gradient of the minimized objective is the
    marginal mismatch of the current Gibbs state. We therefore get an
    iteration-resolved marginal-error trajectory without reconstructing the
    primal density matrix at every step.
    """

    ensure_paper79_import_path()
    import jax  # type: ignore
    import jax.numpy as jnp  # type: ignore
    import optax  # type: ignore
    import optax.tree_utils as otu  # type: ignore
    from utils import get_entropy_reg_qot  # type: ignore

    def _trace_norm_hermitian(A):
        Hh = 0.5 * (A + jnp.conj(jnp.swapaxes(A, -1, -2)))
        return jnp.sum(jnp.abs(jnp.linalg.eigvalsh(Hh)))

    def _max_trace_error(grad):
        vals = [_trace_norm_hermitian(g) for g in grad]
        return jnp.max(jnp.stack(vals))

    @partial(jax.jit, static_argnums=(3, 4, 5, 6))
    def _run(cost_matrix, ptraces, reg, max_iter, tol, verbose, log_every):
        fun = lambda dls: -get_entropy_reg_qot(dls, ptraces, cost_matrix, reg)
        init_duals = jax.tree_util.tree_map(jnp.zeros_like, ptraces)
        opt = optax.lbfgs()
        value_and_grad_fun = optax.value_and_grad_from_state(fun)
        state0 = opt.init(init_duals)

        loss_log0 = jnp.full((max_iter,), jnp.nan)
        e_tr_log0 = jnp.full((max_iter,), jnp.nan)
        grad_norm_log0 = jnp.full((max_iter,), jnp.nan)
        gibbs_log0 = jnp.zeros((max_iter,), dtype=jnp.int32)

        def step(carry):
            params, state, loss_log, e_tr_log, grad_norm_log, gibbs_log, it, last_grad_norm, total_gibbs = carry
            value, grad = value_and_grad_fun(params, state=state)
            e_tr = _max_trace_error(grad)
            grad_norm = otu.tree_l2_norm(grad)

            updates, state = opt.update(
                grad,
                state,
                params,
                value=value,
                grad=grad,
                value_fn=fun,
            )
            params = optax.apply_updates(params, updates)
            line_steps = jnp.asarray(
                otu.tree_get(state, "num_linesearch_steps", default=jnp.asarray(0, dtype=jnp.int32)),
                dtype=jnp.int32,
            )
            total_gibbs = total_gibbs + 1 + line_steps

            loss_log = loss_log.at[it].set(-value)
            e_tr_log = e_tr_log.at[it].set(e_tr)
            grad_norm_log = grad_norm_log.at[it].set(grad_norm)
            gibbs_log = gibbs_log.at[it].set(total_gibbs)

            cond = verbose & (it % log_every == 0)

            def _print(args):
                i, v, err, calls = args
                jax.debug.print("[iter {i}] value={v:.6e} e_tr={err:.6e} gibbs={calls}", i=i, v=v, err=err, calls=calls)

            _ = jax.lax.cond(cond, _print, lambda x: None, operand=(it, value, e_tr, total_gibbs))
            return params, state, loss_log, e_tr_log, grad_norm_log, gibbs_log, it + 1, grad_norm, total_gibbs

        def cont(carry):
            _, _, _, _, _, _, it, last_grad_norm, _ = carry
            return (it < 2) | ((it < max_iter) & (last_grad_norm >= tol))

        init_carry = (
            init_duals,
            state0,
            loss_log0,
            e_tr_log0,
            grad_norm_log0,
            gibbs_log0,
            0,
            jnp.inf,
            jnp.asarray(0, dtype=jnp.int32),
        )
        params, state, loss_log, e_tr_log, grad_norm_log, gibbs_log, n_iters, last_grad_norm, total_gibbs = jax.lax.while_loop(
            cont, step, init_carry
        )
        tol_reached = jnp.where(n_iters < max_iter, True, False)
        return {
            "params": params,
            "n_iters": n_iters,
            "state": state[0],
            "loss_history": loss_log,
            "e_tr_history": e_tr_log,
            "grad_norm_history": grad_norm_log,
            "gibbs_history": gibbs_log,
            "gibbs_calls": total_gibbs,
            "tol_reached": tol_reached,
            "last_grad_norm": last_grad_norm,
        }

    return _run(cost_matrix, ptraces, float(reg), int(max_iter), float(tol), bool(verbose), int(log_every))


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
    from utils.entropy import reconstruct_primal  # type: ignore

    cost_matrix = jnp.asarray(np.real_if_close(H))
    ptraces = tuple(jnp.asarray(np.real_if_close(g)) for g in gammas)

    start = time.time()
    state = run_lbfgs_with_error_history(
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
    n_iters = int(state["n_iters"])
    e_tr_history = np.asarray(state.get("e_tr_history", []), dtype=float)[:n_iters]
    gibbs_history = np.asarray(state.get("gibbs_history", []), dtype=int)[:n_iters]
    # Preserve the original paper79/optimizer count, and add a conservative
    # count that includes one diagnostic Gibbs evaluation per recorded point
    # plus the final primal reconstruction performed below.
    gibbs_history_unified = gibbs_history + np.arange(1, n_iters + 1, dtype=int)
    state["gibbs_calls_optimizer"] = int(state.get("gibbs_calls", 0))
    state["gibbs_calls_unified"] = int(state["gibbs_calls_optimizer"] + n_iters + 1)
    return make_result_from_lbfgs(
        state=state,
        pi=pi,
        elapsed=elapsed,
        H=H,
        gammas=gammas,
        dims=dims,
        e_tr_history=e_tr_history,
        gibbs_history=gibbs_history,
        gibbs_history_unified=gibbs_history_unified,
    )


def make_lbfgs_first_hit_row(*, res: SimpleNamespace, args: argparse.Namespace) -> Dict[str, Any]:
    """Create a trajectory-only L-BFGS row for first-hit reporting.

    This row is deliberately not used as the final L-BFGS reference coupling.
    It records when the L-BFGS marginal trajectory first reaches the requested
    trace-error tolerances.
    """

    e = list(getattr(res, "e_tr_list", []) or [])
    gibbs = list(getattr(res, "gibbs_calls_list", []) or [])
    gibbs_unified = list(getattr(res, "gibbs_calls_list_unified", []) or [])
    primary_idx = first_hit_index(e, args.tol_tr) if e else -1
    primary_gibbs = int(gibbs[primary_idx]) if primary_idx >= 0 and len(gibbs) == len(e) else -1
    primary_gibbs_unified = (
        int(gibbs_unified[primary_idx]) if primary_idx >= 0 and len(gibbs_unified) == len(e) else ""
    )
    primary_e = float(e[primary_idx]) if primary_idx >= 0 else (float(e[-1]) if e else "")

    row: Dict[str, Any] = {
        "method": "L-BFGS entropy dual (first hit)",
        "status": "ok",
        "converged": bool(primary_idx >= 0),
        "iters": primary_gibbs if primary_gibbs >= 0 else int(getattr(res, "n_iters", 0) or 0),
        "time_sec": "",
        "gibbs_calls": primary_gibbs if primary_gibbs >= 0 else int(getattr(res, "gibbs_calls", 0) or 0),
        "gibbs_calls_optimizer": primary_gibbs if primary_gibbs >= 0 else int(getattr(res, "gibbs_calls", 0) or 0),
        "gibbs_calls_unified": primary_gibbs_unified if primary_gibbs_unified != "" else getattr(res, "gibbs_calls_unified", ""),
        "final_cost": "",
        "final_entropic_primal": "",
        "final_linear_cost": "",
        "final_entropy": "",
        "final_dual_value": "",
        "final_primal_dual_gap": "",
        "final_F_marg": "",
        "final_e_tr": primary_e,
        "hit_F_iter": "",
        "hit_tr_iter": primary_idx,
        "hit_F_gibbs": "",
        "hit_tr_gibbs": primary_gibbs,
        "hit_tr_gibbs_unified": primary_gibbs_unified,
    }
    for tol in args.tol_f_grid:
        lab = tol_label(tol)
        row[f"hit_F_le_{lab}"] = ""
        row[f"hit_F_iter_le_{lab}"] = ""
        row[f"hit_F_gibbs_le_{lab}"] = ""
    for tol in args.tol_tr_grid:
        lab = tol_label(tol)
        idx = first_hit_index(e, tol) if e else -1
        hit_gibbs = int(gibbs[idx]) if idx >= 0 and len(gibbs) == len(e) else ""
        hit_gibbs_unified = int(gibbs_unified[idx]) if idx >= 0 and len(gibbs_unified) == len(e) else ""
        row[f"hit_tr_le_{lab}"] = bool(idx >= 0) if e else ""
        row[f"hit_tr_iter_le_{lab}"] = idx if e else ""
        row[f"hit_tr_gibbs_le_{lab}"] = hit_gibbs
        row[f"hit_tr_gibbs_unified_le_{lab}"] = hit_gibbs_unified
    return row


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
                tol=-1.0 if getattr(args, "lbfgs_split_rows", True) else args.lbfgs_tol,
            )
            row = summarize_solver_result(
                label="L-BFGS entropy dual (fixed budget)",
                res=res,
                H=H,
                gammas=gammas,
                dims=dims,
                tol_f=args.tol_f,
                tol_tr=args.tol_tr,
                eps=eps,
            )
            row["status"] = "ok"
            row["lbfgs_run_mode"] = "fixed_budget" if getattr(args, "lbfgs_split_rows", True) else "tol_stopped"
        except Exception as exc:
            row = {
                "method": "L-BFGS entropy dual (fixed budget)",
                "status": f"error: {type(exc).__name__}: {exc}",
            }
            res = None
        result_entries.append({"row": row, "res": res, "M_inner": None})
        if res is not None and getattr(args, "lbfgs_split_rows", True):
            result_entries.append(
                {
                    "row": make_lbfgs_first_hit_row(res=res, args=args),
                    "res": None,
                    "M_inner": None,
                    "trajectory_only": True,
                }
            )

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
                    max_gibbs_calls=getattr(args, "max_gibbs_calls", None),
                )
                row = summarize_solver_result(
                    label=label,
                    res=res,
                    H=H,
                    gammas=gammas,
                    dims=dims,
                    tol_f=args.tol_f,
                    tol_tr=args.tol_tr,
                    eps=eps,
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
                    max_gibbs_calls=getattr(args, "max_gibbs_calls", None),
                )
                row = summarize_solver_result(
                    label=f"MD-Sinkhorn (M={int(M_inner)})",
                    res=res,
                    H=H,
                    gammas=gammas,
                    dims=dims,
                    tol_f=args.tol_f,
                    tol_tr=args.tol_tr,
                    eps=eps,
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
        context_row = row_with_context(
            entry["row"],
            experiment=experiment,
            index=inst.index,
            label=inst.label,
            eps=eps,
            dims=dims,
            extra=extra,
        )
        state_path = save_final_state(
            args=args,
            row=context_row,
            res=entry.get("res"),
            H=H,
            gammas=gammas,
            dims=dims,
            eps=eps,
        )
        if state_path:
            context_row["final_state_path"] = state_path
        rows.append(context_row)
    return rows


def add_tolerance_grid_metrics(entries: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    """Add hit columns for several reporting tolerances."""

    for entry in entries:
        if entry.get("trajectory_only"):
            continue
        row = entry["row"]
        res = entry.get("res")
        F = list(getattr(res, "F_list", []) or []) if res is not None else []
        e = list(getattr(res, "e_tr_list", []) or []) if res is not None else []
        gibbs = list(getattr(res, "gibbs_calls_list", []) or []) if res is not None else []
        gibbs_unified = list(getattr(res, "gibbs_calls_list_unified", []) or []) if res is not None else []

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
            row[f"hit_tr_gibbs_unified_le_{lab}"] = (
                int(gibbs_unified[idx]) if idx >= 0 and len(gibbs_unified) == len(e) else ""
            )


def add_cross_method_consistency(entries: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    """Add final-coupling and objective consistency columns in-place."""

    from src.metrics import pi_distance, same_limit

    def _as_float(value: Any) -> Optional[float]:
        if value in (None, ""):
            return None
        try:
            return float(value)
        except Exception:
            return None

    successful = [entry for entry in entries if entry.get("res") is not None and entry["row"].get("status") == "ok"]
    ref = next(
        (entry for entry in successful if entry["row"].get("method") == "L-BFGS entropy dual (fixed budget)"),
        None,
    )

    if ref is None:
        for entry in entries:
            entry["row"]["same_limit_to_lbfgs"] = ""
            entry["row"]["dist_pi_to_lbfgs"] = ""
            entry["row"]["objective_gap_to_lbfgs"] = ""
            entry["row"]["dual_gap_to_lbfgs"] = ""
        return

    ref_res = ref["res"]
    ref_cost = ref["row"].get("final_cost")
    ref_dual = ref["row"].get("final_dual_value")

    for entry in entries:
        row = entry["row"]
        res = entry.get("res")
        if res is None or row.get("status") != "ok":
            row["same_limit_to_lbfgs"] = ""
            row["dist_pi_to_lbfgs"] = ""
            row["objective_gap_to_lbfgs"] = ""
            row["dual_gap_to_lbfgs"] = ""
            continue

        if res is ref_res:
            row["same_limit_to_lbfgs"] = True
            row["dist_pi_to_lbfgs"] = 0.0
            row["objective_gap_to_lbfgs"] = 0.0
            row["dual_gap_to_lbfgs"] = 0.0
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

        ref_cost_f = _as_float(ref_cost)
        cur_cost_f = _as_float(row.get("final_cost"))
        if ref_cost_f is not None and cur_cost_f is not None:
            row["objective_gap_to_lbfgs"] = abs(cur_cost_f - ref_cost_f)
        else:
            row["objective_gap_to_lbfgs"] = ""

        ref_dual_f = _as_float(ref_dual)
        cur_dual_f = _as_float(row.get("final_dual_value"))
        if ref_dual_f is not None and cur_dual_f is not None:
            row["dual_gap_to_lbfgs"] = cur_dual_f - ref_dual_f
        else:
            row["dual_gap_to_lbfgs"] = ""


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
    parser.add_argument("--no_lbfgs_split_rows", action="store_true")
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
    parser.add_argument(
        "--save_final_state",
        action="store_true",
        help="Save final pi, potentials, H, marginals, and metadata as compressed .npz files.",
    )
    parser.add_argument(
        "--state_dir",
        type=Path,
        default=None,
        help="Directory for --save_final_state files. Defaults to a sibling *_states directory.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    args.methods = [x.strip().lower() for x in args.methods.split(",") if x.strip()]
    args.lbfgs_split_rows = not args.no_lbfgs_split_rows
    args.M_list = parse_csv_ints(args.M_list)
    args.eta_kl_rules = parse_csv_strings(args.eta_kl_rules)
    for eta_rule in args.eta_kl_rules:
        if eta_rule not in {"eps_over_N", "eps"}:
            raise ValueError("--eta_kl_rules entries must be eps_over_N or eps")
    args.tol_f_grid = parse_csv_floats(args.tol_f_grid)
    args.tol_tr_grid = parse_csv_floats(args.tol_tr_grid)
    experiments = parse_experiments(args.experiment)
    indices = [0] if args.tiny_smoke else parse_csv_ints(args.indices, all_values=PAPER79_INDICES)
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

    write_csv(args.out, rows, preferred=RUNNER_PREFERRED_FIELDS)
    for row in rows:
        print(row)
    print(f"Wrote {args.out}")
    if checkpoint_path is not None:
        print(f"Wrote partial checkpoint {checkpoint_path}")


if __name__ == "__main__":
    main()
