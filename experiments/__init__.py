# experiments/sweep_eta_bgda.py
#
# Sweep BGDA eta to find the best step size for a fixed instance.
#
# Usage (from repo root):
#   python -m experiments.sweep_eta_bgda
#   python -m experiments.sweep_eta_bgda --dims 2 2 --eps 0.5 --tol 1e-8 --T 300 --eta_grid 0.25 0.5 1 2 4 8
#   python -m experiments.sweep_eta_bgda --save_csv results/bgda_eta_sweep.csv
#
# This script assumes:
# - You have a BGDA function: block_gradient_ascent(...) returning DBGAResult with fields:
#   converged, e_tr_list, F_list, times, gibbs_calls
# - Your core utilities live in src.linalg / src.tensor
#
# If your module file name differs, update the import in "Imports from src" section.

import argparse
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import numpy.linalg as npl

from src.linalg import hermitianize, proj_to_density
from src.tensor import dims_prod

# ====== Imports from src (adjust module name if needed) ======
from src.SolverofEQOT import block_gradient_ascent  # <-- change if your file name differs


# -------------------------
# Instance generation
# -------------------------

def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def gen_random_hermitian(D: int, rng: np.random.Generator, scale: float = 1.0) -> np.ndarray:
    A = (rng.normal(size=(D, D)) + 1j * rng.normal(size=(D, D))) / np.sqrt(2.0)
    H = hermitianize(A)
    wmax = float(np.max(np.abs(npl.eigvalsh(H))))
    if wmax > 0:
        H = (scale / wmax) * H
    return hermitianize(H)


def gen_random_density(d: int, rng: np.random.Generator, jitter: float = 1e-12) -> np.ndarray:
    G = (rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))) / np.sqrt(2.0)
    rho = G @ G.conj().T
    return proj_to_density(rho, jitter=jitter)


# -------------------------
# Selection logic
# -------------------------

@dataclass
class SweepRow:
    eta: float
    converged: bool
    stop_iter: int
    final_e_tr: float
    final_F: float
    gibbs_calls: int
    time_sec: float
    message: str = "ok"


def _pick_best(rows: List[SweepRow], prefer: str = "gibbs_calls") -> SweepRow:
    """
    Rule:
      1) Prefer converged runs over non-converged.
      2) Among converged: minimize prefer (gibbs_calls or time_sec), tie-break by final_e_tr.
      3) If none converged: minimize final_e_tr, tie-break by prefer.
    """
    if prefer not in ("gibbs_calls", "time_sec"):
        raise ValueError("prefer must be 'gibbs_calls' or 'time_sec'.")

    good = [r for r in rows if r.message == "ok"]
    if not good:
        raise RuntimeError("No successful runs (all errored).")

    conv = [r for r in good if r.converged]
    cand = conv if conv else good

    def key(r: SweepRow) -> Tuple[float, float]:
        p = float(r.gibbs_calls) if prefer == "gibbs_calls" else float(r.time_sec)
        if conv:
            # converged: primary prefer, secondary final_e_tr
            return (p, float(r.final_e_tr))
        # non-converged: primary final_e_tr, secondary prefer
        return (float(r.final_e_tr), p)

    return min(cand, key=key)


def _maybe_save_csv(path: str, rows: List[SweepRow]) -> None:
    import csv
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["eta", "converged", "stop_iter", "final_e_tr", "final_F", "gibbs_calls", "time_sec", "message"])
        for r in rows:
            w.writerow([r.eta, int(r.converged), r.stop_iter, r.final_e_tr, r.final_F, r.gibbs_calls, r.time_sec, r.message])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", nargs="+", type=int, default=[2, 2], help="Local dimensions, e.g. --dims 2 2 or --dims 2 2 2")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for instance")
    ap.add_argument("--eps", type=float, default=0.5, help="Entropic regularization eps")
    ap.add_argument("--tol", type=float, default=1e-8, help="Trace-norm marginal tolerance")
    ap.add_argument("--T", type=int, default=300, help="Max iterations for BGDA")
    ap.add_argument("--eta_grid", nargs="+", type=float, default=[0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
    ap.add_argument("--prefer", type=str, default="gibbs_calls", choices=["gibbs_calls", "time_sec"])
    ap.add_argument("--save_csv", type=str, default="", help="Optional path to save CSV, e.g. results/bgda_eta_sweep.csv")
    ap.add_argument("--no_project", action="store_true", help="Disable proj_to_density inside Gibbs mapping (if supported)")
    args = ap.parse_args()

    dims = [int(x) for x in args.dims]
    N = len(dims)
    D = dims_prod(dims)

    print("=" * 78)
    print("BGDA eta sweep")
    print(f"dims={dims}, N={N}, D={D}, seed={args.seed}, eps={args.eps}, tol={args.tol}, T={args.T}")
    print(f"eta_grid={args.eta_grid}")
    print(f"prefer={args.prefer}")
    print("=" * 78)

    rng = _rng(args.seed)
    H = gen_random_hermitian(D, rng, scale=1.0)
    gammas = [gen_random_density(d, rng, jitter=1e-12) for d in dims]

    project_pi = (not args.no_project)

    rows: List[SweepRow] = []
    for eta in args.eta_grid:
        t0 = time.perf_counter()
        try:
            res = block_gradient_ascent(
                H=H,
                gammas=gammas,
                eps=float(args.eps),
                dims=dims,
                T=int(args.T),
                eta=float(eta),
                tol_tr=float(args.tol),
                gauge_trace0=True,
                store_hist=False,
                project_pi=project_pi,
            )
            elapsed = float(time.perf_counter() - t0)
            final_e = float(res.e_tr_list[-1]) if res.e_tr_list else float("inf")
            final_F = float(res.F_list[-1]) if res.F_list else float("inf")
            stop_iter = (len(res.e_tr_list) - 1) if res.e_tr_list else -1

            row = SweepRow(
                eta=float(eta),
                converged=bool(res.converged),
                stop_iter=int(stop_iter),
                final_e_tr=final_e,
                final_F=final_F,
                gibbs_calls=int(res.gibbs_calls),
                time_sec=elapsed,
                message="ok",
            )
        except Exception as e:
            elapsed = float(time.perf_counter() - t0)
            row = SweepRow(
                eta=float(eta),
                converged=False,
                stop_iter=-1,
                final_e_tr=float("inf"),
                final_F=float("inf"),
                gibbs_calls=0,
                time_sec=elapsed,
                message=f"error: {type(e).__name__}: {e}",
            )

        rows.append(row)
        print(
            f"eta={row.eta:>8.4g} | conv={str(row.converged):5s} | "
            f"iter={row.stop_iter:4d} | e_tr={row.final_e_tr:.3e} | "
            f"F={row.final_F:.3e} | calls={row.gibbs_calls:4d} | "
            f"time={row.time_sec:.3f}s | {row.message}"
        )

    best = _pick_best(rows, prefer=args.prefer)
    print("-" * 78)
    print(
        f"BEST eta = {best.eta}  (prefer={args.prefer})\n"
        f"  converged={best.converged}\n"
        f"  stop_iter={best.stop_iter}\n"
        f"  final_e_tr={best.final_e_tr:.3e}\n"
        f"  final_F={best.final_F:.3e}\n"
        f"  gibbs_calls={best.gibbs_calls}\n"
        f"  time_sec={best.time_sec:.3f}"
    )

    if args.save_csv:
        _maybe_save_csv(args.save_csv, rows)
        print(f"\nSaved CSV to: {args.save_csv}")

    print("\nDone.")


if __name__ == "__main__":
    main()
