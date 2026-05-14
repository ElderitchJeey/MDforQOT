# benchmark/run_my_methods.py

from __future__ import annotations

from typing import Any, Optional

from src.SolverofEQOT import (
    potential_marginal_kl_descent,
    md_type_sinkhorn_potential,
    dbga_algorithm_2_2,
)


def run_my_method(
    *,
    method: str,
    H,
    gammas,
    eps: float,
    dims,
    tol_F: float,
    max_gibbs_calls: int,
    M_inner: Optional[int] = None,
    store_hist: bool = True,
) -> Any:
    """
    Adapter for methods implemented in MDforQOT.
    """

    if method == "KL":
        return potential_marginal_kl_descent(
            H=H,
            gammas=gammas,
            eps=eps,
            dims=dims,
            T=int(max_gibbs_calls),
            tol_F=float(tol_F),
            tol_tr=None,
            store_hist=store_hist,
            project_pi=True,
        )

    if method == "MD":
        if M_inner is None:
            raise ValueError("MD requires M_inner.")

        return md_type_sinkhorn_potential(
            H=H,
            gammas=gammas,
            eps=eps,
            dims=dims,
            T_outer=int(max_gibbs_calls),
            tol_tr=0.0,
            tol_F=float(tol_F),
            M_inner=int(M_inner),
            tol_inner=0.0,
            keep_U_hist=store_hist,
            keep_pi_hist=False,
            project_pi=True,
        )

    if method == "BGDA":
        # Auxiliary only. Do not include in the main jobs.csv unless explicitly needed.
        return dbga_algorithm_2_2(
            H=H,
            gammas=gammas,
            eps=eps,
            dims=dims,
            T=int(max_gibbs_calls),
            tol_F=float(tol_F),
            tol_tr=None,
            store_hist=store_hist,
            project_pi=True,
        )

    raise ValueError(f"Unknown MDforQOT method: {method}")