from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


WARM_ROOT = Path("results") / "MDforQOT_rerun_results" / "results" / "hpc_standard_qubit_eps_N4"
LBFGS_ROOT = (
    Path("results")
    / "MDforQOT_lbfgs150k_rerun"
    / "results"
    / "hpc_standard_qubit_eps_N4_lbfgs150k_rerun"
)
OUT_DIR = Path("analysis") / "numerics" / "rerun_tables"
EPS_KEEP = {1e-5, 1e-7}


def fmt_eps(x: float) -> str:
    return f"$10^{{{int(round(np.log10(float(x))))}}}$"


def fmt_hit(x: Any) -> str:
    try:
        v = int(float(x))
    except Exception:
        return "--"
    return "--" if v < 0 else str(v)


def fmt_int(x: Any) -> str:
    try:
        return str(int(float(x)))
    except Exception:
        return "--"


def fmt_sci(x: Any, digits: int = 2) -> str:
    try:
        v = float(x)
    except Exception:
        return "--"
    if not np.isfinite(v):
        return "--"
    if v == 0:
        return "0"
    s = f"{v:.{digits}e}"
    mant, exp = s.split("e")
    return rf"${float(mant):.{digits}f}\times 10^{{{int(exp)}}}$"


def load_pi(path: Path) -> np.ndarray:
    with np.load(path) as data:
        return np.asarray(data["pi"])


def trace_norm(A: np.ndarray) -> float:
    H = 0.5 * (A + A.conj().T)
    return float(np.sum(np.abs(np.linalg.eigvalsh(H))))


def resolve_state_path(path_str: Any) -> Path | None:
    if path_str is None or (isinstance(path_str, float) and np.isnan(path_str)):
        return None
    raw = str(path_str).replace("\\", "/")
    if not raw or raw.lower() == "nan":
        return None
    candidates = [
        Path("results") / "MDforQOT_lbfgs150k_rerun" / raw,
        Path("results") / "MDforQOT_lbfgs_rerun" / raw,
        Path("results") / "MDforQOT_rerun_results" / raw,
        LBFGS_ROOT.parent.parent / raw,
        WARM_ROOT.parent.parent / raw,
        Path(raw),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def method_label(method: str) -> str:
    return (
        method.replace("L-BFGS entropy dual (fixed budget)", "L-BFGS")
        .replace("KL descent (eta=eps/N)", r"KL $\eta=\varepsilon/N$")
        .replace("KL descent (eta=eps)", r"KL $\eta=\varepsilon$")
        .replace("MD-Sinkhorn (M=1)", r"MD $M=1$")
        .replace("MD-Sinkhorn (M=2)", r"MD $M=2$")
        .replace("MD-Sinkhorn (M=5)", r"MD $M=5$")
    )


def read_rows() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted((LBFGS_ROOT / "cold" / "N4").glob("*_lbfgs_eps*.csv")):
        df = pd.read_csv(path)
        df = df[df["method"] == "L-BFGS entropy dual (fixed budget)"].copy()
        df["source_file"] = str(path)
        frames.append(df)
    for path in sorted((WARM_ROOT / "warm" / "N4").glob("*_warm_eps*.csv")):
        df = pd.read_csv(path)
        df = df[df["mode"] == "warm_matched"].copy()
        df["source_file"] = str(path)
        frames.append(df)

    raw = pd.concat(frames, ignore_index=True)
    eps_final = pd.to_numeric(raw["eps_final"], errors="coerce") if "eps_final" in raw.columns else pd.Series(np.nan, index=raw.index)
    eps_base = pd.to_numeric(raw["eps"], errors="coerce") if "eps" in raw.columns else pd.Series(np.nan, index=raw.index)
    raw["eps_used"] = eps_final.fillna(eps_base)
    raw = raw[raw["eps_used"].isin(EPS_KEEP)].copy()
    raw["type"] = raw["small_kind"].str.capitalize()
    return raw


def build_table() -> pd.DataFrame:
    raw = read_rows()

    lbfgs_pi: dict[tuple[str, float], np.ndarray] = {}
    for _, row in raw[raw["method"] == "L-BFGS entropy dual (fixed budget)"].iterrows():
        state = resolve_state_path(row.get("final_state_path"))
        if state is not None:
            lbfgs_pi[(str(row["small_kind"]), float(row["eps_used"]))] = load_pi(state)

    out_rows: list[dict[str, Any]] = []
    order = [
        "L-BFGS entropy dual (fixed budget)",
        "KL descent (eta=eps/N)",
        "KL descent (eta=eps)",
        "MD-Sinkhorn (M=1)",
        "MD-Sinkhorn (M=2)",
        "MD-Sinkhorn (M=5)",
    ]
    raw["method_order"] = raw["method"].map({m: i for i, m in enumerate(order)})
    raw["eps_order"] = raw["eps_used"].map({1e-5: 0, 1e-7: 1})
    raw = raw.sort_values(["type", "eps_order", "method_order"])

    for _, row in raw.iterrows():
        kind = str(row["small_kind"])
        eps = float(row["eps_used"])
        is_warm = str(row.get("mode", "")) == "warm_matched"
        hit_1e3 = row.get("total_hit_tr_gibbs_le_1em03") if is_warm else row.get("hit_tr_gibbs_le_1em03")
        hit_1e4 = row.get("total_hit_tr_gibbs_le_1em04") if is_warm else row.get("hit_tr_gibbs_le_1em04")
        final_gibbs = row.get("total_with_warm_gibbs") if is_warm else row.get("gibbs_calls")

        dist = 0.0 if row["method"] == "L-BFGS entropy dual (fixed budget)" else np.nan
        state = resolve_state_path(row.get("final_state_path"))
        ref = lbfgs_pi.get((kind, eps))
        if state is not None and ref is not None and row["method"] != "L-BFGS entropy dual (fixed budget)":
            dist = trace_norm(load_pi(state) - ref)

        out_rows.append(
            {
                "Type": row["type"],
                "epsilon": eps,
                "Method": method_label(str(row["method"])),
                "hit_1e3": hit_1e3,
                "hit_1e4": hit_1e4,
                "final_gibbs": final_gibbs,
                "final_e_tr": row.get("final_e_tr"),
                "dist_pi_lbfgs": dist,
                "primal": row.get("final_entropic_primal", row.get("final_cost")),
                "dual": row.get("final_dual_value"),
            }
        )
    return pd.DataFrame(out_rows)


def to_latex(df: pd.DataFrame) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\caption{Small-$\varepsilon$ qubit mixed benchmarks with $N=4$. Warm-start methods include the Gibbs calls spent in the continuation stages.}",
        r"\label{tab:small-eps-warm-lbfgs}",
        r"\begin{tabular}{lllrrrrrrr}",
        r"\toprule",
        r"Type & $\varepsilon$ & Method & $E_{\rm tr}\le10^{-3}$ & $E_{\rm tr}\le10^{-4}$ & Final Gibbs & Final $E_{\rm tr}$ & $\|\pi-\pi_{\rm LBFGS}\|_1$ & Primal & Dual \\",
        r"\midrule",
    ]
    last_key = None
    for _, row in df.iterrows():
        key = (row["Type"], row["epsilon"])
        if last_key is not None and key != last_key:
            lines.append(r"\midrule")
        lines.append(
            " & ".join(
                [
                    str(row["Type"]),
                    fmt_eps(row["epsilon"]),
                    str(row["Method"]),
                    fmt_hit(row["hit_1e3"]),
                    fmt_hit(row["hit_1e4"]),
                    fmt_int(row["final_gibbs"]),
                    fmt_sci(row["final_e_tr"]),
                    fmt_sci(row["dist_pi_lbfgs"]),
                    fmt_sci(row["primal"]),
                    fmt_sci(row["dual"]),
                ]
            )
            + r" \\"
        )
        last_key = key
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table = build_table()
    table.to_csv(OUT_DIR / "table_small_eps_warm_lbfgs.csv", index=False)
    (OUT_DIR / "table_small_eps_warm_lbfgs.tex").write_text(to_latex(table), encoding="utf-8")
    print(table.to_string(index=False))
    print(f"Wrote {OUT_DIR / 'table_small_eps_warm_lbfgs.csv'}")
    print(f"Wrote {OUT_DIR / 'table_small_eps_warm_lbfgs.tex'}")


if __name__ == "__main__":
    main()
