from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("results") / "MDforQOT_rerun_results" / "results" / "hpc_qubit_scaling_N3N7"
OUT_DIR = Path("analysis") / "numerics" / "rerun_tables"


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
    mant, exp = f"{v:.{digits}e}".split("e")
    return rf"${float(mant):.{digits}f}\times 10^{{{int(exp)}}}$"


def method_label(method: str) -> str:
    return (
        method.replace("L-BFGS entropy dual (fixed budget)", "L-BFGS")
        .replace("KL descent (eta=eps/N)", r"KL $\eta=\varepsilon/N$")
        .replace("KL descent (eta=eps)", r"KL $\eta=\varepsilon$")
        .replace("MD-Sinkhorn (M=1)", r"MD $M=1$")
        .replace("MD-Sinkhorn (M=2)", r"MD $M=2$")
        .replace("MD-Sinkhorn (M=5)", r"MD $M=5$")
    )


def resolve_state_path(path_str: Any) -> Path | None:
    if path_str is None or (isinstance(path_str, float) and np.isnan(path_str)):
        return None
    raw = str(path_str).replace("\\", "/")
    if not raw or raw.lower() == "nan":
        return None
    for candidate in [
        Path("results") / "MDforQOT_rerun_results" / raw,
        ROOT.parent.parent / raw,
        Path(raw),
    ]:
        if candidate.exists():
            return candidate
    return None


def load_pi(path: Path) -> np.ndarray:
    with np.load(path) as data:
        return np.asarray(data["pi"])


def trace_norm(A: np.ndarray) -> float:
    H = 0.5 * (A + A.conj().T)
    return float(np.sum(np.abs(np.linalg.eigvalsh(H))))


def read_rows() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(ROOT.glob("N*/*_all_eps1em03.csv")):
        df = pd.read_csv(path)
        df = df[df["method"] != "L-BFGS entropy dual (first hit)"].copy()
        df["source_file"] = str(path)
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    raw["N"] = raw["dims"].astype(str).str.split("x").str.len()
    raw["type"] = raw["small_kind"].str.capitalize()
    raw["eps_used"] = pd.to_numeric(raw["eps"], errors="coerce")
    return raw


def build_table() -> pd.DataFrame:
    raw = read_rows()

    lbfgs_pi: dict[tuple[str, int], np.ndarray] = {}
    for _, row in raw[raw["method"] == "L-BFGS entropy dual (fixed budget)"].iterrows():
        state = resolve_state_path(row.get("final_state_path"))
        if state is not None:
            lbfgs_pi[(str(row["small_kind"]), int(row["N"]))] = load_pi(state)

    order = [
        "L-BFGS entropy dual (fixed budget)",
        "KL descent (eta=eps/N)",
        "KL descent (eta=eps)",
        "MD-Sinkhorn (M=1)",
        "MD-Sinkhorn (M=2)",
        "MD-Sinkhorn (M=5)",
    ]
    raw["method_order"] = raw["method"].map({m: i for i, m in enumerate(order)})
    raw = raw.sort_values(["type", "N", "method_order"])

    rows: list[dict[str, Any]] = []
    for _, row in raw.iterrows():
        kind = str(row["small_kind"])
        N = int(row["N"])
        dist = 0.0 if row["method"] == "L-BFGS entropy dual (fixed budget)" else np.nan
        state = resolve_state_path(row.get("final_state_path"))
        ref = lbfgs_pi.get((kind, N))
        if state is not None and ref is not None and row["method"] != "L-BFGS entropy dual (fixed budget)":
            dist = trace_norm(load_pi(state) - ref)
        elif row.get("dist_pi_to_lbfgs") not in (None, ""):
            try:
                dist = float(row.get("dist_pi_to_lbfgs"))
            except Exception:
                pass

        rows.append(
            {
                "Type": row["type"],
                "N": N,
                "epsilon": float(row["eps_used"]),
                "Method": method_label(str(row["method"])),
                "hit_1e3": row.get("hit_tr_gibbs_le_1em03"),
                "hit_1e4": row.get("hit_tr_gibbs_le_1em04"),
                "final_gibbs": row.get("gibbs_calls"),
                "final_e_tr": row.get("final_e_tr"),
                "dist_pi_lbfgs": dist,
                "primal": row.get("final_entropic_primal", row.get("final_cost")),
                "dual": row.get("final_dual_value"),
            }
        )
    return pd.DataFrame(rows)


def to_latex(df: pd.DataFrame) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\caption{Cold-start qubit mixed benchmarks at $\varepsilon=10^{-3}$. The entries report first-hit Gibbs calls for the marginal trace error thresholds.}",
        r"\label{tab:qubit-scaling-N3N7-rerun}",
        r"\begin{tabular}{llllrrrrrrr}",
        r"\toprule",
        r"Type & $N$ & $\varepsilon$ & Method & $E_{\rm tr}\le10^{-3}$ & $E_{\rm tr}\le10^{-4}$ & Final Gibbs & Final $E_{\rm tr}$ & $\|\pi-\pi_{\rm LBFGS}\|_1$ & Primal & Dual \\",
        r"\midrule",
    ]
    last_key = None
    for _, row in df.iterrows():
        key = (row["Type"], row["N"])
        if last_key is not None and key != last_key:
            lines.append(r"\midrule")
        lines.append(
            " & ".join(
                [
                    str(row["Type"]),
                    str(int(row["N"])),
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
    table.to_csv(OUT_DIR / "table_qubit_scaling_N3N7_rerun.csv", index=False)
    (OUT_DIR / "table_qubit_scaling_N3N7_rerun.tex").write_text(to_latex(table), encoding="utf-8")
    print(table.to_string(index=False))
    print(f"Wrote {OUT_DIR / 'table_qubit_scaling_N3N7_rerun.csv'}")
    print(f"Wrote {OUT_DIR / 'table_qubit_scaling_N3N7_rerun.tex'}")


if __name__ == "__main__":
    main()
