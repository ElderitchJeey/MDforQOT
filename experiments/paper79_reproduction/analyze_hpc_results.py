"""Summarize downloaded HPC results for the numerical section.

The script expects the downloaded archives to be unpacked under:

    results/MDforQOT_results/MDforQOT/results
    results/MDforQOT_warm_results/MDforQOT_warm/results

It writes paper-facing tables and figures to analysis/numerics by default.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_ORDER = [
    "L-BFGS entropy dual",
    "KL descent (eta=eps/N)",
    "KL descent (eta=eps)",
    "MD-Sinkhorn (M=1)",
    "MD-Sinkhorn (M=2)",
    "MD-Sinkhorn (M=5)",
]

OURS_ORDER = METHOD_ORDER[1:]
METHOD_LABELS = {
    "L-BFGS entropy dual": "L-BFGS",
    "KL descent (eta=eps/N)": r"KL $\varepsilon/N$",
    "KL descent (eta=eps)": r"KL $\varepsilon$",
    "MD-Sinkhorn (M=1)": "MD M=1",
    "MD-Sinkhorn (M=2)": "MD M=2",
    "MD-Sinkhorn (M=5)": "MD M=5",
}
METHOD_PLAIN_LABELS = {
    "L-BFGS entropy dual": "L-BFGS",
    "KL descent (eta=eps/N)": "KL eps/N",
    "KL descent (eta=eps)": "KL eps",
    "MD-Sinkhorn (M=1)": "MD M=1",
    "MD-Sinkhorn (M=2)": "MD M=2",
    "MD-Sinkhorn (M=5)": "MD M=5",
}


def _read_csvs(root: Path, *, warm: bool) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for path in sorted(root.rglob("*.csv")):
        rel = path.relative_to(root)
        name = path.name.lower()
        if "smoke" in name or "demo" in name:
            continue
        # This summary was produced before cvxpy was installed and duplicates
        # the individual epsilon files, so the per-instance CSVs are preferred.
        if rel.as_posix() == "hpc_qubit_eps_summary.csv":
            continue
        if warm and "hpc_warm_" not in rel.as_posix():
            continue
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            print(f"skip {path}: {exc}")
            continue
        df["source_file"] = rel.as_posix()
        df["source_root"] = "warm" if warm else "cold"
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True, sort=False)
    return out


def load_results(cold_root: Path, warm_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    cold = _read_csvs(cold_root, warm=False)
    warm = _read_csvs(warm_root, warm=True)
    if "status" in cold.columns:
        cold = cold[cold["status"].fillna("ok").eq("ok")].copy()
    for df in (cold, warm):
        for col in ["eps", "eps_final", "N", "d", "gibbs_calls", "time_sec", "final_e_tr", "final_F_marg"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "total_with_warm_gibbs" in df.columns:
            df["total_with_warm_gibbs"] = pd.to_numeric(df["total_with_warm_gibbs"], errors="coerce")
        if "total_with_warm_time_sec" in df.columns:
            df["total_with_warm_time_sec"] = pd.to_numeric(df["total_with_warm_time_sec"], errors="coerce")
    return cold, warm


def method_categorical(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["method"] = pd.Categorical(df["method"], categories=METHOD_ORDER, ordered=True)
    return df.sort_values(["method"]).reset_index(drop=True)


def sci(x: float, digits: int = 2) -> str:
    if pd.isna(x):
        return "--"
    return f"{float(x):.{digits}e}"


def fixed(x: float, digits: int = 1) -> str:
    if pd.isna(x):
        return "--"
    return f"{float(x):.{digits}f}"


def write_latex_table(df: pd.DataFrame, path: Path, *, caption: str, label: str) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(df.to_latex(index=False, escape=False, caption=caption, label=label))


def short_instance(label: str) -> str:
    text = str(label)
    text = text.replace("small_", "")
    text = text.replace("_mixed_", " ")
    text = text.replace("_seed0", "")
    return text


def ach_from_error(x: float, tol: float = 1e-4) -> str:
    if pd.isna(x):
        return "--"
    return "Yes" if float(x) <= tol else "No"


def first_success_gibbs(row: pd.Series) -> float:
    """Return Gibbs calls to reach 1e-4 when available, else final calls."""

    hit = pd.to_numeric(row.get("hit_tr_gibbs_le_1em04", np.nan), errors="coerce")
    if pd.notna(hit) and float(hit) >= 0:
        return float(hit)
    return float(pd.to_numeric(row.get("gibbs_calls", np.nan), errors="coerce"))


def first_success_iter(row: pd.Series) -> float:
    hit = pd.to_numeric(row.get("hit_tr_iter_le_1em04", np.nan), errors="coerce")
    if pd.notna(hit) and float(hit) >= 0:
        return float(hit)
    return float(pd.to_numeric(row.get("iters", np.nan), errors="coerce"))


def summarize_main_qubit(cold: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    q = cold[
        cold["source_file"].str.contains("hpc_qubit_mixed", na=False)
        & cold["eps"].eq(1e-3)
    ].copy()
    q = method_categorical(q)
    q["feasible_1e4"] = q["final_e_tr"].le(1e-4)
    q["gibbs_to_1e4"] = q.apply(first_success_gibbs, axis=1)
    q["iters_to_1e4"] = q.apply(first_success_iter, axis=1)
    summary = (
        q.groupby("method", observed=True)
        .agg(
            instances=("paper79_label", "nunique"),
            feasible_rate=("feasible_1e4", "mean"),
            median_e_tr=("final_e_tr", "median"),
            median_F_marg=("final_F_marg", "median"),
            median_gibbs=("gibbs_to_1e4", "median"),
            median_time_sec=("time_sec", "median"),
            median_obj_gap=("objective_gap_to_lbfgs", "median"),
        )
        .reset_index()
    )
    summary.to_csv(outdir / "table_main_qubit_mixed_summary.csv", index=False)
    tex = pd.DataFrame(
        {
            "Method": summary["method"].astype(str),
            "Inst.": summary["instances"].astype(int),
            "$E_{\\rm tr}\\le 10^{-4}$": summary["feasible_rate"].map(lambda x: f"{100*x:.0f}\\%"),
            "$E_{\\rm tr}$": summary["median_e_tr"].map(sci),
            "$F_{\\rm marg}$": summary["median_F_marg"].map(sci),
            "Gibbs to $10^{-4}$": summary["median_gibbs"].map(lambda x: f"{x:.0f}" if pd.notna(x) else "--"),
            "Time (s)": summary["median_time_sec"].map(lambda x: f"{x:.1f}" if pd.notna(x) else "--"),
            "Obj. gap": summary["median_obj_gap"].map(sci),
        }
    )
    write_latex_table(
        tex,
        outdir / "table_main_qubit_mixed_summary.tex",
        caption="Main qubit mixed benchmark at $\\varepsilon=10^{-3}$. Values are medians over available Ising and random instances with $N=4,6,7,8$.",
        label="tab:main-qubit-mixed",
    )

    paper = q.copy()
    paper["instance_short"] = paper["paper79_label"].map(short_instance)
    paper["method_short"] = paper["method"].astype(str).map(METHOD_LABELS)
    paper["ach"] = paper["final_e_tr"].map(ach_from_error)
    paper["gibbs_to_1e4"] = paper.apply(first_success_gibbs, axis=1)
    paper["iters_to_1e4"] = paper.apply(first_success_iter, axis=1)
    paper = paper.sort_values(["paper79_label", "method"])
    paper_cols = pd.DataFrame(
        {
            "Test": paper["instance_short"],
            "Method": paper["method_short"],
            "Time": paper["time_sec"].map(lambda x: f"{x:.1f}" if pd.notna(x) else "--"),
            "Gibbs to $10^{-4}$": paper["gibbs_to_1e4"].map(lambda x: f"{x:.0f}" if pd.notna(x) else "--"),
            "Final Gibbs": paper["gibbs_calls"].map(lambda x: f"{x:.0f}" if pd.notna(x) else "--"),
            "Ach.": paper["ach"],
            "$E_{\\rm tr}$": paper["final_e_tr"].map(sci),
            "$F_{\\rm marg}$": paper["final_F_marg"].map(sci),
            "Obj. gap": paper["objective_gap_to_lbfgs"].map(sci),
        }
    )
    paper_cols.to_csv(outdir / "table_paper79_style_qubit_mixed.csv", index=False)
    write_latex_table(
        paper_cols,
        outdir / "table_paper79_style_qubit_mixed.tex",
        caption="Paper79-style performance table for qubit mixed benchmarks at $\\varepsilon=10^{-3}$. Ach. means $E_{\\rm tr}\\le 10^{-4}$; Gibbs to $10^{-4}$ is the first recorded Gibbs call count reaching this threshold when available.",
        label="tab:paper79-style-qubit",
    )
    return q


def summarize_kl_stepsize(cold: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    q = cold[
        cold["source_file"].str.contains("hpc_qubit", na=False)
        & cold["method"].isin(["KL descent (eta=eps/N)", "KL descent (eta=eps)"])
    ].copy()
    q["gibbs_to_1e4"] = q.apply(first_success_gibbs, axis=1)
    key_cols = ["paper79_label", "eps"]
    wide = q.pivot_table(
        index=key_cols,
        columns="method",
        values=["gibbs_to_1e4", "gibbs_calls", "time_sec", "final_e_tr", "final_F_marg"],
        aggfunc="first",
    )
    wide.columns = [f"{metric}|{method}" for metric, method in wide.columns]
    wide = wide.reset_index()
    wide["gibbs_speedup_eps_over_epsN"] = (
        wide["gibbs_to_1e4|KL descent (eta=eps/N)"] / wide["gibbs_to_1e4|KL descent (eta=eps)"]
    )
    wide["time_speedup_eps_over_epsN"] = (
        wide["time_sec|KL descent (eta=eps/N)"] / wide["time_sec|KL descent (eta=eps)"]
    )
    wide.to_csv(outdir / "table_kl_stepsize_paired.csv", index=False)
    tex = wide.copy()
    tex = tex.sort_values(["eps", "paper79_label"])
    tex_out = pd.DataFrame(
        {
            "Instance": tex["paper79_label"],
            "$\\varepsilon$": tex["eps"].map(sci),
            "Gibbs $\\varepsilon/N$": tex["gibbs_to_1e4|KL descent (eta=eps/N)"].map(lambda x: f"{x:.0f}"),
            "Gibbs $\\varepsilon$": tex["gibbs_to_1e4|KL descent (eta=eps)"].map(lambda x: f"{x:.0f}"),
            "Speedup": tex["gibbs_speedup_eps_over_epsN"].map(lambda x: f"{x:.2f}x"),
            "$E_{\\rm tr}$, $\\varepsilon$": tex["final_e_tr|KL descent (eta=eps)"].map(sci),
        }
    )
    write_latex_table(
        tex_out,
        outdir / "table_kl_stepsize_paired.tex",
        caption="Paired comparison of KL descent step sizes on qubit benchmarks. Gibbs calls are measured at the first hit of $E_{\\rm tr}\\le 10^{-4}$ when available.",
        label="tab:kl-stepsize",
    )
    return wide


def summarize_warm(warm: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    q = warm[warm["source_file"].str.contains("hpc_warm_qubit", na=False)].copy()
    q["cost_gibbs"] = np.where(q["mode"].eq("warm_matched"), q["total_with_warm_gibbs"], q["gibbs_calls"])
    q["cost_time_sec"] = np.where(q["mode"].eq("warm_matched"), q["total_with_warm_time_sec"], q["time_sec"])
    q["feasible_1e4"] = q["final_e_tr"].le(1e-4)
    q.to_csv(outdir / "merged_warm_qubit.csv", index=False)

    summary = (
        q.groupby(["method", "mode"], observed=True)
        .agg(
            rows=("paper79_label", "count"),
            feasible_rate=("feasible_1e4", "mean"),
            median_e_tr=("final_e_tr", "median"),
            median_F_marg=("final_F_marg", "median"),
            median_cost_gibbs=("cost_gibbs", "median"),
            median_cost_time_sec=("cost_time_sec", "median"),
        )
        .reset_index()
    )
    summary.to_csv(outdir / "table_warm_qubit_summary.csv", index=False)
    tex = pd.DataFrame(
        {
            "Method": summary["method"].astype(str),
            "Mode": summary["mode"].replace({"warm_matched": "annealed", "cold": "cold"}),
            "Rows": summary["rows"].astype(int),
            "$E_{\\rm tr}\\le 10^{-4}$": summary["feasible_rate"].map(lambda x: f"{100*x:.0f}\\%"),
            "$E_{\\rm tr}$": summary["median_e_tr"].map(sci),
            "$F_{\\rm marg}$": summary["median_F_marg"].map(sci),
            "Total Gibbs": summary["median_cost_gibbs"].map(lambda x: f"{x:.0f}" if pd.notna(x) else "--"),
            "Total time (s)": summary["median_cost_time_sec"].map(lambda x: f"{x:.1f}" if pd.notna(x) else "--"),
        }
    )
    write_latex_table(
        tex,
        outdir / "table_warm_qubit_summary.tex",
        caption="Cold and annealed matched warm-start variants on available qubit benchmarks. Annealed cost includes all warm-start stages.",
        label="tab:warm-qubit",
    )

    eps_stress = q[q["source_file"].str.contains("hpc_warm_qubit_eps", na=False)].copy()
    eps_stress["mode_label"] = eps_stress["mode"].replace({"warm_matched": "annealed", "cold": "cold"})
    eps_summary = (
        eps_stress.groupby(["eps_final", "method", "mode_label"], observed=True)
        .agg(
            instances=("paper79_label", "count"),
            feasible_rate=("feasible_1e4", "mean"),
            median_e_tr=("final_e_tr", "median"),
            median_total_gibbs=("cost_gibbs", "median"),
        )
        .reset_index()
        .sort_values(["eps_final", "method", "mode_label"])
    )
    eps_summary.to_csv(outdir / "table_warm_small_eps_summary.csv", index=False)
    tex_eps = pd.DataFrame(
        {
            "$\\varepsilon$": eps_summary["eps_final"].map(sci),
            "Method": eps_summary["method"].map(METHOD_LABELS),
            "Mode": eps_summary["mode_label"],
            "Inst.": eps_summary["instances"].astype(int),
            "$E_{\\rm tr}\\le 10^{-4}$": eps_summary["feasible_rate"].map(lambda x: f"{100*x:.0f}\\%"),
            "Median $E_{\\rm tr}$": eps_summary["median_e_tr"].map(sci),
            "Median Gibbs": eps_summary["median_total_gibbs"].map(lambda x: f"{x:.0f}" if pd.notna(x) else "--"),
        }
    )
    write_latex_table(
        tex_eps,
        outdir / "table_warm_small_eps_summary.tex",
        caption="Small-$\\varepsilon$ stress test with cold and matched annealed initializations. Annealed Gibbs counts include all continuation stages.",
        label="tab:warm-small-eps",
    )
    return q


def summarize_wasserstein(cold: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    w = cold[cold["source_file"].str.contains("hpc_wasserstein", na=False)].copy()
    w = method_categorical(w)
    w.to_csv(outdir / "table_wasserstein_completed.csv", index=False)
    tex = pd.DataFrame(
        {
            "Case": w["wasserstein_case"],
            "Method": w["method"].astype(str),
            "$E_{\\rm tr}$": w["final_e_tr"].map(sci),
            "$F_{\\rm marg}$": w["final_F_marg"].map(sci),
            "Gibbs": w["gibbs_calls"].map(lambda x: f"{x:.0f}" if pd.notna(x) else "--"),
            "Time (s)": w["time_sec"].map(lambda x: f"{x:.1f}" if pd.notna(x) else "--"),
        }
    )
    write_latex_table(
        tex,
        outdir / "table_wasserstein_completed.tex",
        caption="Completed $d=20$ Wasserstein/channel benchmark cases. Missing cases were still running at the time of this snapshot.",
        label="tab:wasserstein-completed",
    )
    return w


def setup_plot() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.figsize": (7.0, 4.2),
        }
    )


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path.with_suffix(".pdf"))
    plt.savefig(path.with_suffix(".png"), dpi=220)
    plt.close()


def plot_main_gibbs(q: pd.DataFrame, outdir: Path) -> None:
    data = q.copy()
    if "gibbs_to_1e4" not in data.columns:
        data["gibbs_to_1e4"] = data.apply(first_success_gibbs, axis=1)
    summary = data.groupby("method", observed=True)["gibbs_to_1e4"].median().reindex(METHOD_ORDER)
    colors = ["#5b6770", "#2a9d8f", "#1f6f61", "#e76f51", "#c95d3f", "#9d3f2f"]
    plt.figure()
    plt.bar(np.arange(len(summary)), summary.values, color=colors)
    plt.yscale("log")
    plt.ylabel(r"Median Gibbs calls to $E_{\rm tr}\leq 10^{-4}$")
    plt.xticks(np.arange(len(summary)), ["L-BFGS", "KL eps/N", "KL eps", "MD M=1", "MD M=2", "MD M=5"], rotation=25, ha="right")
    plt.title(r"Qubit mixed benchmarks, $\varepsilon=10^{-3}$")
    savefig(outdir / "fig_A_main_qubit_gibbs")


def plot_qubit_success_heatmap(q: pd.DataFrame, outdir: Path) -> None:
    data = q.copy()
    data["instance_short"] = data["paper79_label"].map(short_instance)
    instances = sorted(data["instance_short"].unique())
    methods = METHOD_ORDER
    matrix = np.full((len(instances), len(methods)), np.nan)
    labels = [["" for _ in methods] for _ in instances]
    for i, inst in enumerate(instances):
        for j, method in enumerate(methods):
            sub = data[(data["instance_short"].eq(inst)) & (data["method"].astype(str).eq(method))]
            if sub.empty:
                continue
            err = float(sub["final_e_tr"].iloc[0])
            matrix[i, j] = np.log10(max(err, 1e-12))
            labels[i][j] = "✓" if err <= 1e-4 else "×"

    plt.figure(figsize=(9.0, 4.8))
    im = plt.imshow(matrix, aspect="auto", cmap="viridis_r", vmin=-5, vmax=0)
    plt.colorbar(im, label=r"$\log_{10} E_{\rm tr}$")
    plt.xticks(np.arange(len(methods)), [METHOD_PLAIN_LABELS[m] for m in methods], rotation=30, ha="right")
    plt.yticks(np.arange(len(instances)), instances)
    for i in range(len(instances)):
        for j in range(len(methods)):
            if labels[i][j]:
                color = "white" if matrix[i, j] > -2.5 else "black"
                plt.text(j, i, labels[i][j], ha="center", va="center", color=color, fontsize=12, fontweight="bold")
    plt.title(r"Qubit mixed feasibility at $\varepsilon=10^{-3}$")
    savefig(outdir / "fig_1_qubit_mixed_feasibility_heatmap")


def plot_qubit_final_error_by_instance(q: pd.DataFrame, outdir: Path) -> None:
    data = q.copy()
    data["instance_short"] = data["paper79_label"].map(short_instance)
    instances = sorted(data["instance_short"].unique())
    x = np.arange(len(instances))
    offsets = np.linspace(-0.34, 0.34, len(METHOD_ORDER))
    colors = ["#5b6770", "#2a9d8f", "#1f6f61", "#e76f51", "#c95d3f", "#9d3f2f"]
    plt.figure(figsize=(10.0, 4.8))
    for offset, method, color in zip(offsets, METHOD_ORDER, colors):
        vals = []
        for inst in instances:
            sub = data[(data["instance_short"].eq(inst)) & (data["method"].astype(str).eq(method))]
            vals.append(float(sub["final_e_tr"].iloc[0]) if not sub.empty else np.nan)
        plt.scatter(x + offset, vals, label=METHOD_LABELS[method], color=color, s=42)
    plt.axhline(1e-4, color="black", linewidth=1, linestyle=":", label=r"$10^{-4}$")
    plt.yscale("log")
    plt.ylabel(r"Final $E_{\rm tr}$")
    plt.xticks(x, instances, rotation=35, ha="right")
    plt.title(r"Per-instance final marginal error, $\varepsilon=10^{-3}$")
    plt.legend(frameon=False, fontsize=8, ncol=3)
    savefig(outdir / "fig_2_qubit_mixed_final_error_by_instance")


def plot_kl_speedup(wide: pd.DataFrame, outdir: Path) -> None:
    data = wide[wide["eps"].eq(1e-3)].sort_values(["paper79_label"]).copy()
    x = np.arange(len(data))
    a = data["gibbs_to_1e4|KL descent (eta=eps/N)"]
    b = data["gibbs_to_1e4|KL descent (eta=eps)"]
    plt.figure(figsize=(9.0, 4.6))
    for i, (av, bv) in enumerate(zip(a, b)):
        color = "#2a9d8f" if bv <= av else "#b44a3c"
        plt.plot([i, i], [av, bv], color="#b7b7b7", linewidth=1.5, zorder=1)
        plt.scatter(i - 0.08, av, color="#2a9d8f", marker="o", s=48, label=r"$\eta=\varepsilon/N$" if i == 0 else "", zorder=2)
        plt.scatter(i + 0.08, bv, color=color, marker="s", s=48, label=r"$\eta=\varepsilon$" if i == 0 else "", zorder=3)
    plt.yscale("log")
    plt.ylabel(r"Gibbs calls to $E_{\rm tr}\leq 10^{-4}$")
    labels = [short_instance(lbl) for lbl in data["paper79_label"]]
    plt.xticks(x, labels, rotation=45, ha="right", fontsize=8)
    plt.legend(frameon=False)
    plt.title(r"KL step-size comparison on qubit mixed benchmarks, $\varepsilon=10^{-3}$")
    savefig(outdir / "fig_B_kl_stepsize_gibbs")


def plot_warm_small_eps(qwarm: pd.DataFrame, outdir: Path) -> None:
    q = qwarm[qwarm["source_file"].str.contains("hpc_warm_qubit_eps", na=False)].copy()
    q["cost_gibbs"] = np.where(q["mode"].eq("warm_matched"), q["total_with_warm_gibbs"], q["gibbs_calls"])
    methods = OURS_ORDER
    eps_values = sorted(q["eps_final"].dropna().unique(), reverse=True)
    matrix = np.full((len(methods), len(eps_values)), np.nan)
    for i, method in enumerate(methods):
        for j, eps in enumerate(eps_values):
            cold = q[(q["method"].eq(method)) & (q["mode"].eq("cold")) & (q["eps_final"].eq(eps))]["final_e_tr"].median()
            warm = q[(q["method"].eq(method)) & (q["mode"].eq("warm_matched")) & (q["eps_final"].eq(eps))]["final_e_tr"].median()
            if pd.notna(cold) and pd.notna(warm) and warm > 0:
                matrix[i, j] = np.log10(cold / warm)
    plt.figure(figsize=(7.8, 4.8))
    im = plt.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=-0.5, vmax=3.0)
    plt.colorbar(im, label=r"$\log_{10}(E_{\rm cold}/E_{\rm annealed})$")
    plt.xticks(np.arange(len(eps_values)), [f"{eps:.0e}" for eps in eps_values])
    plt.yticks(np.arange(len(methods)), [METHOD_PLAIN_LABELS[m] for m in methods])
    for i in range(len(methods)):
        for j in range(len(eps_values)):
            if pd.notna(matrix[i, j]):
                plt.text(j, i, f"{10**matrix[i,j]:.1f}x", ha="center", va="center", fontsize=8)
    plt.xlabel(r"Target $\varepsilon$")
    plt.title("Annealed warm-start improvement in small-epsilon tests")
    savefig(outdir / "fig_C_warm_small_eps_error")


def plot_warm_small_eps_absolute(qwarm: pd.DataFrame, outdir: Path) -> None:
    q = qwarm[qwarm["source_file"].str.contains("hpc_warm_qubit_eps", na=False)].copy()
    q["mode_label"] = q["mode"].replace({"warm_matched": "annealed", "cold": "cold"})
    methods = ["KL descent (eta=eps/N)", "KL descent (eta=eps)", "MD-Sinkhorn (M=1)", "MD-Sinkhorn (M=5)"]
    plt.figure(figsize=(7.8, 4.8))
    styles = {"cold": ("--", "o"), "annealed": ("-", "s")}
    for method in methods:
        for mode_label in ["cold", "annealed"]:
            sub = q[(q["method"].eq(method)) & (q["mode_label"].eq(mode_label))]
            if sub.empty:
                continue
            agg = sub.groupby("eps_final", observed=True)["final_e_tr"].median().sort_index()
            label = f"{METHOD_PLAIN_LABELS[method]} {mode_label}"
            ls, mk = styles[mode_label]
            plt.plot(agg.index, agg.values, linestyle=ls, marker=mk, label=label)
    plt.axhline(1e-4, color="black", linewidth=1, linestyle=":")
    plt.xscale("log")
    plt.yscale("log")
    plt.gca().invert_xaxis()
    plt.xlabel(r"Target $\varepsilon$")
    plt.ylabel(r"Median final $E_{\rm tr}$")
    plt.title("Small-epsilon final error: cold versus annealed")
    plt.legend(frameon=False, fontsize=8, ncol=2)
    savefig(outdir / "fig_3_warm_small_eps_absolute_error")


def plot_wasserstein(w: pd.DataFrame, outdir: Path) -> None:
    if w.empty:
        return
    labels = ["L-BFGS", "KL eps/N", "KL eps", "MD M=1", "MD M=2", "MD M=5"]
    cases = sorted(w["wasserstein_case"].dropna().unique())
    fig, axes = plt.subplots(1, len(cases), figsize=(4.1 * len(cases), 4.2), sharey=True)
    if len(cases) == 1:
        axes = [axes]
    for ax, case in zip(axes, cases):
        sub = method_categorical(w[w["wasserstein_case"].eq(case)])
        vals = sub.set_index("method")["time_sec"].reindex(METHOD_ORDER)
        ax.bar(np.arange(len(vals)), vals.values, color="#457b9d")
        ax.set_yscale("log")
        ax.set_title(f"Wasserstein {case}, d=20")
        ax.set_xticks(np.arange(len(vals)))
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylabel("Time (s)")
    savefig(outdir / "fig_D_wasserstein_completed_time")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cold-root",
        type=Path,
        default=Path("results") / "MDforQOT_results" / "MDforQOT" / "results",
    )
    parser.add_argument(
        "--warm-root",
        type=Path,
        default=Path("results") / "MDforQOT_warm_results" / "MDforQOT_warm" / "results",
    )
    parser.add_argument("--outdir", type=Path, default=Path("analysis") / "numerics")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    cold, warm = load_results(args.cold_root, args.warm_root)
    cold.to_csv(args.outdir / "merged_cold_hpc.csv", index=False)
    warm.to_csv(args.outdir / "merged_warm_hpc.csv", index=False)

    q_main = summarize_main_qubit(cold, args.outdir)
    kl_wide = summarize_kl_stepsize(cold, args.outdir)
    q_warm = summarize_warm(warm, args.outdir)
    wasserstein = summarize_wasserstein(cold, args.outdir)

    setup_plot()
    plot_main_gibbs(q_main, args.outdir)
    plot_qubit_success_heatmap(q_main, args.outdir)
    plot_qubit_final_error_by_instance(q_main, args.outdir)
    plot_kl_speedup(kl_wide, args.outdir)
    plot_warm_small_eps(q_warm, args.outdir)
    plot_warm_small_eps_absolute(q_warm, args.outdir)
    plot_wasserstein(wasserstein, args.outdir)

    print(f"Loaded cold rows: {len(cold)}")
    print(f"Loaded warm rows: {len(warm)}")
    print(f"Wrote tables and figures to {args.outdir}")


if __name__ == "__main__":
    main()
