from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import ScalarFormatter


CSV = (
    Path("results")
    / "MDforQOT_rerun_results"
    / "results"
    / "hpc_paper79_style_N6"
    / "ising_mixed_N6_eps1em03_trajectory.csv"
)
OUT = Path("analysis") / "numerics" / "rerun_tables" / "fig_N6_ising_mixed_trajectory_panels.pdf"


STYLE = {
    "KL eta=eps/N": {"label": r"$\mathrm{KL}$", "color": "#1f77b4", "linestyle": "-"},
    "KL eta=eps": {"label": r"$\mathsf{KL}_{\rm emp}$", "color": "#ff7f0e", "linestyle": "-"},
    "MD M=1": {"label": r"$\mathrm{MD}_{1}$", "color": "#2ca02c", "linestyle": "-"},
    "MD M=2": {"label": r"$\mathrm{MD}_{2}$", "color": "#d62728", "linestyle": "-"},
    "MD M=5": {"label": r"$\mathrm{MD}_{5}$", "color": "#9467bd", "linestyle": "-"},
}


def paper_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 300,
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9,
            "legend.fontsize": 8.5,
            "lines.linewidth": 1.6,
            "axes.grid": True,
            "grid.alpha": 0.2,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_metric(ax, df: pd.DataFrame, metric: str, title: str, ylabel: str, *, logy: bool) -> None:
    for method, mdf in df.groupby("method", sort=False):
        mdf = mdf.sort_values("total_gibbs_calls")
        style = STYLE.get(method, {"label": method})
        y = pd.to_numeric(mdf[metric], errors="coerce")
        x = pd.to_numeric(mdf["total_gibbs_calls"], errors="coerce")
        ok = x.notna() & y.notna()
        y_plot = y[ok].clip(lower=1e-300) if logy else y[ok]
        ax.plot(
            x[ok],
            y_plot,
            label=style.get("label", method),
            color=style.get("color"),
            linestyle=style.get("linestyle", "-"),
            marker=None,
            markevery=max(1, int(ok.sum()) // 35),
        )
    if logy:
        ax.set_yscale("log")
    if metric == "e_tr":
        ax.axhline(1e-3, color="0.35", linestyle="--", linewidth=1.0)
        ax.axhline(1e-4, color="0.35", linestyle=":", linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel("Gibbs calls")
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))


def main() -> None:
    paper_style()
    df = pd.read_csv(CSV)
    df = df[df["mode"] == "cold"].copy()
    method_order = list(STYLE)
    df["method_order"] = df["method"].map({m: i for i, m in enumerate(method_order)})
    df = df.sort_values(["method_order", "total_gibbs_calls"])

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.15))
    plot_metric(axes[0], df, "e_tr", r"(a) Trace marginal error", r"$E_{\rm tr}$", logy=True)
    plot_metric(axes[1], df, "F_marg", r"(b) Marginal KL error", r"$F_{\rm marg}$", logy=True)
    plot_metric(axes[2], df, "dual_value", r"(c) Dual objective", r"$D_{\varepsilon}$", logy=False)
    axes[2].set_ylim(bottom=-0.96)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=1.4)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), bbox_inches="tight")
    print(f"Wrote {OUT}")
    print(f"Wrote {OUT.with_suffix('.png')}")


if __name__ == "__main__":
    main()
