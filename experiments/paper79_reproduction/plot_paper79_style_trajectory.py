"""Plot paper79-style trajectory curves from trajectory CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def paper_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 8,
            "lines.linewidth": 2,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )


def plot_panel(
    ax,
    df: pd.DataFrame,
    mode: str,
    *,
    metric: str,
    ylabel: str,
    include_warmup: bool,
    logy: bool = True,
) -> None:
    sub = df[df["mode"] == mode].copy()
    if not include_warmup:
        sub = sub[sub["phase"] == "target_eps"].copy()
    for method, mdf in sub.groupby("method"):
        mdf = mdf.sort_values("total_gibbs_calls")
        ax.plot(
            mdf["total_gibbs_calls"],
            mdf[metric].clip(lower=1e-300) if logy else mdf[metric],
            label=method,
            marker="o",
            markersize=2.5,
            markevery=max(1, len(mdf) // 40),
        )
    if metric == "e_tr":
        ax.axhline(1e-3, color="0.35", linestyle="--", linewidth=1)
        ax.axhline(1e-4, color="0.35", linestyle=":", linewidth=1)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("Gibbs calls")
    ax.set_ylabel(ylabel)
    ax.set_title("cold start" if mode == "cold" else "matched warm start")
    ax.legend()


def plot_metric(
    *,
    df: pd.DataFrame,
    out: Path,
    metric: str,
    ylabel: str,
    title_suffix: str,
    include_warmup: bool,
    logy: bool = True,
) -> None:
    plot_df = df[pd.to_numeric(df[metric], errors="coerce").notna()].copy()
    plot_df[metric] = pd.to_numeric(plot_df[metric])
    if plot_df.empty:
        print(f"Skipped {metric}: no numeric values")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), sharey=True)
    plot_panel(
        axes[0],
        plot_df,
        "cold",
        metric=metric,
        ylabel=ylabel,
        include_warmup=include_warmup,
        logy=logy,
    )
    plot_panel(
        axes[1],
        plot_df,
        "warm_matched",
        metric=metric,
        ylabel=ylabel,
        include_warmup=include_warmup,
        logy=logy,
    )
    fig.suptitle(
        rf"Representative hard qubit instance: Ising mixed, $N=7$, $\varepsilon=10^{{-3}}$ ({title_suffix})"
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    png = out.with_suffix(".png")
    fig.savefig(png)
    plt.close(fig)
    print(f"Wrote {out}")
    print(f"Wrote {png}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Plot paper79-style hard-case trajectory")
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results") / "figures_paper79_style" / "N7_trajectory.pdf")
    parser.add_argument("--include_warmup", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    paper_style()
    df = pd.read_csv(args.trajectory)
    df["total_gibbs_calls"] = pd.to_numeric(df["total_gibbs_calls"])

    stem = args.out.with_suffix("")
    plot_metric(
        df=df,
        out=args.out,
        metric="e_tr",
        ylabel=r"$E_{\rm tr}$",
        title_suffix=r"$E_{\rm tr}$",
        include_warmup=args.include_warmup,
        logy=True,
    )
    plot_metric(
        df=df,
        out=stem.with_name(stem.name + "_Fmarg").with_suffix(args.out.suffix),
        metric="F_marg",
        ylabel=r"$F_{\rm marg}$",
        title_suffix=r"$F_{\rm marg}$",
        include_warmup=args.include_warmup,
        logy=True,
    )
    plot_metric(
        df=df,
        out=stem.with_name(stem.name + "_dual").with_suffix(args.out.suffix),
        metric="dual_value",
        ylabel="dual value",
        title_suffix="dual value",
        include_warmup=args.include_warmup,
        logy=False,
    )


if __name__ == "__main__":
    main()
