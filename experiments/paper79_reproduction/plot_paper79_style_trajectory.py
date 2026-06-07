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


def plot_panel(ax, df: pd.DataFrame, mode: str, *, include_warmup: bool) -> None:
    sub = df[df["mode"] == mode].copy()
    if not include_warmup:
        sub = sub[sub["phase"] == "target_eps"].copy()
    for method, mdf in sub.groupby("method"):
        mdf = mdf.sort_values("total_gibbs_calls")
        ax.plot(
            mdf["total_gibbs_calls"],
            mdf["e_tr"].clip(lower=1e-300),
            label=method,
            marker="o",
            markersize=2.5,
            markevery=max(1, len(mdf) // 40),
        )
    ax.axhline(1e-3, color="0.35", linestyle="--", linewidth=1)
    ax.axhline(1e-4, color="0.35", linestyle=":", linewidth=1)
    ax.set_yscale("log")
    ax.set_xlabel("Gibbs calls")
    ax.set_ylabel(r"$E_{\rm tr}$")
    ax.set_title("cold start" if mode == "cold" else "matched warm start")
    ax.legend()


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
    df = df[pd.to_numeric(df["e_tr"], errors="coerce").notna()].copy()
    df["e_tr"] = pd.to_numeric(df["e_tr"])
    df["total_gibbs_calls"] = pd.to_numeric(df["total_gibbs_calls"])

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), sharey=True)
    plot_panel(axes[0], df, "cold", include_warmup=args.include_warmup)
    plot_panel(axes[1], df, "warm_matched", include_warmup=args.include_warmup)
    fig.suptitle(r"Representative hard qubit instance: Ising mixed, $N=7$, $\varepsilon=10^{-3}$")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    png = args.out.with_suffix(".png")
    fig.savefig(png)
    plt.close(fig)
    print(f"Wrote {args.out}")
    print(f"Wrote {png}")


if __name__ == "__main__":
    main()
