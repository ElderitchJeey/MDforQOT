"""Plot Figure A-D for small paper79-like experiments."""

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


def safe_positive(series):
    return series.clip(lower=1e-300)


def plot_trajectory(df: pd.DataFrame, *, y: str, ylabel: str, title: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for method, sub in df.groupby("method"):
        sub = sub.sort_values("gibbs_calls")
        ax.plot(sub["gibbs_calls"], safe_positive(sub[y]), marker="o", markersize=3, label=method)
    ax.set_yscale("log")
    ax.set_xlabel("Gibbs evaluations")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_eps_robustness(summary: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for method, sub in summary.groupby("method"):
        sub = sub.sort_values("eps")
        ax.plot(sub["eps"], safe_positive(sub["final_e_tr"]), marker="o", markersize=4, label=method)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\varepsilon$")
    ax.set_ylabel("final marginal trace error")
    ax.set_title("Small-epsilon robustness")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_cost_summary(summary: pd.DataFrame, out: Path) -> None:
    df = summary.copy()
    df["label"] = df["small_kind"] + " / " + df["method"]
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    x = range(len(df))
    ax.bar(x, df["gibbs_calls"].astype(float))
    ax.set_yscale("log")
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["label"], rotation=35, ha="right")
    ax.set_ylabel("Gibbs / expmap evaluations")
    ax.set_title("Cost to reported final point")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Plot small qubit Figure A-D")
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--eps_summary", type=Path, default=None)
    parser.add_argument("--out_dir", type=Path, default=Path("results") / "figures_small_qubit")
    parser.add_argument("--kind", default="ising")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    paper_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    traj = pd.read_csv(args.trajectory)
    traj = traj[traj["small_kind"] == args.kind].copy()

    summary = pd.read_csv(args.summary)

    plot_trajectory(
        traj,
        y="e_tr",
        ylabel=r"$\max_i \|\mathrm{Tr}_{\ne i}\pi-\gamma_i\|_1$",
        title=f"Figure A: marginal error decay ({args.kind})",
        out=args.out_dir / "figure_A_marginal_error_decay.png",
    )
    plot_trajectory(
        traj,
        y="F_marg",
        ylabel=r"$F_{\mathrm{marg}}$",
        title=f"Figure B: marginal KL decay ({args.kind})",
        out=args.out_dir / "figure_B_marginal_KL_decay.png",
    )

    eps_summary_path = args.eps_summary or args.summary
    eps_summary = pd.read_csv(eps_summary_path)
    eps_summary = eps_summary[eps_summary["small_kind"] == args.kind].copy()
    plot_eps_robustness(eps_summary, args.out_dir / "figure_C_eps_robustness.png")

    plot_cost_summary(summary, args.out_dir / "figure_D_cost_summary.png")
    print(f"Wrote figures to {args.out_dir}")


if __name__ == "__main__":
    main()

