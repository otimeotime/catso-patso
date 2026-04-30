#!/usr/bin/env python3
"""Plot a CATSO/PATSO experiment summary CSV.

Replaces plot.ipynb. Reads a single results_<env>_summary.csv and writes five
PDFs into an output directory.

Usage:
    python3 plot.py results_bettinggame_summary.csv
    python3 plot.py results_guarded_maze_summary.csv --outdir plots --title "Guarded Maze"
    python3 plot.py results_bettinggame_summary.csv --algos UCT CATSO PATSO

Notes:
- mc_cvar_stddev is only emitted by run_bettinggame and run_autonomous_vehicle.
  When absent, the mc_cvar plot is drawn without a stddev band (no error, just
  a bare line).
- Algorithms not present in the CSV are silently skipped.
"""

import argparse
import csv
import os
import shutil
import sys
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt


REQUIRED_COLUMNS = {
    "env",
    "algorithm",
    "trial",
    "mc_mean",
    "mc_stddev",
    "mc_cvar",
    "cvar_regret",
    "optimal_action_prob",
    "catastrophic_count",
}

# Optional columns: if present we draw stddev bands, otherwise we just draw the line.
OPTIONAL_COLUMNS = {"mc_cvar_stddev"}

DEFAULT_ALGOS = ["UCT", "CATSO", "PATSO"]
DEFAULT_COLORS = ["#1f77b4", "#d62728", "#9467bd"]


def configure_matplotlib() -> None:
    has_latex = shutil.which("latex") is not None

    rc = {
        "text.usetex": has_latex,
        "font.family": "serif",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.titlesize": 23,
        "axes.labelsize": 18,
        "legend.fontsize": 14,
        "legend.title_fontsize": 14,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
    }
    if has_latex:
        rc["text.latex.preamble"] = r"""
            \usepackage{newtxtext}
            \usepackage{newtxmath}
        """
    else:
        rc["mathtext.fontset"] = "stix"
        rc["font.serif"] = ["STIX Two Text", "Times New Roman", "DejaVu Serif"]

    mpl.rcParams.update(rc)


def load_rows(path: str) -> tuple[list[dict], set[str]]:
    if not os.path.isfile(path):
        sys.exit(f"error: input CSV not found: {path}")

    rows: list[dict] = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - fieldnames
        if missing:
            sys.exit(
                f"error: {path} is missing required columns: "
                f"{sorted(missing)}\n"
                f"found columns: {sorted(fieldnames)}"
            )

        present_optional = OPTIONAL_COLUMNS & fieldnames

        for r in reader:
            row = {
                "env": r["env"],
                "algorithm": r["algorithm"],
                "trial": int(r["trial"]),
                "mc_mean": float(r["mc_mean"]),
                "mc_stddev": float(r["mc_stddev"]),
                "mc_cvar": float(r["mc_cvar"]),
                "cvar_regret": float(r["cvar_regret"]),
                "optimal_action_prob": float(r["optimal_action_prob"]),
                "catastrophic_count": float(r["catastrophic_count"]),
            }
            if "mc_cvar_stddev" in present_optional:
                row["mc_cvar_stddev"] = float(r["mc_cvar_stddev"])
            rows.append(row)

    if not rows:
        sys.exit(f"error: {path} contained zero data rows")
    return rows, present_optional


def metric_rows(rows: list[dict], algorithm: str) -> list[dict]:
    return sorted(
        (r for r in rows if r["algorithm"] == algorithm),
        key=lambda r: r["trial"],
    )


def save_plot(
    rows: list[dict],
    algos: list[str],
    colors: list[str],
    metric: str,
    ylabel: str,
    out_path: str,
    title: str,
    legend_loc: str = "best",
    band_metric: Optional[str] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))

    drew_anything = False
    for alg, color in zip(algos, colors):
        alg_rows = metric_rows(rows, alg)
        if not alg_rows:
            continue
        drew_anything = True

        xs = [r["trial"] / 1000.0 for r in alg_rows]
        ys = [r[metric] for r in alg_rows]
        ax.plot(xs, ys, linewidth=2.0, label=alg, color=color)

        if band_metric is not None:
            yerr = [r[band_metric] for r in alg_rows]
            lower = [y - e for y, e in zip(ys, yerr)]
            upper = [y + e for y, e in zip(ys, yerr)]
            ax.fill_between(xs, lower, upper, alpha=0.15, color=color)

    if not drew_anything:
        plt.close(fig)
        print(f"  skip {os.path.basename(out_path)} -- no rows for any of {algos}")
        return

    ax.set_xlabel("Number of trials (x1000)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.legend(loc=legend_loc, frameon=True, framealpha=0.8)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a CATSO/PATSO experiment summary CSV.",
    )
    parser.add_argument(
        "csv",
        help="path to results_<env>_summary.csv (produced by an mcts-run-* binary)",
    )
    parser.add_argument(
        "--outdir",
        default="plots",
        help="directory to write the PDF plots into (default: plots/)",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="figure title (default: auto-generated from the env field in the CSV)",
    )
    parser.add_argument(
        "--algos",
        nargs="+",
        default=DEFAULT_ALGOS,
        help=f"algorithms to plot in order (default: {' '.join(DEFAULT_ALGOS)})",
    )
    parser.add_argument(
        "--colors",
        nargs="+",
        default=None,
        help="colors for each algorithm (must match --algos length); default uses tab10-style colors",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.colors is not None:
        if len(args.colors) != len(args.algos):
            sys.exit(
                f"error: --colors has {len(args.colors)} entries but --algos has {len(args.algos)}"
            )
        colors = args.colors
    else:
        colors = (DEFAULT_COLORS * ((len(args.algos) // len(DEFAULT_COLORS)) + 1))[
            : len(args.algos)
        ]

    rows, present_optional = load_rows(args.csv)
    env_name = rows[0]["env"]
    title = args.title if args.title is not None else f"Experiment - {env_name}"

    os.makedirs(args.outdir, exist_ok=True)
    configure_matplotlib()

    print(f"loaded {len(rows)} rows from {args.csv} (env='{env_name}')")
    if "mc_cvar_stddev" not in present_optional:
        print(
            "  note: mc_cvar_stddev column not present -- mc_cvar plot will have no stddev band"
        )

    cvar_band = "mc_cvar_stddev" if "mc_cvar_stddev" in present_optional else None

    plots = [
        # (metric, ylabel, suffix, legend_loc, band_metric)
        ("mc_mean", "Monte-Carlo value estimate", "", "lower right", "mc_stddev"),
        ("mc_cvar", "Monte-Carlo CVaR estimate", "_mc_cvar", "best", cvar_band),
        ("cvar_regret", "CVaR regret", "_cvar_regret", "best", None),
        ("optimal_action_prob", "P(optimal action hit)", "_optimal_action_prob", "best", None),
        ("catastrophic_count", "Catastrophic count", "_catastrophic_count", "best", None),
    ]

    for metric, ylabel, suffix, legend_loc, band_metric in plots:
        out_path = os.path.join(args.outdir, f"{env_name}{suffix}.pdf")
        save_plot(
            rows=rows,
            algos=args.algos,
            colors=colors,
            metric=metric,
            ylabel=ylabel,
            out_path=out_path,
            title=title,
            legend_loc=legend_loc,
            band_metric=band_metric,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
