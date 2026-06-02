"""Create Supplementary Figure 2: DHQ-MRI timing distribution.

The figure uses the matched HC and IBS covariate files prepared for the
NeuroImage revision. The signed interval is computed as:

    DHQ completion date - MRI assessment date

Positive values therefore indicate DHQ completion after MRI, and negative
values indicate DHQ completion before MRI.

Example:
    python scripts/p9_dhq_mri_timing_distribution.py --input-dir data/processed --output-dir results/DHQ_MRI_timing_distribution
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PACKAGE_ROOT / "results" / "DHQ_MRI_timing_distribution"
HC_COV_PATH = PACKAGE_ROOT / "data" / "processed" / "HC_cov_ROME.csv"
IBS_COV_PATH = PACKAGE_ROOT / "data" / "processed" / "IBS_cov_ROME.csv"

GROUPS = {
    "HC": {
        "path": HC_COV_PATH,
        "label": "HC",
        "color": "#4C78A8",
    },
    "IBS": {
        "path": IBS_COV_PATH,
        "label": "IBS",
        "color": "#D55E00",
    },
}

CAPTION = (
    "Supplementary Figure 2. DHQ-MRI timing distribution in the matched "
    "case-control sample. The figure shows the distribution of signed DHQ-MRI "
    "intervals for IBS and HC participants. Positive values indicate DHQ "
    "completion after MRI, whereas negative values indicate DHQ completion "
    "before MRI. Dashed vertical lines indicate +/-3 years."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot signed DHQ-MRI timing intervals in the matched HC/IBS sample."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PACKAGE_ROOT / "data" / "processed",
        help="Directory containing HC_cov_ROME.csv and IBS_cov_ROME.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PACKAGE_ROOT / "results" / "DHQ_MRI_timing_distribution",
    )
    return parser.parse_args()


def configure_paths(args: argparse.Namespace) -> None:
    global OUT_DIR, HC_COV_PATH, IBS_COV_PATH, GROUPS
    input_dir = args.input_dir.resolve()
    OUT_DIR = args.output_dir.resolve()
    HC_COV_PATH = input_dir / "HC_cov_ROME.csv"
    IBS_COV_PATH = input_dir / "IBS_cov_ROME.csv"
    GROUPS = {
        "HC": {"path": HC_COV_PATH, "label": "HC", "color": "#4C78A8"},
        "IBS": {"path": IBS_COV_PATH, "label": "IBS", "color": "#D55E00"},
    }


def load_interval_data() -> pd.DataFrame:
    frames = []
    for group, cfg in GROUPS.items():
        df = pd.read_csv(cfg["path"], usecols=["eid", "mri_date", "DHQ_date"])
        df["group"] = group
        df["mri_date"] = pd.to_datetime(df["mri_date"], errors="coerce")
        df["DHQ_date"] = pd.to_datetime(df["DHQ_date"], errors="coerce")
        df["DHQ_MRI_interval_days"] = (df["DHQ_date"] - df["mri_date"]).dt.days
        df["DHQ_MRI_interval_years"] = df["DHQ_MRI_interval_days"] / 365.25
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    data.to_csv(OUT_DIR / "DHQ_MRI_timing_distribution_data.csv", index=False)
    return data


def make_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group, sub in data.groupby("group", sort=False):
        values = sub["DHQ_MRI_interval_years"].dropna()
        rows.append(
            {
                "group": group,
                "n_total": len(sub),
                "n_with_DHQ_MRI_interval": len(values),
                "n_missing_DHQ_MRI_interval": int(sub["DHQ_MRI_interval_years"].isna().sum()),
                "mean_years": values.mean(),
                "sd_years": values.std(ddof=1),
                "median_years": values.median(),
                "q1_years": values.quantile(0.25),
                "q3_years": values.quantile(0.75),
                "min_years": values.min(),
                "max_years": values.max(),
                "n_before_MRI": int((values < 0).sum()),
                "n_after_MRI": int((values > 0).sum()),
                "n_same_day": int((values == 0).sum()),
                "n_within_3_years": int((values.abs() <= 3).sum()),
                "pct_within_3_years": 100 * float((values.abs() <= 3).mean()),
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "DHQ_MRI_timing_distribution_summary.csv", index=False)
    return summary


def plot_distribution(data: pd.DataFrame, summary: pd.DataFrame) -> plt.Figure:
    plot_data = data.dropna(subset=["DHQ_MRI_interval_years"]).copy()
    values = plot_data["DHQ_MRI_interval_years"]
    bins = np.arange(
        np.floor(values.min() * 2) / 2,
        np.ceil(values.max() * 2) / 2 + 0.25,
        0.25,
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.5), dpi=300)

    for group, cfg in GROUPS.items():
        group_values = plot_data.loc[plot_data["group"] == group, "DHQ_MRI_interval_years"]
        row = summary.loc[summary["group"] == group].iloc[0]
        ax.hist(
            group_values,
            bins=bins,
            density=True,
            histtype="stepfilled",
            alpha=0.26,
            color=cfg["color"],
            edgecolor=cfg["color"],
            linewidth=1.1,
        )
        ax.hist(
            group_values,
            bins=bins,
            density=True,
            histtype="step",
            color=cfg["color"],
            linewidth=1.5,
        )

    for x in [-3, 3]:
        ax.axvline(x, color="0.15", linestyle="--", linewidth=1.1)
    ax.axvline(0, color="0.55", linestyle="-", linewidth=0.8)

    ax.set_xlabel("Signed DHQ-MRI interval (years)")
    ax.set_ylabel("Density")
    ax.set_xlim(-5.25, 4.0)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.grid(axis="y", color="0.90", linewidth=0.7)

    handles = []
    for group, cfg in GROUPS.items():
        row = summary.loc[summary["group"] == group].iloc[0]
        handles.append(
            Patch(
                facecolor=cfg["color"],
                edgecolor=cfg["color"],
                alpha=0.26,
                label=f"{cfg['label']} (n = {int(row['n_with_DHQ_MRI_interval']):,})",
            )
        )
    handles.append(Line2D([0], [0], color="0.15", linestyle="--", linewidth=1.1, label=r"$\pm$3 years"))
    ax.legend(handles=handles, frameon=False, loc="upper left", fontsize=8.5)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("0.2")

    fig.tight_layout()
    return fig


def save_figure(fig: plt.Figure) -> None:
    stems = [OUT_DIR / "Supplementary_Figure_2_DHQ_MRI_timing_distribution"]
    for stem in stems:
        for ext in ["png", "pdf", "svg", "tif"]:
            fig.savefig(stem.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
    for path in [OUT_DIR / "Supplementary_Figure_2_DHQ_MRI_timing_distribution_caption.txt"]:
        path.write_text(CAPTION, encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_paths(args)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_interval_data()
    summary = make_summary(data)
    fig = plot_distribution(data, summary)
    save_figure(fig)
    plt.close(fig)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
