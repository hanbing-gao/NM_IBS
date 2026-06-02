"""Run equivalence tests with original and additional SESOI values.

This reproduces the primary group and group-by-sex equivalence tests from
``p7_EQ_test_with_OLS.ipynb`` while adding an additional reviewer-facing SESOI
of 0.15. The OLS models are fitted directly from the all-predicted deviation
score exports so no ROI is removed by the historical normative-model evaluation
filter.

Example:
    python scripts/p5_equivalence_dual_sesoi.py --input-dir data/processed --output-dir results/equivalence_dual_SESOI
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ORDER_OUT: Path | None = None
REV_OUT = PACKAGE_ROOT / "results" / "equivalence_dual_SESOI"
FIG_DIR = REV_OUT / "figures"
HC_COV_PATH = PACKAGE_ROOT / "data" / "processed" / "HC_cov_ROME.csv"
IBS_COV_PATH = PACKAGE_ROOT / "data" / "processed" / "IBS_cov_ROME.csv"
HC_DEV_PATH = PACKAGE_ROOT / "data" / "processed" / "HC_deviation_scores_by_brain_idp_all_predicted.csv"
IBS_DEV_PATH = PACKAGE_ROOT / "data" / "processed" / "IBS_deviation_scores_by_brain_idp_all_predicted.csv"

MODALITIES = ["CT", "SA", "CV"]
TESTS = {
    "group": {
        "order_input": "group_EQ_test_results.csv",
        "label": "IBS vs HC main effect",
        "term": "Group[T.IBS]",
        "old_sesoi": 0.088,
        "new_sesoi": 0.15,
    },
    "group_sex": {
        "order_input": "group_sex_EQ_test_results.csv",
        "label": "IBS-HC by sex interaction",
        "term": "Group[T.IBS]:C(sex, Sum)[S.Female]",
        "old_sesoi": 0.099,
        "new_sesoi": 0.15,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run equivalence tests for primary group and group-by-sex OLS effects."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PACKAGE_ROOT / "data" / "processed",
        help=(
            "Directory containing HC_cov_ROME.csv, IBS_cov_ROME.csv, "
            "HC_deviation_scores_by_brain_idp_all_predicted.csv, and "
            "IBS_deviation_scores_by_brain_idp_all_predicted.csv."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PACKAGE_ROOT / "results" / "equivalence_dual_SESOI",
        help="Directory for equivalence-test tables and figures.",
    )
    parser.add_argument(
        "--roi-order-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory with historical equivalence outputs used only to reproduce "
            "the old ROI plotting order. If omitted, exported ROI order is used."
        ),
    )
    return parser.parse_args()


def configure_paths(args: argparse.Namespace) -> None:
    global ORDER_OUT, REV_OUT, FIG_DIR, HC_COV_PATH, IBS_COV_PATH, HC_DEV_PATH, IBS_DEV_PATH
    input_dir = args.input_dir.resolve()
    ORDER_OUT = args.roi_order_dir.resolve() if args.roi_order_dir is not None else None
    REV_OUT = args.output_dir.resolve()
    FIG_DIR = REV_OUT / "figures"
    HC_COV_PATH = input_dir / "HC_cov_ROME.csv"
    IBS_COV_PATH = input_dir / "IBS_cov_ROME.csv"
    HC_DEV_PATH = input_dir / "HC_deviation_scores_by_brain_idp_all_predicted.csv"
    IBS_DEV_PATH = input_dir / "IBS_deviation_scores_by_brain_idp_all_predicted.csv"


def tost_from_beta(beta, se, low, high, df, alpha=0.05):
    """Two one-sided tests for an OLS coefficient with known SE and df."""
    if not np.isfinite(beta) or not np.isfinite(se) or se <= 0 or not np.isfinite(df):
        return {
            "p_lower": np.nan,
            "p_upper": np.nan,
            "p_max": np.nan,
            "equivalent": False,
            "ci_lo": np.nan,
            "ci_hi": np.nan,
        }

    t1 = (beta - low) / se
    t2 = (high - beta) / se
    p1 = 1 - stats.t.cdf(t1, df)
    p2 = 1 - stats.t.cdf(t2, df)
    tcrit_90 = stats.t.ppf(0.95, df)

    return {
        "p_lower": float(p1),
        "p_upper": float(p2),
        "p_max": float(max(p1, p2)),
        "equivalent": bool((p1 < alpha) and (p2 < alpha)),
        "ci_lo": float(beta - tcrit_90 * se),
        "ci_hi": float(beta + tcrit_90 * se),
    }


def recode_sex(series):
    """Use the same labels as the original effect-coded OLS notebook."""
    sex_map = {0: "Female", 1: "Male", "0": "Female", "1": "Male"}
    recoded = series.map(sex_map)
    recoded = recoded.fillna(series.astype(str).str.strip().str.title())
    return pd.Categorical(recoded, categories=["Female", "Male"])


def load_group_data(cov_path, dev_path, group_label):
    cov = pd.read_csv(cov_path, usecols=["eid", "sex"]).copy()
    dev = pd.read_csv(dev_path)
    cov["Group"] = group_label
    cov["sex"] = recode_sex(cov["sex"])
    merged = cov[["eid", "Group", "sex"]].merge(dev, on="eid", how="inner")
    if merged.empty:
        raise ValueError(f"No rows remained after merging {cov_path.name} with {dev_path.name}.")
    return merged


def load_model_data():
    hc = load_group_data(HC_COV_PATH, HC_DEV_PATH, "HC")
    ibs = load_group_data(IBS_COV_PATH, IBS_DEV_PATH, "IBS")
    data = pd.concat([hc, ibs], ignore_index=True)
    data["Group"] = pd.Categorical(data["Group"], categories=["HC", "IBS"])
    return data


def idp_columns_for(data, modality):
    prefix = f"{modality}__"
    cols = [col for col in data.columns if col.startswith(prefix)]
    if len(cols) != 60:
        raise ValueError(f"Expected 60 {modality} IDPs, found {len(cols)}.")
    return cols


def extract_model_term(res, expected_term, test_key):
    if expected_term in res.params.index:
        return expected_term

    if test_key == "group":
        candidates = [term for term in res.params.index if term.startswith("Group[T.IBS]")]
    else:
        candidates = [
            term
            for term in res.params.index
            if term.startswith("Group[T.IBS]:C(sex, Sum)")
        ]
    if len(candidates) != 1:
        raise KeyError(
            f"Could not identify the {test_key} coefficient. "
            f"Expected {expected_term!r}; available terms are {list(res.params.index)!r}."
        )
    return candidates[0]


def fit_ols_terms_from_all_predicted():
    """Fit ROI ~ Group*C(sex, Sum) for every exported IDP and save coefficients."""
    data = load_model_data()
    rows = []

    for modality in MODALITIES:
        for col in idp_columns_for(data, modality):
            roi = col.split("__", 1)[1]
            sub = data[["eid", "Group", "sex", col]].dropna().copy()
            n = len(sub)
            if n < 5 or sub[col].nunique() <= 1:
                for test_key, cfg in TESTS.items():
                    rows.append(
                        {
                            "modality": modality,
                            "roi": roi,
                            "idp_column": col,
                            "n": n,
                            "df_resid": np.nan,
                            "test": test_key,
                            "term": cfg["term"],
                            "beta": np.nan,
                            "se": np.nan,
                            "ci_lower": np.nan,
                            "ci_upper": np.nan,
                            "t": np.nan,
                            "p": np.nan,
                            "residual_var": np.nan,
                        }
                    )
                continue

            res = smf.ols(f'Q("{col}") ~ Group*C(sex, Sum)', data=sub).fit()
            conf_int = res.conf_int()
            for test_key, cfg in TESTS.items():
                term = extract_model_term(res, cfg["term"], test_key)
                rows.append(
                    {
                        "modality": modality,
                        "roi": roi,
                        "idp_column": col,
                        "n": n,
                        "df_resid": float(res.df_resid),
                        "test": test_key,
                        "term": term,
                        "beta": float(res.params[term]),
                        "se": float(res.bse[term]),
                        "ci_lower": float(conf_int.loc[term, 0]),
                        "ci_upper": float(conf_int.loc[term, 1]),
                        "t": float(res.tvalues[term]),
                        "p": float(res.pvalues[term]),
                        "residual_var": float(res.scale),
                    }
                )

    ols_terms = pd.DataFrame(rows)
    ols_terms.to_csv(REV_OUT / "ols_terms_all_predicted_all60.csv", index=False)
    return ols_terms


def run_equivalence_tests():
    ols_terms = fit_ols_terms_from_all_predicted()
    rows = []
    for modality in MODALITIES:
        for test_key, cfg in TESTS.items():
            ols = ols_terms[
                (ols_terms["modality"] == modality) & (ols_terms["test"] == test_key)
            ].copy()
            for sesoi_label, sesoi in [
                ("strict", cfg["old_sesoi"]),
                ("sesoi_0p15", cfg["new_sesoi"]),
            ]:
                test_rows = []
                for _, row in ols.iterrows():
                    test_rows.append(
                        {
                            "modality": modality,
                            "test": test_key,
                            "test_label": cfg["label"],
                            "roi": row["roi"],
                            "beta": row["beta"],
                            "se": row["se"],
                            "df_resid": row["df_resid"],
                            "sesoi_label": sesoi_label,
                            "sesoi": sesoi,
                            **tost_from_beta(row["beta"], row["se"], -sesoi, sesoi, row["df_resid"]),
                        }
                    )

                result = pd.DataFrame(test_rows)
                valid = result["p_max"].notna()
                result["p_max_fdr"] = np.nan
                result["equivalent_fdr05"] = False
                if valid.any():
                    reject, p_fdr, _, _ = multipletests(
                        result.loc[valid, "p_max"], alpha=0.05, method="fdr_bh"
                    )
                    result.loc[valid, "p_max_fdr"] = p_fdr
                    result.loc[valid, "equivalent_fdr05"] = reject
                rows.append(result)

    all_results = pd.concat(rows, ignore_index=True)
    all_results.to_csv(REV_OUT / "equivalence_dual_SESOI_all_results.csv", index=False)
    return all_results


def make_plot_classification(all_results):
    rows = []
    for (modality, test_key, roi), sub in all_results.groupby(["modality", "test", "roi"]):
        strict = sub[sub["sesoi_label"] == "strict"].iloc[0]
        new = sub[sub["sesoi_label"] == "sesoi_0p15"].iloc[0]
        if strict["equivalent_fdr05"]:
            classification = "Equivalent under original strict SESOI"
        elif new["equivalent_fdr05"]:
            classification = "Equivalent only under SESOI 0.15"
        else:
            classification = "Not equivalent under SESOI 0.15"

        rows.append(
            {
                "modality": modality,
                "test": test_key,
                "roi": roi,
                "beta": strict["beta"],
                "ci_lo": strict["ci_lo"],
                "ci_hi": strict["ci_hi"],
                "strict_sesoi": strict["sesoi"],
                "new_sesoi": new["sesoi"],
                "strict_p_fdr": strict["p_max_fdr"],
                "new_p_fdr": new["p_max_fdr"],
                "strict_equivalent_fdr05": bool(strict["equivalent_fdr05"]),
                "new_equivalent_fdr05": bool(new["equivalent_fdr05"]),
                "classification": classification,
            }
        )

    plot_df = pd.DataFrame(rows)
    plot_df.to_csv(REV_OUT / "equivalence_dual_SESOI_plot_classification.csv", index=False)

    summary = (
        all_results.groupby(["test", "modality", "sesoi_label", "sesoi"], as_index=False)
        .agg(
            n_idp=("roi", "count"),
            n_equivalent_fdr05=("equivalent_fdr05", "sum"),
            min_p_fdr=("p_max_fdr", "min"),
            max_abs_beta=("beta", lambda x: float(np.nanmax(np.abs(x)))),
        )
    )
    summary.to_csv(REV_OUT / "equivalence_dual_SESOI_summary.csv", index=False)
    return plot_df, summary


COLORS = {
    "Equivalent under original strict SESOI": "#111111",
    "Equivalent only under SESOI 0.15": "#1f77b4",
    "Not equivalent under SESOI 0.15": "#d62728",
}


def format_sesoi(value):
    return f"{value:.3f}".rstrip("0").rstrip(".")


def roi_order_for(test_key, plot_df):
    """Match the original equivalence figures' left/right paired ROI order."""
    order_path = None if ORDER_OUT is None else ORDER_OUT / "SA" / TESTS[test_key]["order_input"]
    if order_path is not None and order_path.exists():
        order = pd.read_csv(order_path)["ROI"].tolist()
    else:
        order = plot_df[(plot_df["modality"] == "SA") & (plot_df["test"] == test_key)][
            "roi"
        ].tolist()

    available = set(plot_df.loc[plot_df["test"] == test_key, "roi"])
    ordered_available = [roi for roi in order if roi in available]
    missing_from_order = sorted(available.difference(ordered_available))
    return ordered_available + missing_from_order


def panel_plot(ax, data, roi_order, title, show_ylabels=False):
    data = data.set_index("roi").reindex(roi_order).reset_index()
    y = np.arange(len(data))

    for i, row in data.iterrows():
        if pd.isna(row["beta"]):
            continue
        color = COLORS[row["classification"]]
        ax.plot([row["ci_lo"], row["ci_hi"]], [i, i], color=color, linewidth=1.6)
        ax.plot(
            row["beta"],
            i,
            marker="D",
            markerfacecolor="white",
            markeredgecolor=color,
            markersize=4.5,
            markeredgewidth=1.2,
        )

    strict = data["strict_sesoi"].dropna().iloc[0]
    new = data["new_sesoi"].dropna().iloc[0]
    ax.axvline(0, color="0.65", linewidth=0.8)
    ax.axvline(-strict, color="black", linestyle="--", linewidth=0.9)
    ax.axvline(strict, color="black", linestyle="--", linewidth=0.9)
    ax.axvline(-new, color="#1f77b4", linestyle=":", linewidth=1.2)
    ax.axvline(new, color="#1f77b4", linestyle=":", linewidth=1.2)

    ax.set_title(title, fontsize=11, pad=6)
    ax.set_xlim(-0.18, 0.18)
    ax.set_ylim(len(data) - 0.5, -0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(data["roi"] if show_ylabels else [], fontsize=5.5)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="0.92", linewidth=0.6)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("0.2")


def sesoi_handles(strict_label):
    return [
        plt.Line2D([0], [0], color="black", linestyle="--", linewidth=1.0, label=strict_label),
        plt.Line2D([0], [0], color="#1f77b4", linestyle=":", linewidth=1.4, label="SESOI = +/-0.15"),
    ]


def save_combined_figure(plot_df):
    handles = sesoi_handles("SESOI = +/-0.088 (main effect), +/-0.099 (interaction)")

    fig, axes = plt.subplots(2, 3, figsize=(13, 19), dpi=300, sharex=True)
    for row_idx, test_key in enumerate(["group", "group_sex"]):
        roi_order = roi_order_for(test_key, plot_df)
        for col_idx, modality in enumerate(MODALITIES):
            panel_plot(
                axes[row_idx, col_idx],
                plot_df[(plot_df["test"] == test_key) & (plot_df["modality"] == modality)],
                roi_order,
                modality,
                show_ylabels=(col_idx == 0),
            )
        axes[row_idx, 0].set_ylabel(TESTS[test_key]["label"], fontsize=12, labelpad=12)
    for ax in axes[-1, :]:
        ax.set_xlabel("Effect size with 90% CI", fontsize=10)
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.suptitle("Equivalence tests using two SESOI thresholds", fontsize=14, y=0.995)
    fig.tight_layout(rect=[0, 0.045, 1, 0.985])
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(FIG_DIR / f"Fig2_equivalence_dual_SESOI_replacement.{ext}", bbox_inches="tight")
    plt.close(fig)


def save_test_figures(plot_df):
    for test_key, cfg in TESTS.items():
        roi_order = roi_order_for(test_key, plot_df)
        handles = sesoi_handles(f"SESOI = +/-{format_sesoi(cfg['old_sesoi'])}")
        fig, axes = plt.subplots(1, 3, figsize=(13, 10), dpi=300, sharex=True)
        for col_idx, modality in enumerate(MODALITIES):
            panel_plot(
                axes[col_idx],
                plot_df[(plot_df["test"] == test_key) & (plot_df["modality"] == modality)],
                roi_order,
                modality,
                show_ylabels=(col_idx == 0),
            )
            axes[col_idx].set_xlabel("Effect size with 90% CI", fontsize=10)
        axes[0].set_ylabel(cfg["label"], fontsize=12, labelpad=12)
        fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=9)
        fig.suptitle(f"{cfg['label']} equivalence tests", fontsize=14, y=0.995)
        fig.tight_layout(rect=[0, 0.055, 1, 0.975])
        for ext in ["png", "pdf", "svg"]:
            fig.savefig(FIG_DIR / f"{test_key}_equivalence_dual_SESOI.{ext}", bbox_inches="tight")
        plt.close(fig)


def save_modality_figures(plot_df):
    for test_key, cfg in TESTS.items():
        for modality in MODALITIES:
            data = plot_df[(plot_df["test"] == test_key) & (plot_df["modality"] == modality)]
            roi_order = roi_order_for(test_key, plot_df)
            handles = sesoi_handles(f"SESOI = +/-{format_sesoi(cfg['old_sesoi'])}")
            fig, ax = plt.subplots(1, 1, figsize=(7.5, 10), dpi=300)
            panel_plot(ax, data, roi_order, modality, show_ylabels=True)
            ax.set_xlabel("Effect size with 90% CI", fontsize=10)
            ax.set_ylabel(cfg["label"], fontsize=12, labelpad=12)
            fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=9)
            fig.tight_layout(rect=[0, 0.055, 1, 0.97])
            for ext in ["png", "pdf", "svg"]:
                fig.savefig(
                    FIG_DIR / f"{modality}_{test_key}_equivalence_dual_SESOI.{ext}",
                    bbox_inches="tight",
                )
            plt.close(fig)


def save_figures(plot_df):
    save_combined_figure(plot_df)
    save_test_figures(plot_df)
    save_modality_figures(plot_df)


def main():
    args = parse_args()
    configure_paths(args)
    REV_OUT.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    all_results = run_equivalence_tests()
    plot_df, summary = make_plot_classification(all_results)
    save_figures(plot_df)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
