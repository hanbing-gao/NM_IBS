"""Compare raw-IDP GLM effect sizes with NM-deviation OLS effect sizes.

This sensitivity analysis fits conventional raw-IDP models on the same matched
HC and IBS participants used in the NM primary analysis:

    z(raw IDP) ~ Group * C(sex, Sum) + age + C(site)

The raw IDPs are z-standardized within the matched HC+IBS analysis sample so
the group and group-by-sex coefficients are comparable across ROIs/modalities.
The script then compares these raw-IDP coefficients with the complete all-60
ROI NM-deviation primary OLS coefficients from
``p4_primary_ols.py``.

Example:
    python scripts/p8_raw_vs_nm_sensitivity.py \
        --processed-dir data/processed \
        --nm-primary-dir results/primary_OLS_all60 \
        --output-dir results/raw_vs_NM_sensitivity
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PACKAGE_ROOT / "data" / "raw_ukb_exports"
DOC_DIR = PACKAGE_ROOT / "docs"
NM_OUT = PACKAGE_ROOT / "results" / "primary_OLS_all60"
OUT_DIR = PACKAGE_ROOT / "results" / "raw_vs_NM_sensitivity"
FIG_DIR = OUT_DIR / "figures"

HC_RAW_PATH = DATA_DIR / "HC_MRI_age.csv"
IBS_RAW_PATH = DATA_DIR / "IBS_MRI_age.csv"
HC_COV_PATH = PACKAGE_ROOT / "data" / "processed" / "HC_cov_ROME.csv"
IBS_COV_PATH = PACKAGE_ROOT / "data" / "processed" / "IBS_cov_ROME.csv"
ROI_PATH = DOC_DIR / "ROI_IBS.csv"

MODALITIES = ["CT", "SA", "CV"]
TERM_ORDER = [
    "Group[T.IBS]",
    "Group[T.IBS]:C(sex, Sum)[S.Female]",
]
TERM_LABELS = {
    "Group[T.IBS]": "IBS vs HC main effect",
    "Group[T.IBS]:C(sex, Sum)[S.Female]": "IBS-HC by sex interaction",
}
MODALITY_COLORS = {
    "CT": "#0072B2",
    "SA": "#009E73",
    "CV": "#D55E00",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare raw-IDP adjusted effects with NM-deviation primary effects."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory containing separate HC_MRI_age.csv and IBS_MRI_age.csv "
            "UKB-derived exports. If omitted, uses HC_cov_ROME.csv and IBS_cov_ROME.csv "
            "from --processed-dir."
        ),
    )
    parser.add_argument("--hc-raw-file", default=None)
    parser.add_argument("--ibs-raw-file", default=None)
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PACKAGE_ROOT / "data" / "processed",
        help="Directory containing matched HC_cov_ROME.csv and IBS_cov_ROME.csv.",
    )
    parser.add_argument("--docs-dir", type=Path, default=PACKAGE_ROOT / "docs")
    parser.add_argument(
        "--nm-primary-dir",
        type=Path,
        default=PACKAGE_ROOT / "results" / "primary_OLS_all60",
        help="Directory containing primary_OLS_all60_all_terms.csv from p4_primary_ols.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PACKAGE_ROOT / "results" / "raw_vs_NM_sensitivity",
    )
    return parser.parse_args()


def configure_paths(args: argparse.Namespace) -> None:
    global DATA_DIR, DOC_DIR, NM_OUT, OUT_DIR, FIG_DIR
    global HC_RAW_PATH, IBS_RAW_PATH, HC_COV_PATH, IBS_COV_PATH, ROI_PATH
    DOC_DIR = args.docs_dir.resolve()
    NM_OUT = args.nm_primary_dir.resolve()
    OUT_DIR = args.output_dir.resolve()
    FIG_DIR = OUT_DIR / "figures"
    processed_dir = args.processed_dir.resolve()
    HC_COV_PATH = processed_dir / "HC_cov_ROME.csv"
    IBS_COV_PATH = processed_dir / "IBS_cov_ROME.csv"
    if args.raw_dir is None:
        DATA_DIR = processed_dir
        HC_RAW_PATH = HC_COV_PATH
        IBS_RAW_PATH = IBS_COV_PATH
    else:
        DATA_DIR = args.raw_dir.resolve()
        HC_RAW_PATH = DATA_DIR / (args.hc_raw_file or "HC_MRI_age.csv")
        IBS_RAW_PATH = DATA_DIR / (args.ibs_raw_file or "IBS_MRI_age.csv")
    ROI_PATH = DOC_DIR / "ROI_IBS.csv"


def recode_sex(series):
    sex_map = {0: "Female", 1: "Male", "0": "Female", "1": "Male"}
    recoded = series.map(sex_map)
    recoded = recoded.fillna(series.astype(str).str.strip().str.title())
    return pd.Categorical(recoded, categories=["Female", "Male"])


def load_roi_list():
    return pd.read_csv(ROI_PATH)["ROI"].tolist()


def sort_idp_rows(df, term_order=None):
    """Sort modalities and ROIs in the manuscript/table display order."""
    roi_order = {roi: idx for idx, roi in enumerate(load_roi_list())}
    modality_order = {modality: idx for idx, modality in enumerate(MODALITIES)}
    out = df.copy()
    out["_modality_order"] = out["modality"].map(modality_order)
    out["_roi_order"] = out["roi"].map(roi_order)
    sort_cols = ["_modality_order"]
    if term_order is not None and "term" in out.columns:
        out["_term_order"] = out["term"].map({term: idx for idx, term in enumerate(term_order)})
        sort_cols.append("_term_order")
    sort_cols.append("_roi_order")
    out = out.sort_values(sort_cols, kind="mergesort").drop(
        columns=[col for col in ["_modality_order", "_term_order", "_roi_order"] if col in out.columns]
    )
    return out.reset_index(drop=True)


def load_idp_mapping(modality, roi_list):
    mapping = pd.read_csv(DOC_DIR / f"aseg_2009_{modality}_formatted.csv")
    mapping = mapping.rename(
        columns={"idx_number": "field_id", "formatted_name": "roi", "name": "field_name"}
    )
    mapping = mapping[mapping["roi"].isin(roi_list)].copy()
    mapping["raw_column"] = mapping["field_id"].astype(int).astype(str) + "-2.0"
    missing = sorted(set(roi_list) - set(mapping["roi"]))
    if missing:
        raise ValueError(f"{modality} mapping is missing ROI names: {missing}")
    return mapping[["roi", "raw_column", "field_id", "field_name"]]


def load_all_mappings():
    roi_list = load_roi_list()
    frames = []
    for modality in MODALITIES:
        mapping = load_idp_mapping(modality, roi_list)
        mapping["modality"] = modality
        frames.append(mapping)
    mapping = pd.concat(frames, ignore_index=True)
    if mapping.groupby("modality")["roi"].nunique().ne(60).any():
        counts = mapping.groupby("modality")["roi"].nunique().to_dict()
        raise ValueError(f"Expected 60 mapped ROIs per modality, got {counts}")
    return sort_idp_rows(mapping)


def matched_eids():
    hc = pd.read_csv(HC_COV_PATH, usecols=["eid"])
    ibs = pd.read_csv(IBS_COV_PATH, usecols=["eid"])
    return set(hc["eid"]), set(ibs["eid"])


def load_raw_group(path, group_label, eids, raw_columns):
    header = pd.read_csv(path, nrows=0).columns
    covariates = {
        "sex": "31-0.0" if "31-0.0" in header else "sex",
        "site": "54-2.0" if "54-2.0" in header else "site",
        "age": "21003-2.0" if "21003-2.0" in header else "age",
    }
    missing_cov = [source for source in covariates.values() if source not in header]
    missing_raw = [column for column in raw_columns if column not in header]
    if missing_cov or missing_raw:
        raise ValueError(
            f"{path} is missing required columns. "
            f"Missing covariates: {missing_cov}; missing raw IDPs: {missing_raw[:10]}"
            f"{'...' if len(missing_raw) > 10 else ''}"
        )
    usecols = ["eid", covariates["sex"], covariates["site"], covariates["age"]] + raw_columns
    df = pd.read_csv(path, usecols=usecols)
    df = df[df["eid"].isin(eids)].copy()
    df["analysis_group"] = group_label
    df = df.rename(
        columns={
            covariates["sex"]: "sex",
            covariates["site"]: "site",
            covariates["age"]: "age",
        }
    )
    df["sex"] = recode_sex(df["sex"])
    df["site"] = df["site"].astype("category")
    df["analysis_group"] = pd.Categorical(df["analysis_group"], categories=["HC", "IBS"])
    return df


def load_raw_analysis_data(mapping):
    raw_columns = sorted(mapping["raw_column"].unique().tolist())
    hc_eids, ibs_eids = matched_eids()
    hc = load_raw_group(HC_RAW_PATH, "HC", hc_eids, raw_columns)
    ibs = load_raw_group(IBS_RAW_PATH, "IBS", ibs_eids, raw_columns)
    data = pd.concat([hc, ibs], ignore_index=True)
    data["analysis_group"] = pd.Categorical(data["analysis_group"], categories=["HC", "IBS"])
    data["site"] = data["site"].astype("category")
    return data


def fit_one_raw_idp(data, modality, roi, raw_col):
    sub = data[["eid", "analysis_group", "sex", "age", "site", raw_col]].dropna().copy()
    n = len(sub)
    if n < 5 or sub[raw_col].nunique() <= 1:
        return [empty_term_row(modality, roi, raw_col, n, term) for term in TERM_ORDER]

    raw_mean = sub[raw_col].mean()
    raw_sd = sub[raw_col].std(ddof=1)
    if not np.isfinite(raw_sd) or raw_sd <= 0:
        return [empty_term_row(modality, roi, raw_col, n, term) for term in TERM_ORDER]

    sub["raw_z"] = (sub[raw_col] - raw_mean) / raw_sd
    res = smf.ols("raw_z ~ analysis_group*C(sex, Sum) + age + C(site)", data=sub).fit()
    conf_int = res.conf_int()
    tcrit_90 = stats.t.ppf(0.95, res.df_resid)

    available_terms = {
        "Group[T.IBS]": "analysis_group[T.IBS]",
        "Group[T.IBS]:C(sex, Sum)[S.Female]": (
            "analysis_group[T.IBS]:C(sex, Sum)[S.Female]"
        ),
    }
    rows = []
    for standardized_term, model_term in available_terms.items():
        rows.append(
            {
                "modality": modality,
                "roi": roi,
                "raw_column": raw_col,
                "n": n,
                "df_resid": float(res.df_resid),
                "term": standardized_term,
                "model_term": model_term,
                "beta": float(res.params[model_term]),
                "se": float(res.bse[model_term]),
                "ci_lower": float(conf_int.loc[model_term, 0]),
                "ci_upper": float(conf_int.loc[model_term, 1]),
                "t": float(res.tvalues[model_term]),
                "p": float(res.pvalues[model_term]),
                "Residual_var": float(res.scale),
                "ci_lower_95": float(res.params[model_term] - tcrit_90 * res.bse[model_term]),
                "ci_higher_95": float(res.params[model_term] + tcrit_90 * res.bse[model_term]),
                "raw_mean": float(raw_mean),
                "raw_sd": float(raw_sd),
            }
        )
    return rows


def empty_term_row(modality, roi, raw_col, n, term):
    return {
        "modality": modality,
        "roi": roi,
        "raw_column": raw_col,
        "n": n,
        "df_resid": np.nan,
        "term": term,
        "model_term": np.nan,
        "beta": np.nan,
        "se": np.nan,
        "ci_lower": np.nan,
        "ci_upper": np.nan,
        "t": np.nan,
        "p": np.nan,
        "Residual_var": np.nan,
        "ci_lower_95": np.nan,
        "ci_higher_95": np.nan,
        "raw_mean": np.nan,
        "raw_sd": np.nan,
    }


def add_fdr(raw_terms):
    raw_terms = raw_terms.copy()
    raw_terms["p_fdr_bh"] = np.nan
    for (modality, term), idx in raw_terms.groupby(["modality", "term"]).groups.items():
        mask = raw_terms.index.isin(idx) & raw_terms["p"].notna()
        if mask.any():
            _, pvals_fdr, _, _ = multipletests(raw_terms.loc[mask, "p"], alpha=0.05, method="fdr_bh")
            raw_terms.loc[mask, "p_fdr_bh"] = pvals_fdr
    return raw_terms


def run_raw_models(data, mapping):
    rows = []
    for _, row in sort_idp_rows(mapping).iterrows():
        rows.extend(fit_one_raw_idp(data, row["modality"], row["roi"], row["raw_column"]))
    raw_terms = sort_idp_rows(add_fdr(pd.DataFrame(rows)), TERM_ORDER)
    raw_terms.to_csv(OUT_DIR / "raw_IDP_GLM_all_terms.csv", index=False)
    raw_terms.to_excel(OUT_DIR / "raw_IDP_GLM_all_terms.xlsx", index=False)
    for modality in MODALITIES:
        modality_dir = OUT_DIR / modality
        modality_dir.mkdir(parents=True, exist_ok=True)
        sub = raw_terms[raw_terms["modality"] == modality].copy()
        sub.to_csv(modality_dir / f"{modality}_raw_IDP_GLM_all_terms.csv", index=False)
        for term in TERM_ORDER:
            term_df = sub[sub["term"] == term].copy()
            safe = term.replace(":", "_").replace(" ", "_")
            term_df.to_csv(modality_dir / f"raw_IDP_GLM_ols_term_{safe}.csv", index=False)
    return raw_terms


def load_nm_terms():
    nm_path = NM_OUT / "primary_OLS_all60_all_terms.csv"
    if not nm_path.exists():
        raise FileNotFoundError(
            f"Missing NM primary OLS output: {nm_path}. "
            "Run p4_primary_ols.py first."
        )
    nm = pd.read_csv(nm_path)
    return nm[nm["term"].isin(TERM_ORDER)].copy()


def compare_raw_and_nm(raw_terms, nm_terms):
    raw = raw_terms.rename(
        columns={
            "beta": "raw_beta",
            "se": "raw_se",
            "p": "raw_p",
            "p_fdr_bh": "raw_p_fdr_bh",
            "ci_lower_95": "raw_ci_lower_90",
            "ci_higher_95": "raw_ci_upper_90",
        }
    )
    nm = nm_terms.rename(
        columns={
            "beta": "nm_beta",
            "se": "nm_se",
            "p": "nm_p",
            "p_fdr_bh": "nm_p_fdr_bh",
            "ci_lower_95": "nm_ci_lower_90",
            "ci_higher_95": "nm_ci_upper_90",
        }
    )
    cols_raw = [
        "modality",
        "roi",
        "term",
        "raw_column",
        "n",
        "df_resid",
        "raw_beta",
        "raw_se",
        "raw_p",
        "raw_p_fdr_bh",
        "raw_ci_lower_90",
        "raw_ci_upper_90",
    ]
    cols_nm = [
        "modality",
        "roi",
        "term",
        "nm_beta",
        "nm_se",
        "nm_p",
        "nm_p_fdr_bh",
        "nm_ci_lower_90",
        "nm_ci_upper_90",
    ]
    comp = raw[cols_raw].merge(nm[cols_nm], on=["modality", "roi", "term"], how="inner")
    comp["term_label"] = comp["term"].map(TERM_LABELS)
    comp["beta_delta_nm_minus_raw"] = comp["nm_beta"] - comp["raw_beta"]
    comp["same_direction"] = np.sign(comp["raw_beta"]) == np.sign(comp["nm_beta"])
    comp["abs_raw_beta"] = comp["raw_beta"].abs()
    comp["abs_nm_beta"] = comp["nm_beta"].abs()
    comp = sort_idp_rows(comp, TERM_ORDER)
    comp.to_csv(OUT_DIR / "raw_vs_NM_effect_size_comparison.csv", index=False)
    return comp


def correlation_summary(comp):
    rows = []
    groupings = [("pooled", ["term"]), ("by_modality", ["term", "modality"])]
    for scope, group_cols in groupings:
        for keys, sub in comp.groupby(group_cols):
            if not isinstance(keys, tuple):
                keys = (keys,)
            item = dict(zip(group_cols, keys))
            if scope == "pooled":
                item["modality"] = "All"
            sub = sub.dropna(subset=["raw_beta", "nm_beta"])
            pearson_r, pearson_p = stats.pearsonr(sub["raw_beta"], sub["nm_beta"])
            rows.append(
                {
                    "scope": scope,
                    **item,
                    "n_idp": len(sub),
                    "pearson_r": pearson_r,
                    "pearson_p": pearson_p,
                    "same_direction_pct": 100 * sub["same_direction"].mean(),
                    "n_raw_fdr_p05": int((sub["raw_p_fdr_bh"] < 0.05).sum()),
                    "n_nm_fdr_p05": int((sub["nm_p_fdr_bh"] < 0.05).sum()),
                    "max_abs_raw_beta": sub["abs_raw_beta"].max(),
                    "max_abs_nm_beta": sub["abs_nm_beta"].max(),
                }
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "raw_vs_NM_effect_size_correlation_summary.csv", index=False)
    return summary


def set_equal_limits(ax, values_x, values_y):
    vals = pd.concat([values_x, values_y]).dropna()
    bound = max(0.02, float(np.nanmax(np.abs(vals))) * 1.12)
    ax.set_xlim(-bound, bound)
    ax.set_ylim(-bound, bound)
    ax.plot([-bound, bound], [-bound, bound], color="0.55", linestyle="--", linewidth=0.9, zorder=0)
    ax.axhline(0, color="0.85", linewidth=0.7, zorder=0)
    ax.axvline(0, color="0.85", linewidth=0.7, zorder=0)


def format_p_value(p_value):
    if p_value < 0.001:
        return "<0.001"
    return f"{p_value:.3f}".rstrip("0").rstrip(".")


def add_stats_text(ax, sub):
    r, p = stats.pearsonr(sub["raw_beta"], sub["nm_beta"])
    sign_pct = 100 * sub["same_direction"].mean()
    text = f"r = {r:.2f}, p{format_p_value(p) if p < 0.001 else ' = ' + format_p_value(p)}\nsame sign = {sign_pct:.0f}%"
    ax.text(
        0.04,
        0.96,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.8", "alpha": 0.95},
    )


def savefig(fig, stem):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(FIG_DIR / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)


def plot_pooled(comp):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), dpi=300)
    for ax, term in zip(axes, TERM_ORDER):
        sub = comp[comp["term"] == term].dropna(subset=["raw_beta", "nm_beta"])
        for modality in MODALITIES:
            m = sub[sub["modality"] == modality]
            ax.scatter(
                m["raw_beta"],
                m["nm_beta"],
                s=26,
                color=MODALITY_COLORS[modality],
                edgecolor="white",
                linewidth=0.35,
                alpha=0.86,
                label=modality,
            )
        set_equal_limits(ax, sub["raw_beta"], sub["nm_beta"])
        add_stats_text(ax, sub)
        ax.set_title(TERM_LABELS[term], fontsize=11)
        ax.set_xlabel("Raw IDP adjusted effect size")
        ax.set_ylabel("NM-deviation effect size")
        ax.grid(color="0.92", linewidth=0.6)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color("0.2")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Raw-IDP GLM vs NM-deviation OLS effect sizes", fontsize=13, y=1.02)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    savefig(fig, "raw_vs_NM_effect_size_pooled")


def plot_by_modality(comp):
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.6), dpi=300)
    for row_idx, term in enumerate(TERM_ORDER):
        for col_idx, modality in enumerate(MODALITIES):
            ax = axes[row_idx, col_idx]
            sub = comp[(comp["term"] == term) & (comp["modality"] == modality)].dropna(
                subset=["raw_beta", "nm_beta"]
            )
            ax.scatter(
                sub["raw_beta"],
                sub["nm_beta"],
                s=28,
                color=MODALITY_COLORS[modality],
                edgecolor="white",
                linewidth=0.35,
                alpha=0.9,
            )
            set_equal_limits(ax, sub["raw_beta"], sub["nm_beta"])
            add_stats_text(ax, sub)
            if row_idx == 0:
                ax.set_title(modality, fontsize=11)
            if col_idx == 0:
                ax.set_ylabel(f"{TERM_LABELS[term]}\nNM-deviation effect size", fontsize=10)
            else:
                ax.set_ylabel("")
            if row_idx == 1:
                ax.set_xlabel("Raw IDP adjusted effect size")
            ax.grid(color="0.92", linewidth=0.6)
            for spine in ax.spines.values():
                spine.set_linewidth(0.8)
                spine.set_color("0.2")
    fig.suptitle("Raw-IDP and NM-deviation effect-size consistency by modality", fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    savefig(fig, "raw_vs_NM_effect_size_by_modality")


def plot_abs_effect_shift(comp):
    summary = (
        comp.groupby(["term", "modality"], as_index=False)
        .agg(
            median_abs_raw=("abs_raw_beta", "median"),
            median_abs_nm=("abs_nm_beta", "median"),
            max_abs_raw=("abs_raw_beta", "max"),
            max_abs_nm=("abs_nm_beta", "max"),
        )
    )
    summary.to_csv(OUT_DIR / "raw_vs_NM_abs_effect_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=300, sharey=True)
    x = np.arange(len(MODALITIES))
    width = 0.34
    for ax, term in zip(axes, TERM_ORDER):
        sub = summary[summary["term"] == term].set_index("modality").reindex(MODALITIES)
        ax.bar(x - width / 2, sub["median_abs_raw"], width, label="Raw IDP", color="#9E9E9E")
        ax.bar(x + width / 2, sub["median_abs_nm"], width, label="NM deviation", color="#4C78A8")
        ax.set_xticks(x)
        ax.set_xticklabels(MODALITIES)
        ax.set_title(TERM_LABELS[term], fontsize=11)
        ax.set_ylabel("Median absolute effect size")
        ax.grid(axis="y", color="0.92", linewidth=0.6)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color("0.2")
    axes[0].legend(frameon=False)
    fig.suptitle("Median absolute effect sizes in raw and NM-deviation analyses", fontsize=13, y=1.02)
    fig.tight_layout()
    savefig(fig, "raw_vs_NM_median_abs_effect_size")


def save_sample_summary(data):
    summary = (
        data.groupby("analysis_group", observed=False)
        .agg(
            n=("eid", "count"),
            age_mean=("age", "mean"),
            age_sd=("age", "std"),
            female_n=("sex", lambda x: int((x == "Female").sum())),
            male_n=("sex", lambda x: int((x == "Male").sum())),
            n_sites=("site", "nunique"),
        )
        .reset_index()
    )
    summary.to_csv(OUT_DIR / "raw_analysis_sample_summary.csv", index=False)
    return summary


def main():
    args = parse_args()
    configure_paths(args)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    mapping = load_all_mappings()
    mapping.to_csv(OUT_DIR / "raw_IDP_ROI_field_mapping.csv", index=False)

    data = load_raw_analysis_data(mapping)
    save_sample_summary(data)

    raw_terms = run_raw_models(data, mapping)
    nm_terms = load_nm_terms()
    comp = compare_raw_and_nm(raw_terms, nm_terms)
    summary = correlation_summary(comp)

    plot_pooled(comp)
    plot_by_modality(comp)
    plot_abs_effect_shift(comp)

    print(summary.to_string(index=False))
    print("\nFDR-significant effects:")
    sig = (
        raw_terms[raw_terms["p_fdr_bh"] < 0.05]
        .sort_values(["term", "modality", "p_fdr_bh"])
        [["modality", "roi", "term", "beta", "p", "p_fdr_bh"]]
    )
    print(sig.to_string(index=False) if not sig.empty else "None")


if __name__ == "__main__":
    main()
