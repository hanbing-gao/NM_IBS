"""Rerun the primary IBS-HC OLS analyses with all 60 ROIs per modality.

The historical p6 OLS notebook excluded CT rh_G_cuneus because the upstream
normative-model evaluation filter removed it from the saved deviation-score
tables. This revision uses the all-predicted deviation-score exports generated
for the NeuroImage revision and refits the original model:

    ROI deviation ~ Group * C(sex, Sum)

Example:
    python scripts/p4_primary_ols.py --input-dir data/processed --output-dir results/primary_OLS_all60
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PACKAGE_ROOT / "results" / "primary_OLS_all60"
OLD_OUT: Path | None = None

HC_COV_PATH = PACKAGE_ROOT / "data" / "processed" / "HC_cov_ROME.csv"
IBS_COV_PATH = PACKAGE_ROOT / "data" / "processed" / "IBS_cov_ROME.csv"
HC_DEV_PATH = PACKAGE_ROOT / "data" / "processed" / "HC_deviation_scores_by_brain_idp_all_predicted.csv"
IBS_DEV_PATH = PACKAGE_ROOT / "data" / "processed" / "IBS_deviation_scores_by_brain_idp_all_predicted.csv"

MODALITIES = ["CT", "SA", "CV"]
TERM_ORDER = [
    "Intercept",
    "Group[T.IBS]",
    "C(sex, Sum)[S.Female]",
    "Group[T.IBS]:C(sex, Sum)[S.Female]",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run primary HC vs IBS OLS analyses on all exported deviation-score IDPs."
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
        default=PACKAGE_ROOT / "results" / "primary_OLS_all60",
        help="Directory for primary-analysis outputs.",
    )
    parser.add_argument(
        "--old-output-dir",
        type=Path,
        default=None,
        help="Optional directory with historical OLS outputs for overlap checks.",
    )
    return parser.parse_args()


def configure_paths(args: argparse.Namespace) -> None:
    global OUT_DIR, OLD_OUT, HC_COV_PATH, IBS_COV_PATH, HC_DEV_PATH, IBS_DEV_PATH
    input_dir = args.input_dir.resolve()
    OUT_DIR = args.output_dir.resolve()
    OLD_OUT = args.old_output_dir.resolve() if args.old_output_dir is not None else None
    HC_COV_PATH = input_dir / "HC_cov_ROME.csv"
    IBS_COV_PATH = input_dir / "IBS_cov_ROME.csv"
    HC_DEV_PATH = input_dir / "HC_deviation_scores_by_brain_idp_all_predicted.csv"
    IBS_DEV_PATH = input_dir / "IBS_deviation_scores_by_brain_idp_all_predicted.csv"


def recode_sex(series):
    """Match the original p6 notebook coding: 0 = Female, 1 = Male."""
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
        raise ValueError(f"No rows after merging {cov_path.name} and {dev_path.name}.")
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


def empty_term_row(modality, roi, idp_col, n, term):
    return {
        "modality": modality,
        "roi": roi,
        "idp_column": idp_col,
        "n": n,
        "df_resid": np.nan,
        "term": term,
        "beta": np.nan,
        "se": np.nan,
        "ci_lower": np.nan,
        "ci_upper": np.nan,
        "t": np.nan,
        "p": np.nan,
        "Residual_var": np.nan,
        "ci_lower_95": np.nan,
        "ci_higher_95": np.nan,
    }


def fit_ols_for_modality(data, modality):
    rows = []
    for idp_col in idp_columns_for(data, modality):
        roi = idp_col.split("__", 1)[1]
        sub = data[["eid", "Group", "sex", idp_col]].dropna().copy()
        n = len(sub)

        if n < 5 or sub[idp_col].nunique() <= 1:
            rows.extend(empty_term_row(modality, roi, idp_col, n, term) for term in TERM_ORDER)
            continue

        res = smf.ols(f'Q("{idp_col}") ~ Group*C(sex, Sum)', data=sub).fit()
        conf_int = res.conf_int()
        tcrit_90 = stats.t.ppf(0.95, res.df_resid)

        for term in res.params.index:
            rows.append(
                {
                    "modality": modality,
                    "roi": roi,
                    "idp_column": idp_col,
                    "n": n,
                    "df_resid": float(res.df_resid),
                    "term": term,
                    "beta": float(res.params[term]),
                    "se": float(res.bse[term]),
                    "ci_lower": float(conf_int.loc[term, 0]),
                    "ci_upper": float(conf_int.loc[term, 1]),
                    "t": float(res.tvalues[term]),
                    "p": float(res.pvalues[term]),
                    "Residual_var": float(res.scale),
                    "ci_lower_95": float(res.params[term] - tcrit_90 * res.bse[term]),
                    "ci_higher_95": float(res.params[term] + tcrit_90 * res.bse[term]),
                }
            )

    return pd.DataFrame(rows)


def term_filename(term):
    safe_term = term.replace(":", "_").replace(" ", "_")
    return f"ols_term_{safe_term}_effectcoding.csv"


def add_fdr(term_df):
    term_df = term_df.copy()
    term_df["p_fdr_bh"] = np.nan
    mask = term_df["p"].notna()
    if mask.any():
        _, pvals_fdr, _, _ = multipletests(term_df.loc[mask, "p"], alpha=0.05, method="fdr_bh")
        term_df.loc[mask, "p_fdr_bh"] = pvals_fdr
    return term_df


def save_modality_outputs(modality_df, modality):
    modality_dir = OUT_DIR / modality
    modality_dir.mkdir(parents=True, exist_ok=True)

    modality_df.to_csv(modality_dir / f"{modality}_OLS_all_terms_all60.csv", index=False)
    saved_terms = []
    for term in TERM_ORDER:
        term_df = modality_df[modality_df["term"] == term].copy()
        term_df = add_fdr(term_df)
        term_df.to_csv(modality_dir / term_filename(term), index=False)
        saved_terms.append(term_df)
    return pd.concat(saved_terms, ignore_index=True)


def summarize(all_terms):
    summary = (
        all_terms.groupby(["modality", "term"], as_index=False)
        .agg(
            n_idp=("roi", "count"),
            n_nominal_p05=("p", lambda x: int((x < 0.05).sum())),
            n_fdr_p05=("p_fdr_bh", lambda x: int((x < 0.05).sum())),
            min_p=("p", "min"),
            min_p_fdr=("p_fdr_bh", "min"),
            max_abs_beta=("beta", lambda x: float(np.nanmax(np.abs(x)))),
        )
    )
    summary.to_csv(OUT_DIR / "primary_OLS_all60_summary.csv", index=False)
    return summary


def compare_with_old(all_terms):
    if OLD_OUT is None:
        comparison = pd.DataFrame()
        comparison.to_csv(OUT_DIR / "primary_OLS_all60_old_overlap_check.csv", index=False)
        return comparison

    rows = []
    old_file_by_term = {
        "Group[T.IBS]": "ols_term_Group[T.IBS]_effectcoding.csv",
        "C(sex, Sum)[S.Female]": "ols_term_C(sex,_Sum)[S.Female]_effectcoding.csv",
        "Group[T.IBS]:C(sex, Sum)[S.Female]": (
            "ols_term_Group[T.IBS]_C(sex,_Sum)[S.Female]_effectcoding.csv"
        ),
    }
    for modality in MODALITIES:
        for term, old_file in old_file_by_term.items():
            old_path = OLD_OUT / modality / old_file
            if not old_path.exists():
                continue
            old = pd.read_csv(old_path)
            new = all_terms[(all_terms["modality"] == modality) & (all_terms["term"] == term)]
            merged = old[["roi", "beta", "se", "df_resid"]].merge(
                new[["roi", "beta", "se", "df_resid"]],
                on="roi",
                suffixes=("_old", "_new"),
            )
            rows.append(
                {
                    "modality": modality,
                    "term": term,
                    "old_n": len(old),
                    "new_n": len(new),
                    "overlap_n": len(merged),
                    "max_abs_beta_diff": float((merged["beta_old"] - merged["beta_new"]).abs().max()),
                    "max_abs_se_diff": float((merged["se_old"] - merged["se_new"]).abs().max()),
                    "df_match": bool((merged["df_resid_old"] == merged["df_resid_new"]).all()),
                    "new_only_rois": ";".join(sorted(set(new["roi"]) - set(old["roi"]))),
                }
            )

    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUT_DIR / "primary_OLS_all60_old_overlap_check.csv", index=False)
    return comparison


def main():
    args = parse_args()
    configure_paths(args)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_model_data()

    saved = []
    for modality in MODALITIES:
        modality_df = fit_ols_for_modality(data, modality)
        saved.append(save_modality_outputs(modality_df, modality))

    all_terms = pd.concat(saved, ignore_index=True)
    all_terms.to_csv(OUT_DIR / "primary_OLS_all60_all_terms.csv", index=False)
    summary = summarize(all_terms)
    comparison = compare_with_old(all_terms)

    print(summary.to_string(index=False))
    if not comparison.empty:
        print("\nOld/new overlap check:")
        print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
