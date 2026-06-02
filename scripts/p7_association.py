"""
Rerun brain-phenotype association analyses with all 60 ROIs per modality.

This revision uses the all-predicted normative-model deviation score exports,
so cortical thickness no longer loses rh_G_cuneus because of the old model
prediction-quality p-value filter.

Example:
    python scripts/p7_association.py --input-dir data/processed --output-dir results/association_all60
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm, shapiro
from statsmodels.formula.api import ols
from statsmodels.multivariate.manova import MANOVA
from statsmodels.stats.multitest import multipletests

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PACKAGE_ROOT
DOCU_DIR = PACKAGE_ROOT / "docs"
INPUT_DIR = PACKAGE_ROOT / "data" / "processed"
OUT_DIR = PACKAGE_ROOT / "results" / "association_all60"
OLD_CORR_DIR: Path | None = None
OLD_SEX_DIR: Path | None = None
MAKE_BRAIN_MAP = False

ROI_LIST = pd.read_csv(DOCU_DIR / "ROI_IBS.csv")["ROI"].tolist()
MODALITIES = ["CT", "SA", "CV"]
MODALITY_LABELS = {
    "CT": "Cortical Thickness",
    "SA": "Surface Area",
    "CV": "Cortical Volume",
}
SCALES = ["IBS-SSS", "PHQ-12", "PHQ-9", "GAD-7"]
S3_SCALES = ["IBS-SSS", "PHQ-12", "PHQ-9", "GAD-7"]
SAFE_SCALE = {
    "IBS-SSS": "IBS_SSS",
    "PHQ-12": "PHQ_12",
    "PHQ-9": "PHQ_9",
    "GAD-7": "GAD_7",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all-60 ROI questionnaire association analyses using NM deviation scores."
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
    parser.add_argument("--docs-dir", type=Path, default=PACKAGE_ROOT / "docs")
    parser.add_argument("--output-dir", type=Path, default=PACKAGE_ROOT / "results" / "association_all60")
    parser.add_argument(
        "--make-brain-map",
        action="store_true",
        help="Also create the S2-style brain map. Requires optional visualization dependencies and annotation files.",
    )
    parser.add_argument("--old-correlation-dir", type=Path, default=None)
    parser.add_argument("--old-sex-correlation-dir", type=Path, default=None)
    return parser.parse_args()


def configure_paths(args: argparse.Namespace) -> None:
    global DOCU_DIR, INPUT_DIR, OUT_DIR, OLD_CORR_DIR, OLD_SEX_DIR, MAKE_BRAIN_MAP, ROI_LIST
    DOCU_DIR = args.docs_dir.resolve()
    INPUT_DIR = args.input_dir.resolve()
    OUT_DIR = args.output_dir.resolve()
    OLD_CORR_DIR = args.old_correlation_dir.resolve() if args.old_correlation_dir is not None else None
    OLD_SEX_DIR = args.old_sex_correlation_dir.resolve() if args.old_sex_correlation_dir is not None else None
    MAKE_BRAIN_MAP = args.make_brain_map
    ROI_LIST = pd.read_csv(DOCU_DIR / "ROI_IBS.csv")["ROI"].tolist()


def ensure_dirs() -> None:
    for sub in ["correlation_ROME", "correlation_ROME_sex", "ANCOVA_correlation_ROME", "tables", "figures"]:
        (OUT_DIR / sub).mkdir(parents=True, exist_ok=True)


def fdr(values: pd.Series | np.ndarray) -> np.ndarray:
    pvals = np.asarray(values, dtype=float)
    out = np.full(pvals.shape, np.nan, dtype=float)
    mask = np.isfinite(pvals)
    if mask.sum():
        out[mask] = multipletests(pvals[mask], method="fdr_bh")[1]
    return out


def fisher_z(r: float) -> float:
    if not np.isfinite(r):
        return np.nan
    clipped = np.clip(r, -0.999999999999, 0.999999999999)
    return 0.5 * np.log((1 + clipped) / (1 - clipped))


def r_to_d(r: float | pd.Series) -> float | pd.Series:
    return 2 * r / np.sqrt(1 - np.asarray(r) ** 2)


def calculate_correlation_and_z_scores(
    merged_data: pd.DataFrame, roi_list: list[str], demo_columns: list[str]
) -> dict[str, pd.DataFrame]:
    """Replicate the old p4 correlation helper."""
    results_by_demo: dict[str, pd.DataFrame] = {}
    merged_data = merged_data.copy()
    merged_data.dropna(how="all", axis=1, inplace=True)
    merged_data.dropna(inplace=True)

    for demo_col in demo_columns:
        roi_results = {}
        for roi in roi_list:
            if roi not in merged_data.columns:
                continue

            _, p_val_demo = shapiro(merged_data[demo_col])
            _, p_val_roi = shapiro(merged_data[roi])
            if p_val_demo > 0.05 and p_val_roi > 0.05:
                corr = merged_data[demo_col].corr(merged_data[roi], method="pearson")
                corr_method = "pearson"
            else:
                corr = merged_data[demo_col].corr(merged_data[roi], method="spearman")
                corr_method = "spearman"

            n = len(merged_data)
            z = fisher_z(corr)
            if abs(corr) < 1 and n > 2:
                p_val = 2 * (1 - norm.cdf(abs(corr * math.sqrt((n - 2) / (1 - corr**2)))))
            else:
                p_val = np.nan

            roi_results[roi] = {
                "corr_method": corr_method,
                "r": corr,
                "z_score": z,
                "p_value": p_val,
                "sample_size": n,
            }

        results_by_demo[demo_col] = pd.DataFrame.from_dict(roi_results, orient="index")

    return results_by_demo


def equivalence_test_for_correlation(results: pd.DataFrame, sesoi: float, alpha: float = 0.05):
    """Equivalence test used by the old p4 sex-stratified correlation helper."""

    def d_to_r(d: float) -> float:
        return d / np.sqrt(d**2 + 4)

    sesoi_r = d_to_r(sesoi) if abs(sesoi) < 1 else sesoi
    z_low_bound = np.arctanh(-sesoi_r)
    z_high_bound = np.arctanh(sesoi_r)
    se = 1 / np.sqrt(results["sample_size"] - 3)

    z_low = (results["z_score"] - z_low_bound) / se
    z_high = (results["z_score"] - z_high_bound) / se
    p_low = 1 - norm.cdf(abs(z_low))
    p_high = 1 - norm.cdf(abs(z_high))
    p_equivalence = np.maximum(p_low, p_high)

    z_ci = norm.ppf(1 - alpha) * se
    ci_low90_r = np.tanh(results["z_score"] - z_ci)
    ci_high90_r = np.tanh(results["z_score"] + z_ci)
    return p_equivalence, r_to_d(ci_low90_r), r_to_d(ci_high90_r)


def calculate_and_compare_correlations(
    results1: dict[str, pd.DataFrame],
    results2: dict[str, pd.DataFrame],
    demo_col: str,
    label1: str,
    label2: str,
    sesoi_1: float,
    sesoi_2: float,
) -> pd.DataFrame:
    data1 = results1[demo_col]
    data2 = results2[demo_col]
    common = data1.index.intersection(data2.index)
    data1 = data1.loc[common]
    data2 = data2.loc[common]

    z_diff = data1["z_score"] - data2["z_score"]
    z_diff_se = np.sqrt(1 / (data1["sample_size"] - 3) + 1 / (data2["sample_size"] - 3))
    z_test = z_diff / z_diff_se
    diff_p_values = 2 * (1 - norm.cdf(abs(z_test)))
    ci_low = z_diff - 1.96 * z_diff_se
    ci_high = z_diff + 1.96 * z_diff_se

    p_eq_1, ci_low90_1, ci_high90_1 = equivalence_test_for_correlation(data1, sesoi_1, 0.05)
    p_eq_2, ci_low90_2, ci_high90_2 = equivalence_test_for_correlation(data2, sesoi_2, 0.05)

    results = pd.DataFrame(index=common)
    for label, data, p_eq, ci_lo, ci_hi in [
        (label1, data1, p_eq_1, ci_low90_1, ci_high90_1),
        (label2, data2, p_eq_2, ci_low90_2, ci_high90_2),
    ]:
        results[f"{label}_n"] = data["sample_size"]
        results[f"{label}_method"] = data["corr_method"]
        results[f"{label}_r"] = data["r"]
        results[f"{label}_z"] = data["z_score"]
        results[f"{label}_p"] = data["p_value"]
        results[f"{label}_p_fdr"] = fdr(data["p_value"])
        results[f"{label}_cohens_d"] = r_to_d(data["r"])
        results[f"{label}_p_equivalence"] = p_eq
        results[f"{label}_p_equivalence_fdr"] = fdr(p_eq)
        results[f"{label}_ci_low90"] = ci_lo
        results[f"{label}_ci_high90"] = ci_hi

        p_eq_correct, ci_low99, ci_high99 = equivalence_test_for_correlation(data, 0.2, 0.00017)
        results[f"{label}_ci_low90_corrected"] = ci_low99
        results[f"{label}_ci_high90_corrected"] = ci_high99

    results["z_diff"] = z_diff
    results["ci_low"] = ci_low
    results["ci_high"] = ci_high
    results["diff_p"] = diff_p_values
    results["diff_p_fdr"] = fdr(diff_p_values)
    return results


def correlation_difference_analysis(
    corr_data: dict[str, dict[str, pd.DataFrame]],
    demo_str: str,
    sesoi: float = 0.112,
    sesoi_hc: float = 0.176,
    sesoi_ibs: float = 0.139,
) -> pd.DataFrame:
    hc = corr_data["HC"][demo_str]
    ibs = corr_data["IBS"][demo_str]
    result_rows = []

    for roi in hc.index.intersection(ibs.index):
        rhc, nhc, zhc, phc = hc.loc[roi, ["r", "sample_size", "z_score", "p_value"]]
        rib, nib, zib, pib = ibs.loc[roi, ["r", "sample_size", "z_score", "p_value"]]
        z_hc = np.arctanh(np.clip(rhc, -0.999999999999, 0.999999999999))
        z_ibs = np.arctanh(np.clip(rib, -0.999999999999, 0.999999999999))

        cohens_d_hc = r_to_d(rhc) if abs(rhc) < 1 else np.nan
        cohens_d_ibs = r_to_d(rib) if abs(rib) < 1 else np.nan
        se_diff = np.sqrt(1 / (nhc - 3) + 1 / (nib - 3)) if (nhc > 3 and nib > 3) else np.nan
        diff_z = z_ibs - z_hc if se_diff > 0 else np.nan
        z_stat = diff_z / se_diff if se_diff > 0 else np.nan
        p_diff = 2 * (1 - norm.cdf(abs(z_stat))) if se_diff > 0 else np.nan

        ci90_z = norm.ppf(1 - 0.05) * se_diff if se_diff > 0 else np.nan
        ci_low90_z = diff_z - ci90_z if se_diff > 0 else np.nan
        ci_high90_z = diff_z + ci90_z if se_diff > 0 else np.nan

        p_eq_diff = np.nan
        if se_diff > 0:
            p_low_eq = 1 - norm.cdf(abs((diff_z + sesoi) / se_diff))
            p_high_eq = 1 - norm.cdf(abs((diff_z - sesoi) / se_diff))
            p_eq_diff = max(p_low_eq, p_high_eq)

        def d_equivalence(corr: float, n: int, d_value: float, bound: float):
            if n <= 3 or abs(corr) >= 1 or not np.isfinite(corr):
                return np.nan, np.nan, np.nan
            se_r = np.sqrt((1 - corr**2) ** 2 / (n - 1))
            se_d = 2 * (1 + corr**2 / 2) / (1 - corr**2) ** 1.5 * se_r
            ci90_d = norm.ppf(1 - 0.05) * se_d
            p_low = 1 - norm.cdf(abs((d_value + bound) / se_d))
            p_high = 1 - norm.cdf(abs((d_value - bound) / se_d))
            return d_value - ci90_d, d_value + ci90_d, max(p_low, p_high)

        hc_ci_low, hc_ci_high, hc_p_eq = d_equivalence(rhc, nhc, cohens_d_hc, sesoi_hc)
        ibs_ci_low, ibs_ci_high, ibs_p_eq = d_equivalence(rib, nib, cohens_d_ibs, sesoi_ibs)

        result_rows.append(
            {
                "ROI": roi,
                "demo": demo_str,
                "HC_n": nhc,
                "HC_r": rhc,
                "HC_cohens_d": cohens_d_hc,
                "HC_p": phc,
                "HC_ci90_d_low": hc_ci_low,
                "HC_ci90_d_high": hc_ci_high,
                "HC_p_equivalence": hc_p_eq,
                "IBS_n": nib,
                "IBS_r": rib,
                "IBS_cohens_d": cohens_d_ibs,
                "IBS_p": pib,
                "IBS_ci90_d_low": ibs_ci_low,
                "IBS_ci90_d_high": ibs_ci_high,
                "IBS_p_equivalence": ibs_p_eq,
                "Difference_z": diff_z,
                "Diff_ci90_z_low": ci_low90_z,
                "Diff_ci90_z_high": ci_high90_z,
                "p_Diff": p_diff,
                "p_equivalence_Diff": p_eq_diff,
            }
        )

    result_rows = pd.DataFrame(result_rows)
    for p_col in [
        "HC_p",
        "IBS_p",
        "p_Diff",
        "p_equivalence_Diff",
        "HC_p_equivalence",
        "IBS_p_equivalence",
    ]:
        result_rows[p_col + "_fdr"] = fdr(result_rows[p_col])
    return result_rows


def did_interaction_analysis(corr_data: dict[str, dict[str, pd.DataFrame]], demo_str: str, sesoi: float = 0.256):
    ibs_f = corr_data["IBS Female"][demo_str]
    ibs_m = corr_data["IBS Male"][demo_str]
    hc_f = corr_data["HC Female"][demo_str]
    hc_m = corr_data["HC Male"][demo_str]

    common = ibs_f.index.intersection(ibs_m.index).intersection(hc_f.index).intersection(hc_m.index)
    rows = []
    for roi in common:
        rf, nf, zf = ibs_f.loc[roi, ["r", "sample_size", "z_score"]]
        rm, nm, zm = ibs_m.loc[roi, ["r", "sample_size", "z_score"]]
        rhf, nhf, zhf = hc_f.loc[roi, ["r", "sample_size", "z_score"]]
        rhm, nhm, zhm = hc_m.loc[roi, ["r", "sample_size", "z_score"]]

        z_ibs_diff = np.arctanh(np.clip(rf, -0.999999999999, 0.999999999999)) - np.arctanh(
            np.clip(rm, -0.999999999999, 0.999999999999)
        )
        se_ibs = np.sqrt(1 / (nf - 3) + 1 / (nm - 3))
        z_hc_diff = np.arctanh(np.clip(rhf, -0.999999999999, 0.999999999999)) - np.arctanh(
            np.clip(rhm, -0.999999999999, 0.999999999999)
        )
        se_hc = np.sqrt(1 / (nhf - 3) + 1 / (nhm - 3))
        did = z_ibs_diff - z_hc_diff
        se_did = np.sqrt(se_ibs**2 + se_hc**2)
        z_stat = did / se_did if se_did > 0 else np.nan
        p_did = 2 * (1 - norm.cdf(abs(z_stat))) if se_did > 0 else np.nan

        ci90_z = norm.ppf(1 - 0.05) * se_did if se_did > 0 else np.nan
        ci_low90_z = did - ci90_z if se_did > 0 else np.nan
        ci_high90_z = did + ci90_z if se_did > 0 else np.nan
        p_eq = np.nan
        if se_did > 0:
            p_low_eq = 1 - norm.cdf(abs((did + sesoi) / se_did))
            p_high_eq = 1 - norm.cdf(abs((did - sesoi) / se_did))
            p_eq = max(p_low_eq, p_high_eq)

        rows.append(
            {
                "ROI": roi,
                "demo": demo_str,
                "IBS_F_r": rf,
                "IBS_M_r": rm,
                "HC_F_r": rhf,
                "HC_M_r": rhm,
                "IBS_F_n": nf,
                "IBS_M_n": nm,
                "HC_F_n": nhf,
                "HC_M_n": nhm,
                "IBS_F_z": zf,
                "IBS_M_z": zm,
                "HC_F_z": zhf,
                "HC_M_z": zhm,
                "IBS_F-M_diff(Cohens_q)": z_ibs_diff,
                "HC_F-M_diff(Cohens_q)": z_hc_diff,
                "DiD_value(Cohens_q)": did,
                "DiD_r": np.tanh(did),
                "SE_DiD": se_did,
                "z_DiD": z_stat,
                "p_DiD": p_did,
                "p_equivalence_DiD": p_eq,
                "DiD_ci90_z_low": ci_low90_z,
                "DiD_ci90_z_high": ci_high90_z,
                "DiD_ci90_r_low": np.tanh(ci_low90_z) if np.isfinite(ci_low90_z) else np.nan,
                "DiD_ci90_r_high": np.tanh(ci_high90_z) if np.isfinite(ci_high90_z) else np.nan,
            }
        )

    out = pd.DataFrame(rows)
    out["p_equivalence_DiD_fdr"] = fdr(out["p_equivalence_DiD"])
    out["p_DiD_fdr"] = fdr(out["p_DiD"])
    return out


def load_covariates() -> tuple[pd.DataFrame, pd.DataFrame]:
    hc = pd.read_csv(INPUT_DIR / "HC_cov_ROME.csv")
    ibs = pd.read_csv(INPUT_DIR / "IBS_cov_ROME.csv")
    hc["Group"] = "HC"
    ibs["Group"] = "IBS"
    return hc, ibs


def load_modality_deviation(modality: str, group: str) -> pd.DataFrame:
    filename = {
        "HC": "HC_deviation_scores_by_brain_idp_all_predicted.csv",
        "IBS": "IBS_deviation_scores_by_brain_idp_all_predicted.csv",
    }[group]
    df = pd.read_csv(INPUT_DIR / filename)
    keep = ["eid"] + [f"{modality}__{roi}" for roi in ROI_LIST]
    missing = [col for col in keep if col not in df.columns]
    if missing:
        raise ValueError(f"Missing {modality} columns in {filename}: {missing}")
    df = df[keep].copy()
    df.rename(columns={f"{modality}__{roi}": roi for roi in ROI_LIST}, inplace=True)
    return df


def prepare_modality_data(modality: str, hc_cov: pd.DataFrame, ibs_cov: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    hc_dev = load_modality_deviation(modality, "HC")
    ibs_dev = load_modality_deviation(modality, "IBS")
    hc_all = pd.merge(hc_cov[["eid", "sex", "Group"] + SCALES], hc_dev, on="eid", how="inner")
    ibs_all = pd.merge(ibs_cov[["eid", "sex", "Group"] + SCALES], ibs_dev, on="eid", how="inner")
    return hc_all, ibs_all


def rerun_correlation_outputs(hc_cov: pd.DataFrame, ibs_cov: pd.DataFrame) -> None:
    corr_base = OUT_DIR / "correlation_ROME"
    sex_base = OUT_DIR / "correlation_ROME_sex"

    for modality in MODALITIES:
        print(f"Correlation analyses: {modality}")
        hc_all, ibs_all = prepare_modality_data(modality, hc_cov, ibs_cov)

        corr_data = {
            "HC": calculate_correlation_and_z_scores(hc_all[["eid", "sex"] + SCALES + ROI_LIST], ROI_LIST, SCALES),
            "IBS": calculate_correlation_and_z_scores(ibs_all[["eid", "sex"] + SCALES + ROI_LIST], ROI_LIST, SCALES),
        }

        sex_corr_data = {
            "HC Female": calculate_correlation_and_z_scores(
                hc_all.loc[hc_all["sex"] == 0, ["eid", "sex"] + SCALES + ROI_LIST], ROI_LIST, SCALES
            ),
            "HC Male": calculate_correlation_and_z_scores(
                hc_all.loc[hc_all["sex"] == 1, ["eid", "sex"] + SCALES + ROI_LIST], ROI_LIST, SCALES
            ),
            "IBS Female": calculate_correlation_and_z_scores(
                ibs_all.loc[ibs_all["sex"] == 0, ["eid", "sex"] + SCALES + ROI_LIST], ROI_LIST, SCALES
            ),
            "IBS Male": calculate_correlation_and_z_scores(
                ibs_all.loc[ibs_all["sex"] == 1, ["eid", "sex"] + SCALES + ROI_LIST], ROI_LIST, SCALES
            ),
        }

        for scale in SCALES:
            scale_dir = corr_base / scale
            scale_dir.mkdir(parents=True, exist_ok=True)
            results = correlation_difference_analysis(corr_data, scale)
            results.to_csv(scale_dir / f"correlation_results_IBS_{modality}.csv", index=False)

            sex_scale_dir = sex_base / scale
            sex_scale_dir.mkdir(parents=True, exist_ok=True)
            did = did_interaction_analysis(sex_corr_data, scale)
            did.to_csv(sex_scale_dir / f"correlation_results_HC_IBS_{modality}_sex_interaction.csv", index=False)

            calculate_and_compare_correlations(
                sex_corr_data["HC Female"],
                sex_corr_data["HC Male"],
                scale,
                "HC Female",
                "HC Male",
                sesoi_1=0.203,
                sesoi_2=0.352,
            ).to_csv(sex_scale_dir / f"correlation_results_HC_{modality}_sex.csv")

            calculate_and_compare_correlations(
                sex_corr_data["IBS Female"],
                sex_corr_data["IBS Male"],
                scale,
                "IBS Female",
                "IBS Male",
                sesoi_1=0.162,
                sesoi_2=0.272,
            ).to_csv(sex_scale_dir / f"correlation_results_IBS_{modality}_sex.csv")

            calculate_and_compare_correlations(
                sex_corr_data["IBS Female"],
                sex_corr_data["HC Female"],
                scale,
                "IBS Female",
                "HC Female",
                sesoi_1=0.162,
                sesoi_2=0.203,
            ).to_csv(sex_scale_dir / f"correlation_results_HC_IBS_{modality}_Female.csv")

            calculate_and_compare_correlations(
                sex_corr_data["IBS Male"],
                sex_corr_data["HC Male"],
                scale,
                "IBS Male",
                "HC Male",
                sesoi_1=0.272,
                sesoi_2=0.352,
            ).to_csv(sex_scale_dir / f"correlation_results_HC_IBS_{modality}_Male.csv")


def s11_rows() -> list[list[object]]:
    rows = []
    corr_base = OUT_DIR / "correlation_ROME"
    for modality in MODALITIES:
        first = True
        for roi in ROI_LIST:
            row: list[object] = [MODALITY_LABELS[modality] if first else "", roi]
            first = False
            scale_tables = {
                scale: pd.read_csv(corr_base / scale / f"correlation_results_IBS_{modality}.csv").set_index("ROI")
                for scale in SCALES
            }
            first_table = scale_tables[SCALES[0]].loc[roi]
            row.extend([first_table["HC_n"], first_table["IBS_n"]])
            for scale in SCALES:
                r = scale_tables[scale].loc[roi]
                row.extend(
                    [
                        r["HC_r"],
                        r["HC_cohens_d"],
                        r["HC_p"],
                        r["HC_p_fdr"],
                        r["HC_ci90_d_low"],
                        r["HC_ci90_d_high"],
                        r["HC_p_equivalence"],
                        r["HC_p_equivalence_fdr"],
                        r["IBS_r"],
                        r["IBS_cohens_d"],
                        r["IBS_p"],
                        r["IBS_p_fdr"],
                        r["IBS_ci90_d_low"],
                        r["IBS_ci90_d_high"],
                        r["IBS_p_equivalence"],
                        r["IBS_p_equivalence_fdr"],
                        r["Difference_z"],
                        r["p_Diff"],
                        r["p_Diff_fdr"],
                        r["Diff_ci90_z_low"],
                        r["Diff_ci90_z_high"],
                        r["p_equivalence_Diff"],
                        r["p_equivalence_Diff_fdr"],
                    ]
                )
            rows.append(row)
    return rows


def s12_rows() -> list[list[object]]:
    rows = []
    sex_base = OUT_DIR / "correlation_ROME_sex"
    for modality in MODALITIES:
        first = True
        for roi in ROI_LIST:
            row: list[object] = [MODALITY_LABELS[modality] if first else "", roi]
            first = False
            scale_tables = {
                scale: pd.read_csv(
                    sex_base / scale / f"correlation_results_HC_IBS_{modality}_sex_interaction.csv"
                ).set_index("ROI")
                for scale in SCALES
            }
            first_table = scale_tables[SCALES[0]].loc[roi]
            row.extend([first_table["IBS_F_n"], first_table["IBS_M_n"], first_table["HC_F_n"], first_table["HC_M_n"]])
            for scale in SCALES:
                r = scale_tables[scale].loc[roi]
                row.extend(
                    [
                        r["IBS_F_r"],
                        r["IBS_M_r"],
                        r["HC_F_r"],
                        r["HC_M_r"],
                        r["DiD_value(Cohens_q)"],
                        r["p_DiD"],
                        r["p_DiD_fdr"],
                        r["DiD_ci90_z_low"],
                        r["DiD_ci90_z_high"],
                        r["p_equivalence_DiD"],
                        r["p_equivalence_DiD_fdr"],
                    ]
                )
            rows.append(row)
    return rows


def write_s11_table() -> Path:
    out_path = OUT_DIR / "tables" / "S11_Table_all60.xlsx"
    header0 = ["S11 Table. Within-group cortical brain-phenotype association and association differences"]
    header1 = ["Brain Measure", "ROI", "N_HC", "N_IBS"]
    header2 = ["", "", "", ""]
    header3 = ["", "", "", ""]
    for scale in SCALES:
        header1.extend([scale] + [""] * 22)
        header2.extend(["HC"] + [""] * 7 + ["IBS"] + [""] * 7 + ["Group Difference"] + [""] * 2 + ["Equivalence of group difference"] + [""] * 3)
        header3.extend(
            [
                "r",
                "Cohen's d",
                "p",
                "p_fdr",
                "ci90_low",
                "ci90_high",
                "p_equivalence",
                "p_equivalence_fdr",
                "r",
                "Cohen's d",
                "p",
                "p_fdr",
                "ci90_low",
                "ci90_high",
                "p_equivalence",
                "p_equivalence_fdr",
                "Cohen's q",
                "p",
                "p_fdr",
                "ci90_low",
                "ci90_high",
                "p_equivalence",
                "p_equivalence_fdr",
            ]
        )
    max_cols = len(header1)
    header0.extend([""] * (max_cols - 1))
    rows = [header0, header1, header2, header3] + s11_rows()
    write_excel_with_headers(out_path, rows, title_merge=(0, 0, 0, max_cols - 1), freeze_row=4)
    return out_path


def write_s12_table() -> Path:
    out_path = OUT_DIR / "tables" / "S12_Table_all60.xlsx"
    header0 = ["S12 Table. Difference-in-differences group x sex x scale interaction effects with equivalence test"]
    header1 = ["Brain Measure", "ROI", "IBS_F_n", "IBS_M_n", "HC_F_n", "HC_M_n"]
    header2 = ["", "", "", "", "", ""]
    for scale in SCALES:
        header1.extend([scale] + [""] * 10)
        header2.extend(
            [
                "IBS_F_r",
                "IBS_M_r",
                "HC_F_r",
                "HC_M_r",
                "DiD Cohen's q",
                "p_DiD",
                "p_DiD_fdr",
                "DiD_ci90_low",
                "DiD_ci90_high",
                "p_equivalence_DiD",
                "p_equivalence_DiD_fdr",
            ]
        )
    max_cols = len(header1)
    header0.extend([""] * (max_cols - 1))
    rows = [header0, header1, header2] + s12_rows()
    write_excel_with_headers(out_path, rows, title_merge=(0, 0, 0, max_cols - 1), freeze_row=3)
    return out_path


def write_excel_with_headers(
    out_path: Path, rows: list[list[object]], title_merge: tuple[int, int, int, int], freeze_row: int
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        sheet_name = out_path.stem.replace("_all60", "").replace("_", " ")
        workbook = writer.book
        worksheet = workbook.add_worksheet(sheet_name[:31])
        writer.sheets[sheet_name[:31]] = worksheet

        title_fmt = workbook.add_format({"bold": True, "font_size": 12, "align": "left", "valign": "vcenter"})
        header_fmt = workbook.add_format(
            {"bold": True, "align": "center", "valign": "vcenter", "border": 1, "text_wrap": True}
        )
        text_fmt = workbook.add_format({"align": "left", "valign": "vcenter", "border": 1})
        int_fmt = workbook.add_format({"align": "center", "valign": "vcenter", "border": 1, "num_format": "0"})
        num_fmt = workbook.add_format({"align": "center", "valign": "vcenter", "border": 1, "num_format": "0.000000"})

        r1, c1, r2, c2 = title_merge
        worksheet.merge_range(r1, c1, r2, c2, rows[0][0], title_fmt)
        for r, row in enumerate(rows[1:], start=1):
            for c, val in enumerate(row):
                fmt = header_fmt if r < freeze_row else num_fmt
                if r >= freeze_row and c in [0, 1]:
                    fmt = text_fmt
                if r >= freeze_row and isinstance(val, (int, np.integer)):
                    fmt = int_fmt
                if pd.isna(val):
                    val = ""
                worksheet.write(r, c, val, fmt)

        worksheet.freeze_panes(freeze_row, 2)
        worksheet.set_column(0, 0, 22)
        worksheet.set_column(1, 1, 28)
        worksheet.set_column(2, len(rows[1]) - 1, 13)

        # Merge scale and section labels.
        if "S11" in out_path.name:
            block_start = 4
            for scale_i in range(len(SCALES)):
                start = block_start + 23 * scale_i
                worksheet.merge_range(1, start, 1, start + 22, SCALES[scale_i], header_fmt)
                worksheet.merge_range(2, start, 2, start + 7, "HC", header_fmt)
                worksheet.merge_range(2, start + 8, 2, start + 15, "IBS", header_fmt)
                worksheet.merge_range(2, start + 16, 2, start + 18, "Group Difference", header_fmt)
                worksheet.merge_range(2, start + 19, 2, start + 22, "Equivalence of group difference", header_fmt)
        else:
            block_start = 6
            for scale_i in range(len(SCALES)):
                start = block_start + 11 * scale_i
                worksheet.merge_range(1, start, 1, start + 10, SCALES[scale_i], header_fmt)


def forest_panel(
    ax,
    df: pd.DataFrame,
    effect_col: str,
    low_col: str,
    high_col: str,
    p_eq_fdr_col: str,
    xlim: tuple[float, float],
    margin: float,
    show_y: bool,
    xlabel: bool = True,
):
    df = df.set_index("ROI").reindex(ROI_LIST).reset_index()
    for idx, row in df.iterrows():
        if not np.isfinite(row[effect_col]) or not np.isfinite(row[low_col]) or not np.isfinite(row[high_col]):
            continue
        color = "red" if row[p_eq_fdr_col] > 0.05 else "0.25"
        ax.plot([row[low_col], row[high_col]], [idx, idx], color=color, linewidth=1.0)
        ax.plot(row[effect_col], idx, marker="D", markersize=3.2, markerfacecolor="white", markeredgecolor=color, markeredgewidth=0.8)
    ax.axvline(-margin, color="0.35", linestyle=":", linewidth=0.8)
    ax.axvline(margin, color="0.35", linestyle=":", linewidth=0.8)
    ax.axvline(0, color="0.6", linestyle="-", linewidth=0.6)
    ax.set_xlim(*xlim)
    ax.set_ylim(len(ROI_LIST) - 0.5, -0.5)
    ax.set_yticks(range(len(ROI_LIST)))
    if show_y:
        ax.set_yticklabels(ROI_LIST, fontsize=5)
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=6)
    if xlabel:
        ax.set_xlabel("Effect sizes with 90% CIs", fontsize=6, labelpad=2)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.7)


def add_modality_label(ax, modality: str, y: float = -0.16) -> None:
    ax.text(
        0.5,
        y,
        modality,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7,
        fontweight="bold",
        clip_on=False,
    )


def save_figure_all_formats(fig: plt.Figure, stem: str) -> list[str]:
    paths = []
    for ext in ["png", "pdf", "tif"]:
        path = OUT_DIR / "figures" / f"{stem}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        paths.append(str(path))
    return paths


def make_s3_forest_plot_for_scale(scale: str, stem: str) -> None:
    corr_base = OUT_DIR / "correlation_ROME"
    fig = plt.figure(figsize=(8, 8.5), dpi=300)
    outer = GridSpec(1, 3, figure=fig, wspace=0.18)
    for j, modality in enumerate(MODALITIES):
        ax = fig.add_subplot(outer[j])
        df = pd.read_csv(corr_base / scale / f"correlation_results_IBS_{modality}.csv")
        forest_panel(
            ax,
            df,
            "Difference_z",
            "Diff_ci90_z_low",
            "Diff_ci90_z_high",
            "p_equivalence_Diff_fdr",
            (-0.2, 0.2),
            0.112,
            show_y=(j == 0),
        )
        add_modality_label(ax, modality)
    fig.suptitle(scale, fontsize=10, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0.07, 1, 0.965])
    save_figure_all_formats(fig, stem)
    plt.close(fig)


def make_s3_forest_plot() -> None:
    corr_base = OUT_DIR / "correlation_ROME"
    scales = S3_SCALES
    fig = plt.figure(figsize=(14, 15), dpi=300)
    outer = GridSpec(2, 2, figure=fig, wspace=0.22, hspace=0.18)
    letters = ["a", "b", "c", "d"]
    for i, scale in enumerate(scales):
        inner = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[i], wspace=0.18)
        for j, modality in enumerate(MODALITIES):
            ax = fig.add_subplot(inner[j])
            df = pd.read_csv(corr_base / scale / f"correlation_results_IBS_{modality}.csv")
            forest_panel(
                ax,
                df,
                "Difference_z",
                "Diff_ci90_z_low",
                "Diff_ci90_z_high",
                "p_equivalence_Diff_fdr",
                (-0.2, 0.2),
                0.112,
                show_y=(j == 0),
            )
            add_modality_label(ax, modality, y=-0.14)
            if j == 0:
                ax.text(-0.55, 1.02, f"{letters[i]}  {scale}", transform=ax.transAxes, fontsize=9, fontweight="bold")
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    save_figure_all_formats(fig, "S3_Fig_all60")
    plt.close(fig)
    for scale in scales:
        make_s3_forest_plot_for_scale(scale, f"S3_Fig_all60_{SAFE_SCALE[scale]}")


def make_s4_forest_plot_for_scale(scale: str, stem: str) -> None:
    sex_base = OUT_DIR / "correlation_ROME_sex"
    fig = plt.figure(figsize=(8, 8.5), dpi=300)
    outer = GridSpec(1, 3, figure=fig, wspace=0.18)
    for j, modality in enumerate(MODALITIES):
        ax = fig.add_subplot(outer[j])
        df = pd.read_csv(sex_base / scale / f"correlation_results_HC_IBS_{modality}_sex_interaction.csv")
        forest_panel(
            ax,
            df,
            "DiD_value(Cohens_q)",
            "DiD_ci90_z_low",
            "DiD_ci90_z_high",
            "p_equivalence_DiD_fdr",
            (-0.4, 0.4),
            0.256,
            show_y=(j == 0),
        )
        add_modality_label(ax, modality)
    fig.suptitle(scale, fontsize=10, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0.07, 1, 0.965])
    save_figure_all_formats(fig, stem)
    plt.close(fig)


def make_s4_forest_plot() -> None:
    sex_base = OUT_DIR / "correlation_ROME_sex"
    fig = plt.figure(figsize=(14, 15), dpi=300)
    outer = GridSpec(2, 2, figure=fig, wspace=0.22, hspace=0.18)
    letters = ["a", "b", "c", "d"]
    for i, scale in enumerate(SCALES):
        inner = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[i], wspace=0.18)
        for j, modality in enumerate(MODALITIES):
            ax = fig.add_subplot(inner[j])
            df = pd.read_csv(sex_base / scale / f"correlation_results_HC_IBS_{modality}_sex_interaction.csv")
            forest_panel(
                ax,
                df,
                "DiD_value(Cohens_q)",
                "DiD_ci90_z_low",
                "DiD_ci90_z_high",
                "p_equivalence_DiD_fdr",
                (-0.4, 0.4),
                0.256,
                show_y=(j == 0),
            )
            add_modality_label(ax, modality, y=-0.14)
            if j == 0:
                ax.text(-0.55, 1.02, f"{letters[i]}  {scale}", transform=ax.transAxes, fontsize=9, fontweight="bold")
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    save_figure_all_formats(fig, "S4_Fig_all60")
    plt.close(fig)
    for scale in SCALES:
        make_s4_forest_plot_for_scale(scale, f"S4_Fig_all60_{SAFE_SCALE[scale]}")


def make_s2_brain_map() -> None:
    """Create an S2-style SA group-difference inconclusive brain-map panel."""
    try:
        sys.path.insert(0, str(CODE_DIR))
        from utils_norm.utils_visual import plot_brain_stat_figure
        from PIL import Image, ImageDraw, ImageFont
    except Exception as exc:
        (OUT_DIR / "figures" / "S2_Fig_all60_NOT_CREATED.txt").write_text(
            f"S2-style brain map was not created because the plotting dependency was unavailable:\n{exc}\n",
            encoding="utf-8",
        )
        print(f"Skipping S2 brain map: {exc}")
        return

    corr_base = OUT_DIR / "correlation_ROME"
    tmp_dir = OUT_DIR / "figures" / "_s2_brain_map_rows"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    row_paths = []
    for scale in SCALES:
        df = pd.read_csv(corr_base / scale / "correlation_results_IBS_SA.csv")
        df_to_plot = df.copy()
        # Match the previous S2-style "association-difference inconclusive" display:
        # keep group-difference effects that did not meet FDR-corrected equivalence.
        df_to_plot.loc[df_to_plot["p_equivalence_Diff_fdr"] < 0.05, "Difference_z"] = np.nan
        df_to_plot = df_to_plot[["ROI", "Difference_z"]].rename(columns={"ROI": "eid"})
        plot_brain_stat_figure(
            df_to_plot,
            "coolwarm",
            str(DOCU_DIR),
            str(tmp_dir),
            f"SA_IBS_diff_inconclusive_{SAFE_SCALE[scale]}",
            vmin=-0.2,
            vmax=0.2,
            colorbar=True,
            half=False,
        )
        row_paths.append(tmp_dir / f"brain_visualization_SA_IBS_diff_inconclusive_{SAFE_SCALE[scale]}.png")

    images = [Image.open(path).convert("RGBA") for path in row_paths]
    label_width = 260
    row_gap = 10
    width = label_width + max(im.width for im in images)
    height = sum(im.height for im in images) + row_gap * (len(images) - 1)
    canvas = Image.new("RGBA", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 44)
        small_font = ImageFont.truetype("arial.ttf", 36)
    except Exception:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    y = 0
    for idx, (scale, im) in enumerate(zip(SCALES, images)):
        if idx < len(images) - 1:
            # Blank repeated colorbar area; keep only the final row colorbar.
            blank = Image.new("RGBA", im.size, "white")
            blank.alpha_composite(im)
            draw_blank = ImageDraw.Draw(blank)
            draw_blank.rectangle([im.width - 260, 0, im.width, im.height], fill="white")
            im_to_paste = blank
        else:
            im_to_paste = im
        canvas.alpha_composite(im_to_paste, (label_width, y))
        draw.text((20, y + 30), chr(ord("a") + idx), fill="black", font=font)
        draw.text((80, y + 35), scale, fill="black", font=small_font)
        if idx < len(images) - 1:
            draw.line([(label_width, y + im.height + row_gap // 2), (width, y + im.height + row_gap // 2)], fill="black", width=2)
        y += im.height + row_gap

    for ext in ["png", "tif"]:
        canvas.save(OUT_DIR / "figures" / f"S2_Fig_all60.{ext}")
    canvas.convert("RGB").save(OUT_DIR / "figures" / "S2_Fig_all60.pdf")


def zscore(series: pd.Series) -> pd.Series:
    return (series - np.nanmean(series)) / np.nanstd(series, ddof=1)


def fit_one_roi(df: pd.DataFrame, roi: str, scale: str) -> dict[str, dict[str, float]]:
    model = ols(f"{roi} ~ {scale} * Group * C(sex, Sum)", data=df).fit()
    terms = [
        scale,
        f"{scale}:Group[T.IBS]",
        f"{scale}:C(sex, Sum)[S.Female]",
        f"{scale}:Group[T.IBS]:C(sex, Sum)[S.Female]",
    ]
    return {
        term: {
            "beta": model.params[term],
            "SE": model.bse[term],
            "p": model.pvalues[term],
            "df_resid": int(model.df_resid),
            "Residual_var": model.scale,
        }
        for term in terms
        if term in model.params.index
    }


def run_roiwise_ancova(df: pd.DataFrame, scale: str) -> dict[str, pd.DataFrame]:
    rows = {"Scale": [], "Scale:Group": [], "Scale:sex": [], "Scale:Group:sex": []}
    mapping = {
        scale: "Scale",
        f"{scale}:Group[T.IBS]": "Scale:Group",
        f"{scale}:C(sex, Sum)[S.Female]": "Scale:sex",
        f"{scale}:Group[T.IBS]:C(sex, Sum)[S.Female]": "Scale:Group:sex",
    }
    for roi in ROI_LIST:
        stats_by_term = fit_one_roi(df, roi, scale)
        for raw_term, label in mapping.items():
            if raw_term in stats_by_term:
                rows[label].append({"ROI": roi, "term": label, **stats_by_term[raw_term]})

    out = {}
    for effect, effect_rows in rows.items():
        effect_df = pd.DataFrame(effect_rows)
        if not effect_df.empty:
            effect_df["p_fdr"] = fdr(effect_df["p"])
            effect_df["signif"] = effect_df["p_fdr"] < 0.05
        out[effect] = effect_df
    return out


def save_scale_tables(effect_tables: dict[str, pd.DataFrame], scale: str, outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{scale}_ANCOVA_ROIwise.xlsx"
    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        for effect, df in effect_tables.items():
            if df.empty:
                continue
            sheet = effect.replace(":", "x")[:31]
            df.to_excel(writer, sheet_name=sheet, index=False)
            worksheet = writer.sheets[sheet]
            for idx, col in enumerate(df.columns):
                width = max(12, min(50, int(df[col].astype(str).str.len().max()) + 2))
                worksheet.set_column(idx, idx, width)
    return path


def scale_terms(scale: str) -> list[str]:
    return [scale, f"{scale}:Group", f"{scale}:sex", f"{scale}:Group:sex"]


def normalize_effect(effect_raw: str, scale_name: str) -> str:
    return effect_raw.replace(scale_name, "Scale")


def rerun_ancova_outputs(hc_cov: pd.DataFrame, ibs_cov: pd.DataFrame) -> None:
    ancova_base = OUT_DIR / "ANCOVA_correlation_ROME"
    mancova_buckets = {"Scale": [], "Scale:Group": [], "Scale:sex": [], "Scale:Group:sex": []}
    test_name = "Wilks' lambda"

    hc_safe = hc_cov.rename(columns=SAFE_SCALE).copy()
    ibs_safe = ibs_cov.rename(columns=SAFE_SCALE).copy()
    safe_scales = [SAFE_SCALE[s] for s in SCALES]
    sex_map = {0: "Female", 1: "Male"}
    hc_safe["sex"] = hc_safe["sex"].map(sex_map).astype("category")
    ibs_safe["sex"] = ibs_safe["sex"].map(sex_map).astype("category")
    hc_safe["Group"] = "HC"
    ibs_safe["Group"] = "IBS"

    for modality in MODALITIES:
        print(f"ANCOVA/MANCOVA analyses: {modality}")
        hc_dev = load_modality_deviation(modality, "HC")
        ibs_dev = load_modality_deviation(modality, "IBS")
        hc_all = pd.merge(hc_safe[["eid", "sex", "Group"] + safe_scales], hc_dev, on="eid", how="inner")
        ibs_all = pd.merge(ibs_safe[["eid", "sex", "Group"] + safe_scales], ibs_dev, on="eid", how="inner")
        df = pd.concat([hc_all, ibs_all], ignore_index=True)
        df["Group"] = pd.Categorical(df["Group"], categories=["HC", "IBS"])
        df["sex"] = pd.Categorical(df["sex"], categories=["Female", "Male"])
        df.dropna(how="all", axis=1, inplace=True)
        df.dropna(inplace=True)

        for scale in safe_scales:
            df_model = df.copy()
            df_model[scale] = zscore(df_model[scale])
            effect_tables = run_roiwise_ancova(df_model, scale)
            save_scale_tables(effect_tables, scale, ancova_base / modality)

        # Multivariate omnibus association tests.
        for scale in safe_scales:
            df_model = df.copy()
            df_model[scale] = zscore(df_model[scale])
            formula = " + ".join(ROI_LIST) + f" ~ {scale} * Group * sex"
            mv = MANOVA.from_formula(formula, data=df_model)
            res = mv.mv_test()
            for term in scale_terms(scale):
                if term not in res.results:
                    continue
                stat_df = res.results[term]["stat"]
                row = stat_df.loc[test_name] if test_name in stat_df.index else stat_df.loc["Wilks' lambda"]
                effect = normalize_effect(term, scale)
                mancova_buckets[effect].append(
                    {
                        "measure": modality,
                        "scale": scale,
                        "effect": effect,
                        "test": row.name,
                        "stat_value": row.get("Value", None),
                        "F": row.get("F Value", None),
                        "df1": row.get("Num DF", None),
                        "df2": row.get("Den DF", None),
                        "p": row.get("Pr > F", None),
                    }
                )

    mancova_base = ancova_base / "MANCOVA"
    for effect, rows in mancova_buckets.items():
        df_effect = pd.DataFrame(rows)
        if not df_effect.empty:
            df_effect["p_fdr"] = fdr(pd.to_numeric(df_effect["p"], errors="coerce"))
            df_effect["signif"] = df_effect["p_fdr"] < 0.05
        for scale in safe_scales:
            scale_dir = mancova_base / scale
            scale_dir.mkdir(parents=True, exist_ok=True)
            df_effect.loc[df_effect["scale"] == scale].to_csv(scale_dir / f"{effect.replace(':', 'x')}.csv", index=False)

    run_ancova_equivalence_outputs()


def tost_from_beta(beta: float, se: float, df: float, low: float, high: float, alpha: float = 0.05) -> dict[str, object]:
    if not np.isfinite(beta) or not np.isfinite(se) or se <= 0:
        return {"beta": beta, "p_lower": np.nan, "p_upper": np.nan, "p_max": np.nan, "eq": False, "ci_lo": np.nan, "ci_hi": np.nan}
    t1 = (beta - low) / se
    t2 = (high - beta) / se
    p1 = 1 - stats.t.cdf(t1, df)
    p2 = 1 - stats.t.cdf(t2, df)
    tcrit_90 = stats.t.ppf(0.95, df)
    ci_lo = beta - tcrit_90 * se
    ci_hi = beta + tcrit_90 * se
    return {"beta": beta, "p_lower": float(p1), "p_upper": float(p2), "p_max": float(max(p1, p2)), "eq": bool((p1 < alpha) and (p2 < alpha)), "ci_lo": float(ci_lo), "ci_hi": float(ci_hi)}


def run_ancova_equivalence_outputs() -> None:
    ancova_base = OUT_DIR / "ANCOVA_correlation_ROME"
    sesoi_dict = {
        "group": {"custom": 0.1, "95": 0.112, "90": 0.102},
        "interaction": {"custom": 0.1, "95": 0.128, "90": 0.117},
    }
    safe_scales = [SAFE_SCALE[s] for s in SCALES]
    effect_to_sheet = {
        "Scale_Group": ("ScalexGroup", "group"),
        "Scale_Group_se_": ("ScalexGroupxsex", "interaction"),
    }

    for modality in MODALITIES:
        for scale in safe_scales:
            workbook = ancova_base / modality / f"{scale}_ANCOVA_ROIwise.xlsx"
            if not workbook.exists():
                continue
            for effect_safe, (sheet_name, sesoi_direction) in effect_to_sheet.items():
                effect_df = pd.read_excel(workbook, sheet_name=sheet_name)
                for sesoi_label, sesoi_value in sesoi_dict[sesoi_direction].items():
                    rows = []
                    for _, row in effect_df.iterrows():
                        result = tost_from_beta(
                            beta=row["beta"],
                            se=row["SE"],
                            df=row["df_resid"],
                            low=-sesoi_value,
                            high=sesoi_value,
                        )
                        rows.append({"ROI": row["ROI"], **result})
                    out = pd.DataFrame(rows)
                    if not out.empty:
                        out["p_max_fdr"] = fdr(out["p_max"])
                        out["fdr_reject"] = out["p_max_fdr"] < 0.05
                    out_dir = ancova_base / modality / sesoi_label
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out.to_csv(out_dir / f"{scale}_{effect_safe}_EQ_test_results.csv", index=False)


def compare_old_new_overlap() -> pd.DataFrame:
    if OLD_CORR_DIR is None or OLD_SEX_DIR is None:
        out = pd.DataFrame()
        out.to_csv(OUT_DIR / "association_all60_old_overlap_check.csv", index=False)
        return out

    rows = []
    old_corr = OLD_CORR_DIR
    old_sex = OLD_SEX_DIR
    new_corr = OUT_DIR / "correlation_ROME"
    new_sex = OUT_DIR / "correlation_ROME_sex"
    numeric_cols_main = ["HC_r", "IBS_r", "HC_cohens_d", "IBS_cohens_d", "Difference_z", "p_Diff", "p_equivalence_Diff"]
    numeric_cols_did = ["IBS_F_r", "IBS_M_r", "HC_F_r", "HC_M_r", "DiD_value(Cohens_q)", "p_DiD", "p_equivalence_DiD"]
    for scale in SCALES:
        for modality in MODALITIES:
            old_path = old_corr / scale / f"correlation_results_IBS_{modality}.csv"
            if old_path.exists():
                old = pd.read_csv(old_path).set_index("ROI")
                new = pd.read_csv(new_corr / scale / f"correlation_results_IBS_{modality}.csv").set_index("ROI")
                common = old.index.intersection(new.index)
                max_abs = np.nanmax(np.abs(old.loc[common, numeric_cols_main].values - new.loc[common, numeric_cols_main].values))
                rows.append({"analysis": "correlation_main", "scale": scale, "modality": modality, "old_n": len(old), "new_n": len(new), "overlap_n": len(common), "max_abs_diff": max_abs})
            old_path = old_sex / scale / f"correlation_results_HC_IBS_{modality}_sex_interaction.csv"
            if old_path.exists():
                old = pd.read_csv(old_path).set_index("ROI")
                new = pd.read_csv(new_sex / scale / f"correlation_results_HC_IBS_{modality}_sex_interaction.csv").set_index("ROI")
                common = old.index.intersection(new.index)
                max_abs = np.nanmax(np.abs(old.loc[common, numeric_cols_did].values - new.loc[common, numeric_cols_did].values))
                rows.append({"analysis": "did_interaction", "scale": scale, "modality": modality, "old_n": len(old), "new_n": len(new), "overlap_n": len(common), "max_abs_diff": max_abs})
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "association_all60_old_overlap_check.csv", index=False)
    return out


def write_summary() -> Path:
    rows = []
    corr_base = OUT_DIR / "correlation_ROME"
    sex_base = OUT_DIR / "correlation_ROME_sex"
    ancova_base = OUT_DIR / "ANCOVA_correlation_ROME"
    for scale in SCALES:
        for modality in MODALITIES:
            main = pd.read_csv(corr_base / scale / f"correlation_results_IBS_{modality}.csv")
            did = pd.read_csv(sex_base / scale / f"correlation_results_HC_IBS_{modality}_sex_interaction.csv")
            rows.append(
                {
                    "analysis": "correlation_main",
                    "scale": scale,
                    "modality": modality,
                    "n_roi": len(main),
                    "HC_p_fdr_lt_0.05": int((main["HC_p_fdr"] < 0.05).sum()),
                    "IBS_p_fdr_lt_0.05": int((main["IBS_p_fdr"] < 0.05).sum()),
                    "Difference_p_fdr_lt_0.05": int((main["p_Diff_fdr"] < 0.05).sum()),
                    "Difference_equivalence_fdr_lt_0.05": int((main["p_equivalence_Diff_fdr"] < 0.05).sum()),
                }
            )
            rows.append(
                {
                    "analysis": "did_interaction",
                    "scale": scale,
                    "modality": modality,
                    "n_roi": len(did),
                    "DiD_p_fdr_lt_0.05": int((did["p_DiD_fdr"] < 0.05).sum()),
                    "DiD_equivalence_fdr_lt_0.05": int((did["p_equivalence_DiD_fdr"] < 0.05).sum()),
                }
            )
            safe = SAFE_SCALE[scale]
            workbook = ancova_base / modality / f"{safe}_ANCOVA_ROIwise.xlsx"
            if workbook.exists():
                for sheet in ["Scale", "ScalexGroup", "Scalexsex", "ScalexGroupxsex"]:
                    df = pd.read_excel(workbook, sheet_name=sheet)
                    rows.append(
                        {
                            "analysis": "roiwise_ancova",
                            "scale": scale,
                            "modality": modality,
                            "effect": sheet,
                            "n_roi": len(df),
                            "p_fdr_lt_0.05": int((df["p_fdr"] < 0.05).sum()),
                        }
                    )
    out = pd.DataFrame(rows)
    path = OUT_DIR / "association_all60_summary.csv"
    out.to_csv(path, index=False)
    return path


def main() -> None:
    args = parse_args()
    configure_paths(args)
    ensure_dirs()
    hc_cov, ibs_cov = load_covariates()
    rerun_correlation_outputs(hc_cov, ibs_cov)
    s11_path = write_s11_table()
    s12_path = write_s12_table()
    make_s3_forest_plot()
    make_s4_forest_plot()
    if MAKE_BRAIN_MAP:
        make_s2_brain_map()
    rerun_ancova_outputs(hc_cov, ibs_cov)
    overlap = compare_old_new_overlap()
    summary_path = write_summary()

    figure_paths = [
        str(OUT_DIR / "figures" / "S3_Fig_all60.tif"),
        str(OUT_DIR / "figures" / "S4_Fig_all60.tif"),
    ]
    s2_path = OUT_DIR / "figures" / "S2_Fig_all60.tif"
    if s2_path.exists():
        figure_paths.insert(0, str(s2_path))
    figure_paths.extend(
        str(OUT_DIR / "figures" / f"S3_Fig_all60_{SAFE_SCALE[scale]}.tif")
        for scale in S3_SCALES
    )
    figure_paths.extend(
        str(OUT_DIR / "figures" / f"S4_Fig_all60_{SAFE_SCALE[scale]}.tif")
        for scale in SCALES
    )

    manifest = {
        "output_dir": str(OUT_DIR),
        "tables": [str(s11_path), str(s12_path)],
        "figures": figure_paths,
        "summary": str(summary_path),
        "overlap_check_max_abs_diff": float(overlap["max_abs_diff"].max()) if not overlap.empty else None,
    }
    (OUT_DIR / "association_all60_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
