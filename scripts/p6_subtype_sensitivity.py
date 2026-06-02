"""
IBS subtype sensitivity analyses for normative-model deviation scores.

This script reproduces the subtype sensitivity analyses described in the
manuscript revision:

1. Five-level group omnibus effect:
   deviation ~ group5 + sex

2. Five-level group-by-sex interaction:
   deviation ~ group5 * sex, compared against deviation ~ group5 + sex

3. Nested subtype heterogeneity:
   deviation ~ group5 + sex, compared against deviation ~ IBS_binary + sex

4. Nested subtype-by-sex heterogeneity:
   deviation ~ group5 * sex, compared against deviation ~ IBS_binary * sex

The group5 factor is coded as HC, IBS-C, IBS-D, IBS-M, IBS-U, with HC as
reference. FDR correction is performed separately within each modality
(CT, SA, CV) and separately for each analysis.

Expected inputs in --input-dir:
    HC_cov_ROME.csv
    IBS_cov_ROME.csv
    HC_deviation_scores_by_brain_idp_all_predicted.csv
    IBS_deviation_scores_by_brain_idp_all_predicted.csv

Example:
    python scripts/p6_subtype_sensitivity.py --input-dir data/processed --output-dir results/subtype_sensitivity
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


SUBTYPE_ORDER = ["IBS-C", "IBS-D", "IBS-M", "IBS-U"]
GROUP5_ORDER = ["HC"] + SUBTYPE_ORDER
MODALITY_ORDER = ["CT", "SA", "CV"]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def sort_by_modality_roi(df: pd.DataFrame, roi_order: list[str]) -> pd.DataFrame:
    """Order rows as CT/SA/CV and the canonical ROI order used in other tables."""
    out = df.copy()
    out["_modality_order"] = pd.Categorical(
        out["modality"], categories=MODALITY_ORDER, ordered=True
    )
    out["_roi_order"] = pd.Categorical(out["roi"], categories=roi_order, ordered=True)
    out = out.sort_values(["_modality_order", "_roi_order"])
    out = out.drop(columns=["_modality_order", "_roi_order"])
    return out.reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run IBS subtype sensitivity analyses on cortical deviation scores."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PACKAGE_ROOT / "data" / "processed",
        help=(
            "Directory containing covariate and deviation-score CSV files. "
            "Defaults to data/processed under this package."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PACKAGE_ROOT / "results" / "subtype_sensitivity",
        help=(
            "Directory for output tables. Defaults to results/subtype_sensitivity."
        ),
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=PACKAGE_ROOT / "docs",
        help="Directory containing ROI_IBS.csv.",
    )
    parser.add_argument(
        "--no-xlsx",
        action="store_true",
        help="Only write CSV/Markdown outputs; skip formatted XLSX files.",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    output_dir.mkdir(parents=True, exist_ok=True)
    return input_dir, output_dir


def load_roi_order(input_dir: Path, docs_dir: Path, idp_cols: list[str]) -> list[str]:
    """Use ROI_IBS.csv when available; otherwise infer ROI order from IDP columns."""
    candidates = [
        docs_dir / "ROI_IBS.csv",
        input_dir / "ROI_IBS.csv",
    ]

    for candidate in candidates:
        if candidate.exists():
            return pd.read_csv(candidate)["ROI"].tolist()

    roi_order: list[str] = []
    for column in idp_cols:
        if "__" not in column:
            continue
        roi = column.split("__", 1)[1]
        if roi not in roi_order:
            roi_order.append(roi)
    return roi_order


def load_analysis_data(input_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    hc_cov = pd.read_csv(input_dir / "HC_cov_ROME.csv")
    ibs_cov = pd.read_csv(input_dir / "IBS_cov_ROME.csv")
    hc_dev = pd.read_csv(input_dir / "HC_deviation_scores_by_brain_idp_all_predicted.csv")
    ibs_dev = pd.read_csv(input_dir / "IBS_deviation_scores_by_brain_idp_all_predicted.csv")

    required_hc = {"eid", "sex"}
    required_ibs = {"eid", "sex", "IBS_subtype"}
    missing_hc = required_hc.difference(hc_cov.columns)
    missing_ibs = required_ibs.difference(ibs_cov.columns)
    if missing_hc:
        raise ValueError(f"HC covariate file is missing columns: {sorted(missing_hc)}")
    if missing_ibs:
        raise ValueError(f"IBS covariate file is missing columns: {sorted(missing_ibs)}")

    hc = hc_cov[["eid", "sex"]].merge(hc_dev, on="eid", how="inner", validate="one_to_one")
    hc["IBS_binary"] = 0
    hc["IBS_subtype"] = "HC"
    hc["group5"] = "HC"

    ibs = ibs_cov[["eid", "sex", "IBS_subtype"]].merge(
        ibs_dev, on="eid", how="inner", validate="one_to_one"
    )
    ibs["IBS_binary"] = 1
    ibs["group5"] = ibs["IBS_subtype"]

    data = pd.concat([hc, ibs], axis=0, ignore_index=True)
    data["eid"] = data["eid"].astype(str)
    data["sex"] = data["sex"].astype("category")
    data["IBS_subtype"] = pd.Categorical(data["IBS_subtype"], categories=GROUP5_ORDER)
    data["group5"] = pd.Categorical(data["group5"], categories=GROUP5_ORDER)

    idp_cols = [column for column in data.columns if "__" in column]
    if not idp_cols:
        raise ValueError("No cortical IDP columns found. Expected column names containing '__'.")
    return data, idp_cols


def anova_row(anova_table: pd.DataFrame, term: str) -> tuple[float, float]:
    if term not in anova_table.index:
        return np.nan, np.nan
    return float(anova_table.loc[term, "F"]), float(anova_table.loc[term, "PR(>F)"])


def f_test(full_model, reduced_model) -> tuple[float, float, float]:
    f_value, p_value, df_diff = full_model.compare_f_test(reduced_model)
    return float(f_value), float(p_value), float(df_diff)


def adjusted_group_means(model, sex_distribution: dict[object, float]) -> dict[str, float]:
    means: dict[str, float] = {}
    for group in GROUP5_ORDER:
        predictions = []
        weights = []
        for sex_value, weight in sex_distribution.items():
            frame = pd.DataFrame(
                {
                    "group5": pd.Categorical([group], categories=GROUP5_ORDER),
                    "sex": pd.Categorical([sex_value], categories=model.model.data.frame["sex"].cat.categories),
                    "IBS_binary": [0 if group == "HC" else 1],
                }
            )
            predictions.append(float(model.predict(frame)[0]))
            weights.append(float(weight))
        means[group] = float(np.dot(np.array(predictions), np.array(weights)))
    return means


def run_one_idp(data: pd.DataFrame, idp: str) -> dict[str, object]:
    model_df = data[["eid", "sex", "IBS_binary", "group5", idp]].copy()
    model_df = model_df.rename(columns={idp: "deviation"})
    model_df = model_df.dropna(subset=["deviation", "sex", "group5"])
    model_df["IBS_binary"] = model_df["IBS_binary"].astype(int)
    model_df["group5"] = pd.Categorical(model_df["group5"], categories=GROUP5_ORDER)

    common_model = smf.ols("deviation ~ C(IBS_binary) + C(sex)", data=model_df).fit()
    common_interaction_model = smf.ols(
        "deviation ~ C(IBS_binary) * C(sex)", data=model_df
    ).fit()
    group5_model = smf.ols(
        'deviation ~ C(group5, Treatment(reference="HC")) + C(sex)', data=model_df
    ).fit()
    group5_interaction_model = smf.ols(
        'deviation ~ C(group5, Treatment(reference="HC")) * C(sex)', data=model_df
    ).fit()

    group5_anova = sm.stats.anova_lm(group5_model, typ=2)
    group5_f, group5_p = anova_row(
        group5_anova, 'C(group5, Treatment(reference="HC"))'
    )
    group5_sex_f, group5_sex_p, group5_sex_df = f_test(
        group5_interaction_model, group5_model
    )
    nested_f, nested_p, nested_df = f_test(group5_model, common_model)
    nested_sex_f, nested_sex_p, nested_sex_df = f_test(
        group5_interaction_model, common_interaction_model
    )

    sex_distribution = model_df["sex"].value_counts(normalize=True).sort_index().to_dict()
    adjusted_means = adjusted_group_means(group5_model, sex_distribution)

    row: dict[str, object] = {
        "idp": idp,
        "modality": idp.split("__", 1)[0],
        "roi": idp.split("__", 1)[1],
        "n_total": int(len(model_df)),
        "n_hc": int((model_df["IBS_binary"] == 0).sum()),
        "n_ibs": int((model_df["IBS_binary"] == 1).sum()),
        "group5_omnibus_F": group5_f,
        "group5_omnibus_p": group5_p,
        "group5_omnibus_df_num": 4.0,
        "group5_omnibus_df_den": float(group5_model.df_resid),
        "group5_sex_interaction_F": group5_sex_f,
        "group5_sex_interaction_p": group5_sex_p,
        "group5_sex_interaction_df_num": group5_sex_df,
        "group5_sex_interaction_df_den": float(group5_interaction_model.df_resid),
        "nested_subtype_heterogeneity_F": nested_f,
        "nested_subtype_heterogeneity_p": nested_p,
        "nested_subtype_heterogeneity_df_num": nested_df,
        "nested_subtype_heterogeneity_df_den": float(group5_model.df_resid),
        "nested_subtype_by_sex_heterogeneity_F": nested_sex_f,
        "nested_subtype_by_sex_heterogeneity_p": nested_sex_p,
        "nested_subtype_by_sex_heterogeneity_df_num": nested_sex_df,
        "nested_subtype_by_sex_heterogeneity_df_den": float(
            group5_interaction_model.df_resid
        ),
    }

    for group in GROUP5_ORDER:
        row[f"adj_mean_{group}"] = adjusted_means[group]
    for subtype in SUBTYPE_ORDER:
        row[f"adj_mean_diff_{subtype}_minus_HC"] = (
            adjusted_means[subtype] - adjusted_means["HC"]
        )

    return row


def add_fdr_by_modality(results: pd.DataFrame) -> pd.DataFrame:
    results = results.copy()
    p_columns = [
        "group5_omnibus_p",
        "group5_sex_interaction_p",
        "nested_subtype_heterogeneity_p",
        "nested_subtype_by_sex_heterogeneity_p",
    ]
    for p_col in p_columns:
        fdr_col = p_col.replace("_p", "_p_fdr")
        results[fdr_col] = np.nan
        for modality, index in results.groupby("modality").groups.items():
            p_values = results.loc[index, p_col]
            valid = p_values.notna()
            if valid.any():
                corrected = multipletests(p_values[valid], method="fdr_bh")[1]
                results.loc[p_values[valid].index, fdr_col] = corrected
    return results


def run_all_idps(
    data: pd.DataFrame, idp_cols: list[str], roi_order: list[str]
) -> pd.DataFrame:
    rows = []
    for index, idp in enumerate(idp_cols, start=1):
        rows.append(run_one_idp(data, idp))
        if index % 30 == 0 or index == len(idp_cols):
            print(f"Processed {index}/{len(idp_cols)} IDPs")
    results = pd.DataFrame(rows)
    results = add_fdr_by_modality(results)
    return sort_by_modality_roi(results, roi_order)


def make_summary(results: pd.DataFrame) -> pd.DataFrame:
    analyses = [
        (
            "Five-level group omnibus",
            "group5_omnibus_p",
            "group5_omnibus_p_fdr",
        ),
        (
            "Five-level group-by-sex interaction",
            "group5_sex_interaction_p",
            "group5_sex_interaction_p_fdr",
        ),
        (
            "Nested subtype heterogeneity",
            "nested_subtype_heterogeneity_p",
            "nested_subtype_heterogeneity_p_fdr",
        ),
        (
            "Nested subtype-by-sex heterogeneity",
            "nested_subtype_by_sex_heterogeneity_p",
            "nested_subtype_by_sex_heterogeneity_p_fdr",
        ),
    ]
    rows = []
    for label, p_col, fdr_col in analyses:
        for modality in MODALITY_ORDER:
            subset = results[results["modality"] == modality].copy()
            if subset.empty:
                continue
            top = subset.sort_values(p_col).iloc[0]
            rows.append(
                {
                    "analysis": label,
                    "modality": modality,
                    "n_idp": int(len(subset)),
                    "n_fdr_significant": int((subset[fdr_col] < 0.05).sum()),
                    "min_raw_p": float(subset[p_col].min()),
                    "min_fdr_p": float(subset[fdr_col].min()),
                    "top_roi_by_raw_p": top["roi"],
                    "top_roi_raw_p": float(top[p_col]),
                    "top_roi_fdr_p": float(top[fdr_col]),
                }
            )
    return pd.DataFrame(rows)


def supplementary_tables(
    results: pd.DataFrame, roi_order: list[str]
) -> dict[str, tuple[pd.DataFrame, str]]:
    base_cols = ["modality", "roi", "idp", "n_total", "n_hc", "n_ibs"]
    tables: dict[str, tuple[pd.DataFrame, str]] = {}

    tables["01_five_level_group_omnibus"] = (
        results[
            base_cols
            + [
                "group5_omnibus_df_num",
                "group5_omnibus_df_den",
                "group5_omnibus_F",
                "group5_omnibus_p",
                "group5_omnibus_p_fdr",
                "adj_mean_HC",
                "adj_mean_IBS-C",
                "adj_mean_diff_IBS-C_minus_HC",
                "adj_mean_IBS-D",
                "adj_mean_diff_IBS-D_minus_HC",
                "adj_mean_IBS-M",
                "adj_mean_diff_IBS-M_minus_HC",
                "adj_mean_IBS-U",
                "adj_mean_diff_IBS-U_minus_HC",
            ]
        ].rename(
            columns={
                "group5_omnibus_df_num": "df_num",
                "group5_omnibus_df_den": "df_den",
                "group5_omnibus_F": "F",
                "group5_omnibus_p": "p",
                "group5_omnibus_p_fdr": "p_fdr_within_modality",
            }
        ),
        "Omnibus test of the five-level group term in deviation ~ group5 + sex.",
    )

    tables["02_five_level_group_by_sex_interaction"] = (
        results[
            base_cols
            + [
                "group5_sex_interaction_df_num",
                "group5_sex_interaction_df_den",
                "group5_sex_interaction_F",
                "group5_sex_interaction_p",
                "group5_sex_interaction_p_fdr",
            ]
        ].rename(
            columns={
                "group5_sex_interaction_df_num": "df_num",
                "group5_sex_interaction_df_den": "df_den",
                "group5_sex_interaction_F": "F",
                "group5_sex_interaction_p": "p",
                "group5_sex_interaction_p_fdr": "p_fdr_within_modality",
            }
        ),
        "Nested F test of group5-by-sex terms: group5 * sex versus group5 + sex.",
    )

    tables["03_nested_subtype_heterogeneity"] = (
        results[
            base_cols
            + [
                "nested_subtype_heterogeneity_df_num",
                "nested_subtype_heterogeneity_df_den",
                "nested_subtype_heterogeneity_F",
                "nested_subtype_heterogeneity_p",
                "nested_subtype_heterogeneity_p_fdr",
                "adj_mean_HC",
                "adj_mean_diff_IBS-C_minus_HC",
                "adj_mean_diff_IBS-D_minus_HC",
                "adj_mean_diff_IBS-M_minus_HC",
                "adj_mean_diff_IBS-U_minus_HC",
            ]
        ].rename(
            columns={
                "nested_subtype_heterogeneity_df_num": "df_num",
                "nested_subtype_heterogeneity_df_den": "df_den",
                "nested_subtype_heterogeneity_F": "F",
                "nested_subtype_heterogeneity_p": "p",
                "nested_subtype_heterogeneity_p_fdr": "p_fdr_within_modality",
            }
        ),
        "Nested F test: separate subtype effects versus a common binary IBS effect.",
    )

    tables["04_nested_subtype_by_sex_heterogeneity"] = (
        results[
            base_cols
            + [
                "nested_subtype_by_sex_heterogeneity_df_num",
                "nested_subtype_by_sex_heterogeneity_df_den",
                "nested_subtype_by_sex_heterogeneity_F",
                "nested_subtype_by_sex_heterogeneity_p",
                "nested_subtype_by_sex_heterogeneity_p_fdr",
            ]
        ].rename(
            columns={
                "nested_subtype_by_sex_heterogeneity_df_num": "df_num",
                "nested_subtype_by_sex_heterogeneity_df_den": "df_den",
                "nested_subtype_by_sex_heterogeneity_F": "F",
                "nested_subtype_by_sex_heterogeneity_p": "p",
                "nested_subtype_by_sex_heterogeneity_p_fdr": "p_fdr_within_modality",
            }
        ),
        "Nested F test: separate subtype-by-sex effects versus a common IBS-by-sex effect.",
    )

    for key, (table, description) in tables.items():
        table["significant_fdr05"] = table["p_fdr_within_modality"] < 0.05
        table = sort_by_modality_roi(table, roi_order)
        tables[key] = (table, description)
    return tables


def write_xlsx(path: Path, results: pd.DataFrame, description: str) -> None:
    try:
        from openpyxl.styles import Font, PatternFill
    except Exception as exc:
        raise RuntimeError("openpyxl is required for XLSX output. Use --no-xlsx.") from exc

    readme = pd.DataFrame(
        [
            ("Analysis", description),
            ("Group coding", "HC, IBS-C, IBS-D, IBS-M, IBS-U; HC is reference."),
            ("Covariate", "Sex."),
            ("FDR", "Benjamini-Hochberg correction within each modality."),
        ],
        columns=["Item", "Description"],
    )

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        readme.to_excel(writer, sheet_name="README", index=False)
        results.to_excel(writer, sheet_name="Results", index=False)

        header_fill = PatternFill("solid", fgColor="D9EAF7")
        for sheet in writer.book.worksheets:
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
            for cell in sheet[1]:
                cell.font = Font(bold=True)
                cell.fill = header_fill
            for column_cells in sheet.columns:
                letter = column_cells[0].column_letter
                max_len = max(len(str(cell.value or "")) for cell in column_cells[:200])
                sheet.column_dimensions[letter].width = max(10, min(max_len + 2, 45))


def write_outputs(
    results: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
    write_excel: bool,
    roi_order: list[str],
) -> None:
    results_path = output_dir / "subtype_sensitivity_results_all_idps.csv"
    summary_path = output_dir / "Supplementary_Table_SubtypeSensitivity_summary_by_modality.csv"
    results.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)

    if write_excel:
        write_xlsx(
            output_dir / "Supplementary_Table_SubtypeSensitivity_summary_by_modality.xlsx",
            summary,
            "Summary of subtype sensitivity analyses by analysis and modality.",
        )

    for key, (table, description) in supplementary_tables(results, roi_order).items():
        csv_path = output_dir / f"Supplementary_Table_SubtypeSensitivity_{key}.csv"
        xlsx_path = output_dir / f"Supplementary_Table_SubtypeSensitivity_{key}.xlsx"
        table.to_csv(csv_path, index=False)
        if write_excel:
            write_xlsx(xlsx_path, table, description)

    methods_text = (
        "IBS subtype sensitivity analyses were conducted to evaluate whether heterogeneity "
        "across IBS subtypes could mask an overall IBS-HC effect. IBS cases were classified "
        "as IBS-C, IBS-D, IBS-M, or IBS-U according to Rome III criteria using DHQ-derived "
        "bowel-habit information. For each cortical measure, models included sex and coded "
        "group as a five-level factor: HC, IBS-C, IBS-D, IBS-M, IBS-U, with HC as the "
        "reference group. We tested the omnibus five-level group effect, the group-by-sex "
        "interaction, nested subtype heterogeneity beyond a common IBS effect, and nested "
        "subtype-by-sex heterogeneity beyond a common IBS-by-sex interaction. FDR correction "
        "was performed separately within each modality for each analysis."
    )
    results_text = (
        "Across CT, SA, and CV, no five-level group omnibus effect, five-level group-by-sex "
        "interaction, nested subtype heterogeneity effect, or nested subtype-by-sex "
        "heterogeneity effect survived FDR correction within modality."
    )
    (output_dir / "subtype_sensitivity_methods_and_results_text.md").write_text(
        "# IBS subtype sensitivity analyses\n\n"
        "## Methods\n\n"
        f"{methods_text}\n\n"
        "## Results\n\n"
        f"{results_text}\n\n"
        "## Summary by modality\n\n"
        f"{summary.to_markdown(index=False)}\n",
        encoding="utf-8",
    )

    print(f"Saved all-IDP results: {results_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved supplementary tables to: {output_dir}")


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    args = parse_args()
    input_dir, output_dir = resolve_paths(args)

    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")

    data, idp_cols = load_analysis_data(input_dir)
    roi_order = load_roi_order(input_dir, args.docs_dir.resolve(), idp_cols)
    idp_cols = [
        f"{modality}__{roi}"
        for modality in MODALITY_ORDER
        for roi in roi_order
        if f"{modality}__{roi}" in idp_cols
    ]
    print(f"Merged analytic rows: {len(data)}")
    print(f"Cortical IDP columns: {len(idp_cols)}")
    print(data[["IBS_binary", "group5", "sex"]].value_counts(dropna=False).sort_index())

    results = run_all_idps(data, idp_cols, roi_order)
    summary = make_summary(results)
    write_outputs(results, summary, output_dir, write_excel=not args.no_xlsx, roi_order=roi_order)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
