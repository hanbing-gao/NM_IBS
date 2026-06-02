"""Define IBS/HC cohorts, IBS subtypes, matching, and analysis covariates.

This script starts from researcher-exported UK Biobank tabular data. No raw UKB
data are included in this repository. Users with UKB access should provide a CSV
containing the required field-instance columns described in docs/DATA_REQUIREMENTS.md.

Main outputs:
    data/processed/HC_cov_ROME.csv
    data/processed/IBS_cov_ROME.csv
    data/processed/IBS_matched_pairs.csv
    data/processed/cohort_flow_summary.csv

Example:
    python scripts/p1_define_cohorts_and_covariates.py --ukb-csv data/raw_ukb_exports/ukb_fields.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


PACKAGE_ROOT = Path(__file__).resolve().parents[1]

HC_SELF_REPORT_EXCLUSIONS = {
    1135,
    1154,
    1164,
    1165,
    1191,
    1456,
    1458,
    1459,
    1461,
    1462,
    1463,
    1509,
    1510,
    1562,
    1599,
    1600,
    1601,
    1602,
}
IBS_SELF_REPORT_EXCLUSIONS = {
    1135,
    1164,
    1165,
    1191,
    1456,
    1459,
    1461,
    1462,
    1463,
    1509,
    1600,
    1601,
    1602,
}

RENAME_COLUMNS = {
    "31-0.0": "sex",
    "54-2.0": "site",
    "21003-2.0": "age",
    "21001-2.0": "BMI",
    "48-2.0": "WC",
    "21862-2.0": "date_visit_assessment_center",
    "21000-2.0": "ethnic_background",
    "20116-2.0": "smoking_status",
    "738-2.0": "averaged_house_hold_income",
    "21023-0.0": "DHQ_finish_time",
    "20400-0.0": "mental_health_finish_time",
}

IBS_SSS_PROMPT_COLUMNS = ["21035-0.0", "21035-0.0", "21038-0.0"]
IBS_SSS_SCORE_COLUMNS = ["21036-0.0", "21037-0.0", "21039-0.0"]
IBS_SSS_OTHER_COLUMNS = ["21040-0.0", "21041-0.0"]
PHQ12_COLUMNS = [
    "21048-0.0",
    "21051-0.0",
    "21052-0.0",
    "21053-0.0",
    "21054-0.0",
    "21055-0.0",
    "21056-0.0",
    "21057-0.0",
    "21049-0.0",
    "21060-0.0",
    "21061-0.0",
]
GAD7_COLUMNS = [
    "20506-0.0",
    "20509-0.0",
    "20520-0.0",
    "20515-0.0",
    "20516-0.0",
    "20505-0.0",
    "20512-0.0",
]
PHQ9_COLUMNS = [
    "20514-0.0",
    "20510-0.0",
    "20517-0.0",
    "20519-0.0",
    "20511-0.0",
    "20507-0.0",
    "20508-0.0",
    "20518-0.0",
    "20513-0.0",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply IBS/HC inclusion-exclusion rules, define IBS subtypes, and propensity-match HC to IBS."
    )
    parser.add_argument("--ukb-csv", type=Path, required=True, help="Researcher-exported UKB field CSV.")
    parser.add_argument("--docs-dir", type=Path, default=PACKAGE_ROOT / "docs")
    parser.add_argument("--output-dir", type=Path, default=PACKAGE_ROOT / "data" / "processed")
    parser.add_argument(
        "--alcohol-csv",
        type=Path,
        default=None,
        help="Optional CSV with columns eid and Alc, merged into covariate outputs.",
    )
    parser.add_argument(
        "--disease-history-csv",
        type=Path,
        default=None,
        help="Optional first-occurrence/history CSV with eid, eventname, and eventdate for previous IBS diagnosis.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def columns_starting(df: pd.DataFrame, field: str) -> list[str]:
    prefix = f"{field}-"
    return [column for column in df.columns if column.startswith(prefix)]


def field(df: pd.DataFrame, column: str) -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series(np.nan, index=df.index)


def read_code_list(path: Path) -> set[str]:
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


def has_any_value(df: pd.DataFrame, columns: list[str], values: set[object]) -> pd.Series:
    if not columns:
        return pd.Series(False, index=df.index)
    numeric_values = pd.Series(list(values), dtype="float64")
    numeric_df = df[columns].apply(pd.to_numeric, errors="coerce")
    return numeric_df.isin(numeric_values).any(axis=1)


def has_any_prefix(df: pd.DataFrame, columns: list[str], prefixes: set[str]) -> pd.Series:
    if not columns:
        return pd.Series(False, index=df.index)
    string_df = df[columns].fillna("").astype(str)
    mask = pd.Series(False, index=df.index)
    for prefix in prefixes:
        mask = mask | string_df.apply(lambda col: col.str.startswith(prefix)).any(axis=1)
    return mask


def dhq_hc_exclusion(df: pd.DataFrame) -> pd.Series:
    abdominal_pain = field(df, "21025-0.0").isin(range(2, 7))
    hard_stools = field(df, "21033-0.0").isin(range(-504, -501))
    loose_stools = field(df, "21034-0.0").isin(range(-504, -501))
    gluten_sensitivity = field(df, "21069-0.0").isin(range(-705, -702))
    self_report_ibs = field(df, "21024-0.0").eq(1)
    return abdominal_pain | hard_stools | loose_stools | gluten_sensitivity | self_report_ibs


def between_exclusive(series: pd.Series, lower: float, upper: float) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return (numeric > lower) & (numeric < upper)


def rome_iii_ibs(df: pd.DataFrame) -> pd.Series:
    pain_frequency = pd.to_numeric(field(df, "21025-0.0"), errors="coerce") > 2
    pain_less_menstruation = (pd.to_numeric(field(df, "21026-0.0"), errors="coerce") < 1) | field(
        df, "31-0.0"
    ).eq(1)
    pain_related_defecation = field(df, "21027-0.0").eq(1)
    c1 = between_exclusive(field(df, "21028-0.0"), -600, -500)
    c2 = between_exclusive(field(df, "21029-0.0"), -600, -500) | between_exclusive(
        field(df, "21030-0.0"), -600, -500
    )
    c3 = between_exclusive(field(df, "21031-0.0"), -600, -500) | between_exclusive(
        field(df, "21032-0.0"), -600, -500
    )
    associated_symptoms = (c1 & c2) | (c1 & c3) | (c2 & c3)
    return pain_frequency & pain_less_menstruation & pain_related_defecation & associated_symptoms


def classify_ibs_subtype(df: pd.DataFrame) -> pd.Series:
    mapping = {-500.0: 1, -501.0: 2, -502.0: 3, -503.0: 4, -504.0: 5}
    hard_lumpy = pd.to_numeric(field(df, "21033-0.0"), errors="coerce").map(mapping)
    loose_watery = pd.to_numeric(field(df, "21034-0.0"), errors="coerce").map(mapping)

    is_constipation = (hard_lumpy >= 2) & (loose_watery == 1)
    is_diarrhea = (hard_lumpy == 1) & (loose_watery >= 2)
    is_mixed = (hard_lumpy >= 2) & (loose_watery >= 2)
    return pd.Series(
        np.select(
            [is_constipation, is_diarrhea, is_mixed],
            ["IBS-C", "IBS-D", "IBS-M"],
            default="IBS-U",
        ),
        index=df.index,
    )


def add_questionnaire_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if set(IBS_SSS_PROMPT_COLUMNS + IBS_SSS_SCORE_COLUMNS + IBS_SSS_OTHER_COLUMNS).issubset(out.columns):
        out["IBS-SSS"] = np.nan
        complete = out[IBS_SSS_PROMPT_COLUMNS + IBS_SSS_OTHER_COLUMNS].notna().all(axis=1)
        out.loc[complete, "IBS-SSS"] = 0.0
        for prompt_col, score_col in zip(IBS_SSS_PROMPT_COLUMNS, IBS_SSS_SCORE_COLUMNS):
            add_score = np.where(
                (out[prompt_col] == 1) & (pd.to_numeric(out[score_col], errors="coerce") >= 0),
                pd.to_numeric(out[score_col], errors="coerce"),
                0,
            )
            add_score = pd.Series(add_score, index=out.index)
            out.loc[complete, "IBS-SSS"] = out.loc[complete, "IBS-SSS"] + add_score.loc[complete]
        out.loc[complete, "IBS-SSS"] = out.loc[complete, "IBS-SSS"] + out.loc[
            complete, IBS_SSS_OTHER_COLUMNS
        ].clip(lower=0).sum(axis=1)

    if set(PHQ12_COLUMNS).issubset(out.columns):
        response_to_score = {-600: 0, -601: 1, -602: 2}
        phq12 = out[PHQ12_COLUMNS].apply(lambda col: col.map(response_to_score).fillna(0)).sum(axis=1)
        phq12[out[PHQ12_COLUMNS[:-2]].isna().any(axis=1)] = np.nan
        out["PHQ-12"] = phq12

    if set(GAD7_COLUMNS).issubset(out.columns):
        response_to_score = {1: 0, 2: 1, 3: 2, 4: 3}
        gad7 = out[GAD7_COLUMNS].apply(lambda col: col.map(response_to_score).fillna(0)).sum(axis=1)
        gad7[out[GAD7_COLUMNS].isna().any(axis=1)] = np.nan
        out["GAD-7"] = gad7

    if set(PHQ9_COLUMNS).issubset(out.columns):
        response_to_score = {1: 0, 2: 1, 3: 2, 4: 3}
        phq9 = out[PHQ9_COLUMNS].apply(lambda col: col.map(response_to_score).fillna(0)).sum(axis=1)
        phq9[out[PHQ9_COLUMNS].isna().any(axis=1)] = np.nan
        out["PHQ-9"] = phq9

    return out


def clean_date(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce")
    return dt.mask(dt.dt.year < 1934)


def add_covariates(df: pd.DataFrame, alcohol_csv: Path | None, disease_history_csv: Path | None) -> pd.DataFrame:
    out = df.rename(columns=RENAME_COLUMNS).copy()
    if "sex" in out.columns:
        out["sex"] = out["sex"].map({0: "Female", 1: "Male", "0": "Female", "1": "Male"}).fillna(out["sex"])
    for old, new in [
        ("date_visit_assessment_center", "mri_date"),
        ("DHQ_finish_time", "DHQ_date"),
        ("mental_health_finish_time", "mental_health_date"),
    ]:
        if old in out.columns:
            out[new] = clean_date(out[old])

    edu_cols = [column for column in out.columns if column.startswith("6138-2.")]
    if edu_cols:
        edu_year_map = {1: 16, 2: 12, 3: 10, 4: 10, 5: 13, 6: 14, -7: 6, -3: np.nan}
        edu = out[edu_cols].apply(pd.to_numeric, errors="coerce").replace(edu_year_map)
        out["education_years"] = edu.max(axis=1)

    med_cols = [f"6154-2.{index}" for index in range(1, 6) if f"6154-2.{index}" in out.columns]
    if med_cols:
        med = out[med_cols]
        out["pain_medication"] = med.apply(
            lambda row: np.nan if row.isna().all() else any(value in [1, 2, 3] for value in row.dropna()),
            axis=1,
        )
        out["gastro_medication"] = med.apply(
            lambda row: np.nan if row.isna().all() else any(value in [4, 5, 6] for value in row.dropna()),
            axis=1,
        )

    if "smoking_status" in out.columns:
        out["smoking_status"] = out["smoking_status"].replace(-3, np.nan)

    if alcohol_csv is not None:
        alcohol = pd.read_csv(alcohol_csv, usecols=["eid", "Alc"])
        out = out.merge(alcohol, on="eid", how="left")

    out = add_questionnaire_scores(out)
    out["prev_IBS_diag"] = np.nan
    if disease_history_csv is not None and "mri_date" in out.columns:
        disease_history = pd.read_csv(disease_history_csv)
        needed = {"eid", "eventname", "eventdate"}
        if needed.issubset(disease_history.columns):
            assessment_date = out.set_index("eid")["mri_date"]
            disease_history["assessment_date"] = disease_history["eid"].map(assessment_date)
            disease_history["eventdate"] = pd.to_datetime(disease_history["eventdate"], errors="coerce")
            before = disease_history["assessment_date"] > disease_history["eventdate"]
            prev = (
                disease_history.loc[before & disease_history["eventname"].astype(str).str.startswith("K58"), ["eid"]]
                .assign(prev_IBS_diag=True)
                .drop_duplicates("eid")
            )
            out = out.drop(columns=["prev_IBS_diag"]).merge(prev, on="eid", how="left")

    return out


def propensity_match(hc: pd.DataFrame, ibs: pd.DataFrame, random_state: int) -> pd.DataFrame:
    match_cols = ["age", "sex", "site"]
    hc_pool = hc.dropna(subset=match_cols).copy()
    ibs_pool = ibs.dropna(subset=match_cols).copy()
    hc_pool["case"] = 0
    ibs_pool["case"] = 1
    combined = pd.concat([hc_pool[["eid", "case"] + match_cols], ibs_pool[["eid", "case"] + match_cols]])
    design = pd.get_dummies(combined[match_cols], columns=["sex", "site"], drop_first=False)
    x = StandardScaler().fit_transform(design)
    y = combined["case"].to_numpy()
    model = LogisticRegression(max_iter=1000, random_state=random_state)
    model.fit(x, y)
    combined["propensity"] = model.predict_proba(x)[:, 1]
    hc_scores = combined.loc[combined["case"] == 0, ["eid", "propensity"]].copy()
    ibs_scores = combined.loc[combined["case"] == 1, ["eid", "propensity"]].copy()

    neighbors = NearestNeighbors(n_neighbors=len(hc_scores), metric="euclidean")
    neighbors.fit(hc_scores[["propensity"]])
    distances, indices = neighbors.kneighbors(ibs_scores[["propensity"]])

    used_hc: set[object] = set()
    pairs = []
    for ibs_row_position, ibs_row in ibs_scores.reset_index(drop=True).iterrows():
        for distance, hc_idx in zip(distances[ibs_row_position], indices[ibs_row_position]):
            hc_eid = hc_scores.iloc[hc_idx]["eid"]
            if hc_eid in used_hc:
                continue
            used_hc.add(hc_eid)
            pairs.append(
                {
                    "eid": ibs_row["eid"],
                    "matched_ID": hc_eid,
                    "propensity_distance": float(distance),
                    "IBS_propensity": float(ibs_row["propensity"]),
                    "HC_propensity": float(hc_scores.iloc[hc_idx]["propensity"]),
                }
            )
            break
    return pd.DataFrame(pairs)


def define_cohorts(raw: pd.DataFrame, docs_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    icd_cols = columns_starting(raw, "41270")
    opcs_cols = columns_starting(raw, "41272")
    illness_cols = columns_starting(raw, "20002")
    icd_hc = read_code_list(docs_dir / "icd_10_HC.txt")
    icd_ibs = read_code_list(docs_dir / "icd_10_IBS.txt")
    opcs_hc = read_code_list(docs_dir / "OPCS4_HC.txt")
    opcs_ibs = read_code_list(docs_dir / "OPCS4_IBS.txt")

    has_mri = field(raw, "21003-2.0").notna()
    hc_excluded = (
        has_any_prefix(raw, icd_cols, icd_hc)
        | has_any_prefix(raw, opcs_cols, opcs_hc)
        | dhq_hc_exclusion(raw)
        | has_any_value(raw, illness_cols, HC_SELF_REPORT_EXCLUSIONS)
    )
    hc = raw.loc[has_mri & ~hc_excluded].copy()

    ibs_excluded = (
        field(raw, "21069-0.0").isin(range(-705, -702))
        | has_any_prefix(raw, icd_cols, icd_ibs)
        | has_any_prefix(raw, opcs_cols, opcs_ibs)
        | has_any_value(raw, illness_cols, IBS_SELF_REPORT_EXCLUSIONS)
    )
    rome = rome_iii_ibs(raw)
    ibs = raw.loc[has_mri & ~ibs_excluded & rome].copy()
    ibs["IBS_subtype"] = classify_ibs_subtype(ibs)

    flow = pd.DataFrame(
        [
            {"step": "raw_rows", "n": len(raw)},
            {"step": "mri_available", "n": int(has_mri.sum())},
            {"step": "hc_after_exclusion", "n": len(hc)},
            {"step": "ibs_rome_iii_after_exclusion", "n": len(ibs)},
        ]
    )
    return hc, ibs, flow


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.ukb_csv, low_memory=False)
    if "eid" not in raw.columns:
        raise ValueError("Input CSV must contain an eid column.")

    hc_raw, ibs_raw, flow = define_cohorts(raw, args.docs_dir.resolve())
    hc_cov = add_covariates(hc_raw, args.alcohol_csv, args.disease_history_csv)
    ibs_cov = add_covariates(ibs_raw, args.alcohol_csv, args.disease_history_csv)

    pairs = propensity_match(hc_cov, ibs_cov, random_state=args.random_state)
    matched_hc = hc_cov[hc_cov["eid"].isin(pairs["matched_ID"])].copy()
    matched_ibs = ibs_cov[ibs_cov["eid"].isin(pairs["eid"])].copy()

    matched_hc.to_csv(args.output_dir / "HC_cov_ROME.csv", index=False)
    matched_ibs.to_csv(args.output_dir / "IBS_cov_ROME.csv", index=False)
    pairs.to_csv(args.output_dir / "IBS_matched_pairs.csv", index=False)
    flow = pd.concat(
        [
            flow,
            pd.DataFrame(
                [
                    {"step": "matched_hc", "n": len(matched_hc)},
                    {"step": "matched_ibs", "n": len(matched_ibs)},
                ]
            ),
        ],
        ignore_index=True,
    )
    flow.to_csv(args.output_dir / "cohort_flow_summary.csv", index=False)
    print(flow.to_string(index=False))


if __name__ == "__main__":
    main()
