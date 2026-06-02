"""Train BLR normative models and predict HC/IBS deviation scores.

This is a command-line wrapper around utils_norm.nm_training.process_and_train_model.
It uses only user-supplied UKB-derived CSV files; no participant data are
included in this repository.

Expected columns:
    eid, age, sex, site, and raw cortical IDP columns named like <field_id>-2.0

Example:
    python scripts/p2_train_normative_models.py \
        --normative-hc-csv data/processed/HC_normative_reference.csv \
        --matched-hc-csv data/processed/HC_cov_ROME.csv \
        --matched-ibs-csv data/processed/IBS_cov_ROME.csv \
        --output-folder results/normative_models
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train normative models and apply them to matched HC/IBS participants."
    )
    parser.add_argument(
        "--normative-hc-csv",
        type=Path,
        required=True,
        help="Healthy-control reference sample used to train the normative model.",
    )
    parser.add_argument(
        "--matched-hc-csv",
        type=Path,
        required=True,
        help="Matched HC sample with covariates and raw cortical IDP columns.",
    )
    parser.add_argument(
        "--matched-ibs-csv",
        type=Path,
        required=True,
        help="Matched IBS sample with covariates and raw cortical IDP columns.",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default="results/normative_models",
        help="Output folder relative to the package root, or an absolute path.",
    )
    parser.add_argument("--patient-name", default="IBS", help="Label used in prediction filenames.")
    parser.add_argument("--permutation", type=int, default=42, help="Random seed used for HC train/test split.")
    parser.add_argument(
        "--covariates",
        nargs="+",
        default=["age", "sex", "site"],
        help="Normative-model covariates. site is used for site indicators and removed from the spline covariate list.",
    )
    parser.add_argument(
        "--roi-list",
        nargs="*",
        default=None,
        help="Optional ROI names to model. Omit to use all ROIs in the packaged mapping files.",
    )
    return parser.parse_args()


def normalize_sex_for_nm(series: pd.Series) -> pd.Series:
    sex = series.copy()
    if pd.api.types.is_numeric_dtype(sex):
        return pd.to_numeric(sex, errors="coerce")
    normalized = sex.astype(str).str.strip().str.lower()
    return normalized.map(
        {
            "0": 0,
            "female": 0,
            "f": 0,
            "woman": 0,
            "1": 1,
            "male": 1,
            "m": 1,
            "man": 1,
        }
    ).astype(float)


def read_input(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(
        columns={
            "21003-2.0": "age",
            "31-0.0": "sex",
            "54-2.0": "site",
        }
    )
    required = {"eid", "age", "sex", "site"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"{path} is missing required columns: {sorted(missing)}. "
            "Use columns age/sex/site or UKB fields 21003-2.0/31-0.0/54-2.0."
        )
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["sex"] = normalize_sex_for_nm(df["sex"])
    if df[["age", "sex", "site"]].isna().any().any():
        missing_counts = df[["age", "sex", "site"]].isna().sum().to_dict()
        raise ValueError(f"{path} has missing/non-parsable NM covariates: {missing_counts}")
    if not np.isfinite(df["age"]).all():
        raise ValueError(f"{path} contains non-finite age values.")
    return df


def main() -> None:
    args = parse_args()
    try:
        from utils_norm.nm_training import process_and_train_model
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Normative-model training requires pcntoolkit and related dependencies. "
            "Install them with `pip install -r requirements.txt` before running p2."
        ) from exc

    hc_normative = read_input(args.normative_hc_csv)
    hc_matched = read_input(args.matched_hc_csv)
    ibs_matched = read_input(args.matched_ibs_csv)

    output_folder = Path(args.output_folder)
    if output_folder.is_absolute():
        try:
            output_folder = output_folder.relative_to(PACKAGE_ROOT)
        except ValueError as exc:
            raise ValueError(
                "--output-folder must be relative to the package root, or an absolute path under it."
            ) from exc

    process_and_train_model(
        root_dir=str(PACKAGE_ROOT),
        out_folder=str(output_folder),
        cov=args.covariates,
        perm=args.permutation,
        HC_nm_data=hc_normative,
        HC_data=hc_matched,
        pat_data=ibs_matched,
        pat_name=args.patient_name,
        ROI_list=args.roi_list,
        split_data=True,
    )
    print(f"Normative-model outputs written under: {PACKAGE_ROOT / output_folder}")


if __name__ == "__main__":
    main()
