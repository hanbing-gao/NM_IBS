"""Compare normative-model evaluation metrics across validation, HC, and IBS sets.

The normative-model training step writes one metric table for the held-out
healthy-control reference validation set and one prediction metric table for
each applied group:

    blr_metrics.csv
    blr_metrics_HC.csv
    blr_metrics_IBS.csv

This script compares EV, MSLL, absolute skewness, and kurtosis across those
three metric tables using paired ROI-wise tests, separately by modality and
hemisphere.

Example:
    python scripts/p2b_evaluate_normative_model_metrics.py \
        --model-dir results/normative_models \
        --output-dir results/normative_model_evaluation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
MODALITIES = ["SA", "CT", "CV"]
METRICS = ["EV", "MSLL", "Skew", "Kurtosis"]
HEMISPHERES = ["all", "lh", "rh"]
COMPARISONS = [
    ("validation_vs_HC", "validation", "HC"),
    ("validation_vs_IBS", "validation", "IBS"),
    ("HC_vs_IBS", "HC", "IBS"),
]
ALTERNATIVES = {
    "EV": "greater",
    "MSLL": "less",
    "Skew": "less",
    "Kurtosis": "less",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run paired ROI-wise comparisons of NM evaluation metrics."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=PACKAGE_ROOT / "results" / "normative_models",
        help="Directory containing <modality>_age_45_85/perm_<seed>/ metric files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PACKAGE_ROOT / "results" / "normative_model_evaluation",
    )
    parser.add_argument("--permutation", type=int, default=42)
    parser.add_argument("--hc-label", default="HC")
    parser.add_argument("--patient-label", default="IBS")
    parser.add_argument(
        "--test-method",
        choices=["paired_t", "legacy"],
        default="paired_t",
        help=(
            "paired_t uses scipy.stats.ttest_rel. legacy reproduces the notebook "
            "logic: Levene screen followed by paired t-test or Wilcoxon signed-rank."
        ),
    )
    parser.add_argument(
        "--legacy-prho-filter",
        action="store_true",
        help="Drop ROIs with validation pRho_fdr > 0.05, reproducing the older notebook check.",
    )
    return parser.parse_args()


def metric_file_for(group_label: str, hc_label: str, patient_label: str) -> str:
    if group_label == "validation":
        return "blr_metrics.csv"
    if group_label == "HC":
        return f"blr_metrics_{hc_label}.csv"
    if group_label == "IBS":
        return f"blr_metrics_{patient_label}.csv"
    raise ValueError(f"Unknown group label: {group_label}")


def load_metric_table(metric_dir: Path, group_label: str, hc_label: str, patient_label: str) -> pd.DataFrame:
    path = metric_dir / metric_file_for(group_label, hc_label, patient_label)
    df = pd.read_csv(path)
    required = {"eid", *METRICS}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    out = df.copy()
    out["roi"] = out["eid"].astype(str)
    out["group"] = group_label
    return out


def hemisphere_mask(roi: pd.Series, hemisphere: str) -> pd.Series:
    if hemisphere == "all":
        return pd.Series(True, index=roi.index)
    return roi.astype(str).str.startswith(f"{hemisphere}_")


def paired_test(a: pd.Series, b: pd.Series, alternative: str, method: str) -> dict[str, object]:
    paired = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(paired) < 2:
        return {"test": method, "statistic": np.nan, "p": np.nan, "levene_p": np.nan}

    if method == "legacy":
        levene_p = stats.levene(paired["a"], paired["b"], center="mean").pvalue
        if levene_p > 0.05:
            result = stats.ttest_rel(paired["a"], paired["b"], alternative=alternative)
            return {
                "test": "paired_t",
                "statistic": float(result.statistic),
                "p": float(result.pvalue),
                "levene_p": float(levene_p),
            }
        try:
            result = stats.wilcoxon(paired["a"], paired["b"], alternative=alternative)
            return {
                "test": "wilcoxon_signed_rank",
                "statistic": float(result.statistic),
                "p": float(result.pvalue),
                "levene_p": float(levene_p),
            }
        except ValueError:
            return {"test": "wilcoxon_signed_rank", "statistic": np.nan, "p": np.nan, "levene_p": float(levene_p)}

    result = stats.ttest_rel(paired["a"], paired["b"], alternative=alternative)
    return {"test": "paired_t", "statistic": float(result.statistic), "p": float(result.pvalue), "levene_p": np.nan}


def metric_values_for_group(df: pd.DataFrame, metric: str) -> pd.Series:
    values = pd.to_numeric(df[metric], errors="coerce")
    if metric == "Skew":
        values = values.abs()
    return values


def load_modality_metrics(
    model_dir: Path,
    modality: str,
    permutation: int,
    hc_label: str,
    patient_label: str,
    legacy_prho_filter: bool,
) -> dict[str, pd.DataFrame]:
    metric_dir = model_dir / f"{modality}_age_45_85" / f"perm_{permutation}"
    tables = {
        group: load_metric_table(metric_dir, group, hc_label, patient_label)
        for group in ["validation", "HC", "IBS"]
    }

    if legacy_prho_filter:
        if "pRho_fdr" not in tables["validation"].columns:
            raise ValueError(
                f"{metric_dir / 'blr_metrics.csv'} has no pRho_fdr column required by --legacy-prho-filter."
            )
        keep_roi = set(tables["validation"].loc[tables["validation"]["pRho_fdr"] <= 0.05, "roi"])
        tables = {group: table[table["roi"].isin(keep_roi)].copy() for group, table in tables.items()}

    return tables


def make_long_metric_table(tables_by_modality: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    rows = []
    for modality, tables in tables_by_modality.items():
        for group, table in tables.items():
            for metric in METRICS:
                values = metric_values_for_group(table, metric)
                metric_label = "Absolute Skew" if metric == "Skew" else metric
                for roi, value in zip(table["roi"], values):
                    rows.append(
                        {
                            "modality": modality,
                            "group": group,
                            "roi": roi,
                            "hemisphere": "lh" if str(roi).startswith("lh_") else "rh" if str(roi).startswith("rh_") else "other",
                            "metric": metric_label,
                            "value": value,
                        }
                    )
    return pd.DataFrame(rows)


def run_tests(tables_by_modality: dict[str, dict[str, pd.DataFrame]], test_method: str) -> pd.DataFrame:
    rows = []
    for modality, tables in tables_by_modality.items():
        for hemisphere in HEMISPHERES:
            for metric in METRICS:
                metric_label = "Absolute Skew" if metric == "Skew" else metric
                alternative = ALTERNATIVES[metric]
                value_by_group = {}
                for group, table in tables.items():
                    sub = table.loc[hemisphere_mask(table["roi"], hemisphere), ["roi"]].copy()
                    sub["value"] = metric_values_for_group(table.loc[sub.index], metric).to_numpy()
                    value_by_group[group] = sub.set_index("roi")["value"]

                for comparison, group_a, group_b in COMPARISONS:
                    paired = pd.concat(
                        [value_by_group[group_a], value_by_group[group_b]],
                        axis=1,
                        join="inner",
                    )
                    paired.columns = ["a", "b"]
                    test = paired_test(paired["a"], paired["b"], alternative, test_method)
                    diff = paired.dropna()["a"] - paired.dropna()["b"]
                    rows.append(
                        {
                            "modality": modality,
                            "hemisphere": hemisphere,
                            "metric": metric_label,
                            "comparison": comparison,
                            "group_a": group_a,
                            "group_b": group_b,
                            "alternative": f"{group_a} {alternative} {group_b}",
                            "n_roi": int(len(diff)),
                            "mean_group_a": float(paired["a"].mean()),
                            "mean_group_b": float(paired["b"].mean()),
                            "mean_difference_a_minus_b": float(diff.mean()) if len(diff) else np.nan,
                            "median_difference_a_minus_b": float(diff.median()) if len(diff) else np.nan,
                            **test,
                        }
                    )

    out = pd.DataFrame(rows)
    out["p_fdr_bh_within_metric"] = np.nan
    for _, idx in out.groupby(["modality", "metric"]).groups.items():
        valid_idx = out.loc[idx].index[out.loc[idx, "p"].notna()]
        if len(valid_idx):
            out.loc[valid_idx, "p_fdr_bh_within_metric"] = multipletests(
                out.loc[valid_idx, "p"], method="fdr_bh"
            )[1]
    return out


def make_summary(long_values: pd.DataFrame) -> pd.DataFrame:
    return (
        long_values.groupby(["modality", "hemisphere", "metric", "group"], as_index=False)
        .agg(
            n_roi=("value", "count"),
            mean=("value", "mean"),
            sd=("value", "std"),
            median=("value", "median"),
            q1=("value", lambda x: x.quantile(0.25)),
            q3=("value", lambda x: x.quantile(0.75)),
        )
        .sort_values(["modality", "hemisphere", "metric", "group"])
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.model_dir.resolve()

    tables_by_modality = {
        modality: load_modality_metrics(
            model_dir,
            modality,
            args.permutation,
            args.hc_label,
            args.patient_label,
            args.legacy_prho_filter,
        )
        for modality in MODALITIES
    }

    long_values = make_long_metric_table(tables_by_modality)
    summary = make_summary(long_values)
    tests = run_tests(tables_by_modality, args.test_method)

    values_path = args.output_dir / "normative_model_metric_values_long.csv"
    summary_path = args.output_dir / "normative_model_metric_summary.csv"
    tests_path = args.output_dir / "normative_model_metric_paired_tests.csv"
    long_values.to_csv(values_path, index=False)
    summary.to_csv(summary_path, index=False)
    tests.to_csv(tests_path, index=False)

    manifest = {
        "model_dir": str(model_dir),
        "permutation": args.permutation,
        "test_method": args.test_method,
        "legacy_prho_filter": args.legacy_prho_filter,
        "metrics": ["EV", "MSLL", "Absolute Skew", "Kurtosis"],
        "outputs": [str(values_path), str(summary_path), str(tests_path)],
    }
    (args.output_dir / "normative_model_metric_evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(tests.to_string(index=False))


if __name__ == "__main__":
    main()
