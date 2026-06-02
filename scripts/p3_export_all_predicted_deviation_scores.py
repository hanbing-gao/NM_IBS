"""Export all predicted normative-model deviation scores by participant.

The historical analysis helper could zero or omit IDPs when model-evaluation
p-values did not pass a threshold. This revision export intentionally ignores
those diagnostic files and reads every available Z_predict file directly.

Example:
    python scripts/p3_export_all_predicted_deviation_scores.py \
        --model-dir results/normative_models \
        --processed-dir data/processed \
        --output-dir data/processed
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
MODALITIES = ["CT", "SA", "CV"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create HC and IBS deviation-score matrices from all predicted Z scores."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=PACKAGE_ROOT / "results" / "normative_models",
        help="Directory containing <modality>_age_45_85/perm_<seed>/<ROI>/Z_predict_*.txt.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PACKAGE_ROOT / "data" / "processed",
        help="Directory containing HC_cov_ROME.csv and IBS_cov_ROME.csv in prediction order.",
    )
    parser.add_argument("--docs-dir", type=Path, default=PACKAGE_ROOT / "docs")
    parser.add_argument("--output-dir", type=Path, default=PACKAGE_ROOT / "data" / "processed")
    parser.add_argument("--permutation", type=int, default=42)
    parser.add_argument("--hc-label", default="HC")
    parser.add_argument("--patient-label", default="IBS")
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Fill missing Z files with NaN columns instead of raising an error.",
    )
    return parser.parse_args()


def load_roi_order(docs_dir: Path) -> list[str]:
    roi_path = docs_dir / "ROI_IBS.csv"
    roi = pd.read_csv(roi_path)["ROI"].tolist()
    if len(roi) != 60:
        raise ValueError(f"Expected 60 ROIs in {roi_path}, found {len(roi)}.")
    return roi


def load_z(path: Path, expected_n: int, allow_missing: bool) -> np.ndarray:
    if not path.exists():
        if allow_missing:
            return np.full(expected_n, np.nan)
        raise FileNotFoundError(f"Missing predicted deviation file: {path}")
    values = np.atleast_1d(np.loadtxt(path)).astype(float)
    if len(values) != expected_n:
        raise ValueError(f"{path} has {len(values)} rows, expected {expected_n}.")
    return values


def export_group(
    model_dir: Path,
    cov_path: Path,
    roi_order: list[str],
    group_label: str,
    permutation: int,
    allow_missing: bool,
) -> pd.DataFrame:
    cov = pd.read_csv(cov_path, usecols=["eid"])
    out = pd.DataFrame({"eid": cov["eid"].to_numpy()})
    for modality in MODALITIES:
        modality_dir = model_dir / f"{modality}_age_45_85" / f"perm_{permutation}"
        for roi in roi_order:
            z_path = modality_dir / roi / f"Z_predict_{group_label}.txt"
            out[f"{modality}__{roi}"] = load_z(z_path, len(cov), allow_missing)
    return out


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    roi_order = load_roi_order(args.docs_dir.resolve())
    model_dir = args.model_dir.resolve()
    processed_dir = args.processed_dir.resolve()

    hc = export_group(
        model_dir,
        processed_dir / "HC_cov_ROME.csv",
        roi_order,
        args.hc_label,
        args.permutation,
        args.allow_missing,
    )
    ibs = export_group(
        model_dir,
        processed_dir / "IBS_cov_ROME.csv",
        roi_order,
        args.patient_label,
        args.permutation,
        args.allow_missing,
    )

    hc_path = args.output_dir / "HC_deviation_scores_by_brain_idp_all_predicted.csv"
    ibs_path = args.output_dir / "IBS_deviation_scores_by_brain_idp_all_predicted.csv"
    hc.to_csv(hc_path, index=False)
    ibs.to_csv(ibs_path, index=False)
    pd.DataFrame(
        [
            {"group": "HC", "path": str(hc_path), "n": len(hc), "n_idp": hc.shape[1] - 1},
            {"group": "IBS", "path": str(ibs_path), "n": len(ibs), "n_idp": ibs.shape[1] - 1},
        ]
    ).to_csv(args.output_dir / "deviation_score_export_manifest.csv", index=False)
    print(f"Saved {hc_path}")
    print(f"Saved {ibs_path}")


if __name__ == "__main__":
    main()
