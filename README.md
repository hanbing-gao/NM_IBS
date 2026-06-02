# IBS Normative-Model Revision Code

Open-source analysis code for the IBS cortical morphometry revision. This folder
contains code, field mappings, ROI lists, and diagnosis/procedure exclusion code
lists only. It does not contain UK Biobank participant-level data, derived
covariate tables, deviation-score matrices, or analysis results.

## Safety

UK Biobank is not a public dataset. Keep all local participant-level exports in
`data/`, which is ignored by git. Before releasing or committing, run:

```bash
rg -n "<your-local-user-name>|<your-local-drive>|<your-institutional-sync-folder>" open_source_revision_code
```

The `eid` term will appear in code and documentation because it is the expected
identifier column, but participant rows should not be present.

## Setup

```bash
cd open_source_revision_code
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt
```

On macOS/Linux, activate with `source .venv/bin/activate`.

## Recommended Workflow

1. Define cohorts, covariates, Rome III IBS status, IBS subtypes, and matched HC/IBS samples:

```bash
python scripts/p1_define_cohorts_and_covariates.py \
  --ukb-csv data/raw_ukb_exports/ukb_fields.csv \
  --output-dir data/processed
```

2. Train normative models and predict matched HC/IBS deviation scores:

```bash
python scripts/p2_train_normative_models.py \
  --normative-hc-csv data/processed/HC_normative_reference.csv \
  --matched-hc-csv data/processed/HC_cov_ROME.csv \
  --matched-ibs-csv data/processed/IBS_cov_ROME.csv \
  --output-folder results/normative_models
```

3. Evaluate normative-model metrics across the HC-reference validation, matched
HC, and IBS samples:

```bash
python scripts/p2b_evaluate_normative_model_metrics.py \
  --model-dir results/normative_models \
  --output-dir results/normative_model_evaluation
```

4. Export all predicted deviation scores. This step does not remove ROIs with
model-evaluation `p > 0.05`:

```bash
python scripts/p3_export_all_predicted_deviation_scores.py \
  --model-dir results/normative_models \
  --processed-dir data/processed \
  --output-dir data/processed
```

5. Run primary HC vs IBS OLS analyses:

```bash
python scripts/p4_primary_ols.py \
  --input-dir data/processed \
  --output-dir results/primary_OLS_all60
```

6. Run dual-SESOI equivalence tests:

```bash
python scripts/p5_equivalence_dual_sesoi.py \
  --input-dir data/processed \
  --output-dir results/equivalence_dual_SESOI
```

7. Run IBS subtype sensitivity analyses:

```bash
python scripts/p6_subtype_sensitivity.py \
  --input-dir data/processed \
  --output-dir results/subtype_sensitivity
```

8. Run questionnaire association analyses:

```bash
python scripts/p7_association.py \
  --input-dir data/processed \
  --output-dir results/association_all60
```

9. Run raw-IDP vs NM-deviation sensitivity analysis:

```bash
python scripts/p8_raw_vs_nm_sensitivity.py \
  --processed-dir data/processed \
  --nm-primary-dir results/primary_OLS_all60 \
  --output-dir results/raw_vs_NM_sensitivity
```

10. Create DHQ-MRI timing distribution figure:

```bash
python scripts/p9_dhq_mri_timing_distribution.py \
  --input-dir data/processed \
  --output-dir results/DHQ_MRI_timing_distribution
```

## Included Metadata

- `docs/ROI_IBS.csv`: canonical 60-ROI order
- `docs/aseg_2009_*_formatted.csv`: UKB field to ROI mappings for CT, SA, and CV
- `docs/icd_10_*.txt`: ICD-10 exclusion code lists
- `docs/OPCS4_*.txt`: OPCS4 exclusion code lists

See `docs/DATA_REQUIREMENTS.md` and `docs/METHOD_OVERVIEW.md` for input schemas
and analysis definitions.
