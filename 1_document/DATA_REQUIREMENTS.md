# Data Requirements

This repository does not include UK Biobank participant-level data. Users must
create local input files from their own approved UKB access and place them under
`data/`, which is ignored by git.

## Minimum raw UKB export for cohort definition

`scripts/p1_define_cohorts_and_covariates.py` expects one CSV with an `eid`
column and the UKB field-instance columns used for exclusion, IBS definition,
matching, and covariate derivation.

Core columns:

- `31-0.0`: sex
- `54-2.0`: imaging assessment centre / scanner site
- `21003-2.0`: age at imaging
- `21001-2.0`: BMI
- `48-2.0`: waist circumference
- `21862-2.0`: MRI assessment date
- `21023-0.0`: DHQ completion time
- `20400-0.0`: mental-health questionnaire completion time
- `21000-2.0`: ethnic background
- `20116-2.0`: smoking status
- `738-2.0`: average household income
- `6138-2.0` to `6138-2.5`: educational qualifications
- `6154-2.0` to `6154-2.5`: medication categories

IBS/DHQ fields:

- `21024-0.0` to `21061-0.0`
- `21069-0.0`

Diagnosis/procedure/self-report fields:

- All available `41270-*` ICD-10 diagnosis columns
- All available `41272-*` OPCS4 procedure columns
- All available `20002-*` self-reported non-cancer illness columns

Optional local files:

- `--alcohol-csv`: CSV with `eid` and `Alc`
- `--disease-history-csv`: CSV with `eid`, `eventname`, and `eventdate`

## Normative-model inputs

`scripts/p2_train_normative_models.py` needs three local CSVs:

- `HC_normative_reference.csv`: healthy-control reference sample for model training
- `HC_cov_ROME.csv`: matched HC sample with covariates and raw IDPs
- `IBS_cov_ROME.csv`: matched IBS sample with covariates and raw IDPs

Each file must contain:

- `eid`
- `age`
- `sex`
- `site`
- Raw cortical IDP columns named `<field_id>-2.0`

The required field IDs are defined in:

- `docs/aseg_2009_CT_formatted.csv`
- `docs/aseg_2009_SA_formatted.csv`
- `docs/aseg_2009_CV_formatted.csv`

If the raw UKB export supplied to `p1_define_cohorts_and_covariates.py` already
contains these cortical IDP columns, the matched `HC_cov_ROME.csv` and
`IBS_cov_ROME.csv` outputs can be passed directly to `p2_train_normative_models.py`
and `p8_raw_vs_nm_sensitivity.py`.

## Processed files consumed by downstream scripts

Downstream scripts expect these local files in `data/processed/`:

- `HC_cov_ROME.csv`
- `IBS_cov_ROME.csv`
- `HC_deviation_scores_by_brain_idp_all_predicted.csv`
- `IBS_deviation_scores_by_brain_idp_all_predicted.csv`

Deviation-score columns should be named `CT__<ROI>`, `SA__<ROI>`, or `CV__<ROI>`,
with ROI names matching `docs/ROI_IBS.csv`.
