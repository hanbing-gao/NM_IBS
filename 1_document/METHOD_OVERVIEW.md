# Method Overview

## Cohort Definition

Healthy controls were participants with MRI data who did not meet exclusion
criteria based on ICD-10 diagnoses, OPCS4 procedures, DHQ gastrointestinal
responses, IBS self-report, or selected self-reported illness codes.

IBS cases were defined using DHQ-derived Rome III criteria.

IBS subtypes were derived from DHQ bowel-habit items:

- IBS-C: hard/lumpy stools at least sometimes and loose/watery stools never or rarely
- IBS-D: loose/watery stools at least sometimes and hard/lumpy stools never or rarely
- IBS-M: both hard/lumpy and loose/watery stools at least sometimes
- IBS-U: all remaining IBS cases

## Normative Modeling

Bayesian linear regression normative models were trained in healthy controls
using age, sex, and scanner site, with an age spline over 45 to 85 years. Models
were fitted separately for CT, SA, and CV IDPs. Predicted deviation scores were
exported from all `Z_predict_*` files. No ROI is removed or zeroed because of a
model-evaluation p-value threshold in the revision pipeline.

## Normative-Model Evaluation

Normative-model evaluation metrics are compared across the held-out
HC-reference validation set, the matched HC sample, and the IBS sample. For each
modality and hemisphere, paired ROI-wise tests compare EV, MSLL, absolute
skewness, and kurtosis for:

- HC-reference validation vs matched HC
- HC-reference validation vs IBS
- matched HC vs IBS

One-sided paired tests use the expected direction for model quality and
calibration: higher EV, lower MSLL, lower absolute skewness, and lower kurtosis.
The public script retains all ROIs by default, with an optional legacy flag to
reproduce the older `pRho_fdr <= 0.05` filtering check.

## Primary Analysis

For each deviation-score IDP:

```text
deviation ~ Group * C(sex, Sum)
```

`Group` is coded as HC vs IBS, with HC as the reference group. FDR correction is
performed within each modality.

## Equivalence Tests

Two one-sided equivalence tests are run for the primary group effect and
group-by-sex interaction using the original SESOI thresholds and the additional
reviewer-facing SESOI of 0.15.

## Subtype Sensitivity Analysis

Subtype sensitivity analyses include:

- five-level group omnibus model: HC, IBS-C, IBS-D, IBS-M, IBS-U
- five-level group-by-sex interaction
- nested model comparison of separate IBS subtype effects against a common IBS effect
- nested model comparison of subtype-by-sex interactions against a common IBS-by-sex interaction

Models include sex as the only covariate. FDR correction is performed separately
within CT, SA, and CV for each analysis.

## Association Analyses

Questionnaire association analyses are rerun with all 60 ROIs per modality for:

- IBS-SSS
- PHQ-12
- PHQ-9
- GAD-7

The all-60 revision removes the historical omission caused by the old
model-performance filtering rule.
