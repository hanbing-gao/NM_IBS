# Morphometry variations in irritable bowel syndrome revealed by large-scale neuroanatomical normative modeling 

This repository contains the full pipeline used to preprocess, model, analyze, and interpret MRI-derived brain structure data using a normative modeling framework. The project combines machine learning, mixed-effects modeling, and statistical correlation analysis to investigate neuroanatomical deviations in IBS populations.

---

## 📁 Project Structure

```bash
.
├── 1_document/                        # Documentation and region map CSVs (e.g., ROI_IBS.csv)
├── 2_code/                            # All analysis scripts and notebooks
│   ├── p1_data_include.ipynb          # Data loading and filtering
│   ├── p2_NM_model_training_evaluation.ipynb  # Normative model training and evaluation
│   ├── p3_LMM_analysis.ipynb          # Linear mixed-effects model analysis
│   ├── p4_correlation_ana.ipynb       # Correlation with behavioral/clinical phenotypes
│   ├── requirements.txt               # Python dependencies
│   └── utils_norm/                    # Modular scripts for modeling, stats, and visualization
│       ├── utils_analyses.py          # Functions for statistical analyses
│       ├── utils_visual.py            # Functions for visualization
│       ├── nm_training.py             # Funtions for normative model training
│       └── myblr.py                   # Funtions for warped BLR algorithms adapted based on PCNtoolkit
├── 3_rerun_whole_work/                # Main data and results directory (will be gernerated when run the whole project)
│   ├── 1_data_cleaned/                # Cleaned/preprocessed data (e.g., HC_MRI_age.csv)
│   ├── 2_models_sMRI/                 # Normative model outputs (e.g., deviation scores)
│   ├── 3_LMM/                         # LMM analysis results
│   └── 4_eq_test/                     # Equivalence test results
└── README.md                          # Project overview and instructionsrequirements.txt                   
```

---

## 🔄 Workflow Summary

### `p1_data_include.ipynb`
Data inclusion and preprocessing: loads and filters MRI phenotype data for healthy controls and patients.

### `p2_NM_model_training_evaluation.ipynb`
Normative modeling: trains Bayesian warped linear models on healthy controls and evaluates model performance.

### `p3_LMM_analysis.ipynb`
Statistical testing: runs linear mixed-effects models to assess group differences across brain regions.

### `p4_correlation_ana.ipynb`
Phenotype correlation analysis: correlates deviations from the normative model with clinical or behavioral variables.

---

## ⚙️ Requirements

Install required Python packages:

```bash
matplotlib==3.7.0
nibabel==5.0.1
nilearn==0.10.4
numpy==1.22.4
pandas==1.5.3
pcntoolkit==0.26
scikit-learn==1.0.2
scipy==1.10.0
statsmodels==0.14.0
```

Guidance of PCNtoolkit could be found at [https://github.com/amarquand/PCNtoolkit](https://github.com/amarquand/PCNtoolkit).

---

## 📊 Outputs

Each step of the pipeline generates outputs such as:
- Filtered subject data
- Trained model parameters
- Predictive deviations
- Mixed model statistics (e.g., group-by-region effects)
- Correlation results with clinical data
- Brain maps of significant regions

---

## 📘 Reference

> Rutherford, S., Kia, S.M., Wolfers, T. et al. The normative modeling framework for computational psychiatry. Nat Protoc 17, 1711–1734 (2022). https://doi.org/10.1038/s41596-022-00696-5

---

## 🧠 Acknowledgments

- Core modeling algorithms adapted from [PCNtoolkit](https://github.com/amarquand/PCNtoolkit)
- MRI IDPs assumed to be extracted from FreeSurfer and UK Biobank pipelines

---

## Data availability

This study has been conducted using UK Biobank Resource under Application 71300. All raw data are available from the UKB (www.ukbiobank.ac.uk). The raw and processed UKB MRI data are protected and are not openly available due to data privacy laws. Assess can be obtained by applying for access (www.ukbiobank.ac.uk/enable-your-research/apply-for-access).
