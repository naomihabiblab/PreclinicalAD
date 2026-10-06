# Concurrent cognitive, behavioral and systemic physiological features mark preclinical Alzheimer's disease

This repository contains the analysis code for Tsoran & Rom et al., *Concurrent cognitive, behavioral and systemic physiological features mark preclinical Alzheimer's disease*.

## Abstract

Alzheimer's disease (AD) pathology accumulates years before overt cognitive impairment, yet its multidomain phenotype remains poorly defined. To limit confounding by uncontrolled comorbidities, we assembled a cohort of 644 older adults self-reporting intact cognition, free of systemic or neurological diseases and managed medically for risk factors. Plasma p-tau217 was elevated in 20%, and 93 of these were cognitively unimpaired. This biomarker-defined, risk factor-controlled preclinical AD subgroup showed subtle, concurrent cognitive and behavioral differences, accompanied by modest differences in physiological measures within clinical reference ranges, and alongside higher biomarkers of astrogliosis (GFAP) and neurodegeneration (NfL). Reported modifiable risk factor histories were not associated with p-tau217 when effectively managed. A proof-of-concept AI classifier integrating routine multidomain clinical data improved identification of p-tau217-high individuals relative to cognitive screening alone. These findings indicate preclinical AD is not phenotypically silent at the group level and support multidomain-based screening for biomarker testing and early detection.

## Repository structure

- `biomarkers/` — exploratory analyses of p-tau217, GFAP, NfL, and APOE, including biomarker thresholds and feature distributions.
- `clinical_data/` — cohort summaries, cognitive and behavioral analyses, APOE analyses, and reusable statistical-analysis scripts for biomarker/trait associations.
- `biomarker_prediction/` — preprocessing, model training, hyperparameter optimization, bootstrap evaluation, ensemble methods, visualization, and explainability for p-tau217 classification.

## Environment

The analyses were run in three lab conda environments, `BCG`, `stats`, and `tabpfn2`. Those environments use Python 3.12, 3.8, and 3.10, so they cannot be installed as one interpreter. `environment.yml` combines their packages in a single Python 3.10 environment. When a package version differed across the lab environments, the newer version is pinned.

The lab `tabpfn2` environment used PyTorch `2.7.0+cu118`. This file installs PyTorch `2.7.0`. Explainability imports `tabpfn_extensions`, which was an editable local install at version 0.1.0; the environment file installs the earliest published release, `tabpfn-extensions==0.2.0`.

```bash
conda env create -f environment.yml
conda activate preclinical-ad
```

## Running the analyses

Many notebooks currently reference lab-specific absolute paths. Before running them, update the input and output paths for your environment and place approved data in a local ignored directory such as `clinical_data/Data/`.

The main biomarker-classification entry point is:

```bash
python biomarker_prediction/tau_classification.py \
  --data_path /path/to/approved/model_preprocessed_data.csv \
  --output_dir /path/to/output
```

Use `--help` to view the available filtering, GPU, TabPFN, and bootstrap options:

```bash
python biomarker_prediction/tau_classification.py --help
```

The provided shell scripts contain SLURM settings and lab-specific paths and should be adapted before submission on another cluster.

## Reproducibility

Random seeds and analysis thresholds are defined in `biomarker_prediction/config.py` and `clinical_data/clinical_data_analysis/statistical_analysis/settings.py`.
