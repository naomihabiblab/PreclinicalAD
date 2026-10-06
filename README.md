# Preclinical AD

Analysis code and notebooks for studying clinical, cognitive, behavioral, genetic, and blood-biomarker features associated with early and preclinical Alzheimer's disease. The repository focuses on plasma biomarkers—particularly p-tau217, GFAP, and NfL—and includes statistical analyses, cohort summaries, predictive modeling, and model explainability workflows.

## Repository structure

- `biomarkers/` — exploratory analyses of p-tau217, GFAP, NfL, and APOE, including biomarker thresholds and feature distributions.
- `clinical_data/` — cohort summaries, cognitive and behavioral analyses, APOE analyses, and reusable statistical-analysis scripts for biomarker/trait associations.
- `biomarker_prediction/` — preprocessing, model training, hyperparameter optimization, bootstrap evaluation, ensemble methods, visualization, and explainability for p-tau217 classification.

## Data availability and privacy

The participant-level clinical datasets used by these analyses are **not included** in this repository. Several notebooks and scripts expect files such as `fightAD_general_data_table.xlsx` or preprocessed CSV files in a local `clinical_data/Data/` directory. These inputs may contain sensitive research data and must be obtained through the study's approved data-access process.

Do not commit participant-level data, model artifacts containing participant data, credentials, or other restricted outputs. The included `.gitignore` excludes the common local data and generated-artifact locations.

The notebooks are distributed without saved outputs or embedded figures. Run them only with approved local data, and review any regenerated outputs under the study's data-sharing and publication policies before redistribution.

## Environment

The project uses Python and Jupyter notebooks. Dependencies used across the analyses include:

- `numpy`, `pandas`, `scipy`, and `statsmodels`
- `matplotlib` and `seaborn`
- `scikit-learn`, `catboost`, `optuna`, and `torch`
- `tabpfn`
- optional explainability packages used by the scripts in `biomarker_prediction/explainability/`, including SHAP-related tools

No pinned environment file is currently included, so package versions should be recorded before attempting exact reproduction.

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

## Reproducibility notes

- Random seeds and analysis thresholds are defined in the relevant settings/configuration modules, including `biomarker_prediction/config.py` and `clinical_data/clinical_data_analysis/statistical_analysis/settings.py`.
- Notebook outputs reflect the environment and data available when each notebook was last executed.
- Generated figures, logs, serialized models, and derived data should remain outside version control unless they have been explicitly approved for release.

## Intended use

This repository contains research code. It is not a clinical diagnostic tool and should not be used to guide patient care.
