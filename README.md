# Early Detection of ICU Delirium from EHR Data using a Machine Learning Ensemble

This is a college capstone project investigating whether a machine learning ensemble can detect delirium in intensive care unit (ICU) patients earlier and more reliably than the standard shift-based CAM-ICU screening. Delirium is a sudden, serious state of confusion that is common in the ICU and is linked to longer hospital stays, higher mortality, and long-term cognitive impairment, yet it is frequently missed because current screening depends on intermittent manual assessment. Using structured electronic health record (EHR) data from the MIMIC-IV v3.1 dataset, we trained a Random Forest, an XGBoost model, and an LSTM, then combined them into an equal-weight probability-averaging ensemble. The goals were to compare the individual models against the ensemble, quantify how much earlier the model flags delirium relative to CAM-ICU, and analyse performance across demographic subgroups.

_Burak Kilic ([@your-handle](https://github.com/9iiota/)) • Kieran van der Heijden ([@your-handle](https://github.com/)) • Amen Tsehaie ([@your-handle](https://github.com/)) • Hady Al-Tamimi ([@Hadyalt](https://github.com/Hadyalt))_

## Requirements

- Python 3.13.3
- Jupyter Notebook
- A CUDA-capable GPU for the LSTM (developed on an NVIDIA RTX 2080 SUPER, CUDA 12.8). CPU works but is slower.
- Credentialed access to MIMIC-IV v3.1 (see below)
- _Required packages_:

  | Purpose | Library | Minimum version |
  |---------|---------|-----------------|
  | Database | DuckDB | 1.0 |
  | Data handling | pandas | 2.2 |
  | Data handling | NumPy | 2.0 |
  | Data handling | PyArrow | 15.0 |
  | Machine learning | scikit-learn | 1.5 |
  | Machine learning | XGBoost | 2.1 |
  | Deep learning | PyTorch (CUDA 12.8) | 2.9 |
  | Hyperparameter tuning | Optuna | any recent |
  | Interpretability | SHAP | 0.45 |
  | Plotting | Matplotlib | 3.9 |
  | Progress bars | tqdm | any recent |

## Getting the data

MIMIC-IV is a freely accessible but **credentialed** dataset hosted on PhysioNet. It cannot be redistributed, so it is not included in this repository and you must obtain it yourself.

1. Complete the CITI "Data or Specimens Only Research" training and sign the PhysioNet Data Use Agreement (DUA).
2. Download MIMIC-IV v3.1 from https://physionet.org/content/mimiciv/3.1/ in its compressed `.csv.gz` format.
3. Place the extracted folder so that the path resolves to `../mimic/physionet.org/files/mimiciv/3.1` relative to the notebook, or update `MIMIC_PATH` at the top of the Setup section.

The project uses an out-of-core DuckDB architecture, so the raw `.csv.gz` files are queried in place rather than loaded fully into memory. On first run, the Setup cells build a local `mimiciv.duckdb` file (with the `hosp` and `icu` schemas and several helper views) directly from the compressed files. On later runs, the existing database is reused. Only the eight tables the project actually needs are loaded: `hosp.patients`, `hosp.admissions`, `hosp.labevents`, `hosp.d_labitems`, `icu.icustays`, `icu.chartevents`, `icu.d_items`, and `icu.inputevents`.

## Project structure

```
capstone/
├── main.ipynb                      # End-to-end pipeline (run top to bottom)
├── mimiciv.duckdb                  # Local DuckDB, built on first run (not committed)
├── data/                           # Intermediate artifacts (not committed)
│   ├── cohort.parquet              # Final 57,054-stay cohort with labels
│   ├── master_features.parquet     # Windowed, model-ready feature matrix
│   ├── val_predictions.parquet     # Per-model validation probabilities
│   ├── test_predictions.parquet    # Per-model test probabilities
│   ├── subgroup_race.csv           # Subgroup metric tables
│   ├── subgroup_age.csv
│   └── subgroup_gender.csv
├── figures/                        # All generated plots at 150 DPI (not committed)
├── models/                         # Saved XGBoost model + feature columns (not committed)
└── README.md                       # This file
```

The `data/`, `figures/`, and `models/` folders are created automatically when the notebook runs. Their contents are derived artifacts and are not committed to the repository.

## Workflow

The whole project lives in a single notebook, `main.ipynb`, organised into the sections below. Each section reads the artifacts produced by the previous one, so the notebook is meant to be run top to bottom. Where a stage is expensive, it checks for an existing output file first and skips recomputation if it is present.

1. **Setup**: Connects to (or builds) the DuckDB database from the raw MIMIC-IV files and defines a `save_fig` helper.
2. **Exploration**: Characterises the cohort (demographics, ICU length of stay, care units), identifies item `228332` as the usable CAM-ICU delirium label, evaluates assessment coverage, and applies the cohort eligibility criteria. The final cohort of **57,054 ICU stays** is saved to `data/cohort.parquet`.
3. **Feature Engineering**: Segments each stay into fixed **6-hour windows** with a **12-hour look-ahead label** (`delirium_next_12h`), aggregates vitals, labs, and medications per window (mean, min, max, std for continuous variables; binary flags for medication classes), pivots to wide format, and assembles the master feature table saved to `data/master_features.parquet`.
4. **Modeling**: Excludes windows where the patient was already delirious, performs a **patient-stratified 70/15/15 split** on `subject_id` (29,627 / 6,349 / 6,349 patients) to prevent leakage, imputes missing values with the training-set median, and trains the Random Forest, XGBoost, and LSTM. The three models are then combined into an equal-weight ensemble. Per-model probabilities are saved for validation and test sets.
5. **Evaluation**: Selects the decision threshold on the validation set at a target sensitivity of 0.80, evaluates lead time relative to the first positive CAM-ICU assessment, and computes subgroup metrics by race, age, and gender.
6. **Interpretation**: Computes SHAP values for the XGBoost model on a 5,000-window sample (TreeSHAP gives exact attributions for tree models) and produces beeswarm, dependence, and waterfall plots.
7. **Results**: Produces the final test-set comparison tables and figures (model comparison, validation-vs-test, confusion matrix, lead-time and subgroup summaries).

Reproducibility note: a fixed `random_state` / seed of `42` is used throughout (split, Random Forest, XGBoost, and LSTM model selection) so runs are repeatable.

## Results

The full methodology and discussion are in the accompanying paper. In summary, the **equal-weight ensemble was the best model on every reported metric**, though the margins over the individual models were modest. On the held-out test set, evaluated at the threshold selected on the validation set:

| Model | AUROC | AUPRC | Sensitivity | Precision |
|-------|-------|-------|-------------|-----------|
| Random Forest | 0.826 | 0.324 | 0.795 | 0.250 |
| XGBoost | 0.830 | 0.336 | 0.792 | 0.253 |
| LSTM | 0.837 | 0.342 | 0.791 | 0.256 |
| **Ensemble** | **0.840** | **0.348** | **0.792** | **0.260** |

At the operating point (target sensitivity 0.80), roughly one in four flagged windows was a true positive, which is expected when targeting high sensitivity on a low base rate (positive windows make up about 10.3% of all windows).

Other key findings:

- **Lead time**: Among delirious test stays the model flagged, **53.7% were caught before the first positive CAM-ICU assessment**, with a median lead time of **14.7 hours** (mean 25.5 h, IQR 5.5 to 35.2 h). This exceeds the 8-to-12-hour gap inherent in shift-based screening.
- **Subgroup fairness**: Meaningful performance gaps were found for Asian patients (about 13% below the White subgroup) and patients aged 80 or older (about 12% below the 50 to 64 group). These are the smallest subgroups, so part of the gap may be variance, but a genuine disparity cannot be ruled out.
- **Explainability**: SHAP confirmed the model relies on clinically plausible features: GCS subscores (especially verbal), patient age, ICU type, and sedative or antipsychotic exposure.

The model is best understood as a complement to routine CAM-ICU screening rather than a replacement. External multi-centre validation, subgroup bias mitigation, and prospective evaluation remain necessary before any clinical deployment.

## Code availability

The full code for data extraction, preprocessing, modeling, and evaluation is in this repository: https://github.com/Hadyalt/AI_In_Healthcare_Capstone_Project
