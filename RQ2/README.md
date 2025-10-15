<a id="readme-top"></a>

<br />
<div align="center">
  <h3 align="center">RQ2: Model-Based Fairness Mutation Experiments</h3>

  <p align="center">
    Reproducible experiments that assess fairness metrics on ML models trained on mutated datasets across canonical benchmarks (Adult, Bank, COMPAS, Diabetic).
    <br />
    <a href="https://github.com/alfcan/drama"><strong>Back to main documentation »</strong></a>
  </p>
</div>

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-this-folder">About This Folder</a></li>
    <li><a href="#contents">Contents</a></li>
  </ol>
</details>

## About This Folder

This directory hosts experiments for Research Question 2 (RQ2), focusing on how controlled dataset mutations affect model-level fairness metrics. Each subfolder (`adult`, `bank`, `compas`, `diabetic`) contains scripts and assets to run and analyze experiments for a specific dataset.

- Complements the core framework in `src/`, which analyzes dataset-level fairness symptoms.
- RQ2 evaluates fairness impact using trained models and reports statistical significance of changes due to mutations.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contents

Each dataset subfolder shares a consistent structure:

- `analysis.ipynb`: Jupyter notebook to visualize metrics, run Wilcoxon tests, and prepare summary tables.
- `dataset/`: Base and transformed CSVs used for modeling and analysis.
  - Base: `<dataset>.csv` (e.g., `adult.csv`, `bank.csv`, `compas.csv`, `diabetic.csv`)
  - Transformed: `transformed_<dataset>_<feature>_<mutation>.csv` (e.g., `transformed_adult_race_category_flip.csv`)
- `imgs/`: Generated figures (PDFs) of fairness metric distributions for base vs transformed datasets.
- `main.py`: Script to run model training/evaluation and generate results CSVs for baseline and mutated datasets.
- `metrics.py`: Fairness metrics utilities (e.g., Statistical Parity, Equal Opportunity, Average Odds).
- `results/`: Per-dataset CSV outputs from the experiments.
  - Baselines: `<dataset>_rf.csv`, `<dataset>_xgb.csv`
  - Transformed: `transformed_<dataset>_<feature>_<model>.csv`
- `stats_two_sided/`: Wilcoxon Signed-Rank Test (two-sided) results per feature/model/mutation; columns include `p-val`, `CLES`, `metric`.
- `stats_greater/`: Wilcoxon Signed-Rank Test (one-sided, “less”) results per feature/model/mutation; columns include `p-val`, `CLES`, `metric`.
- `table_rf.tex`, `table_xgb.tex`: LaTeX tables summarizing per-feature p-values for fairness metrics.

Dataset-specific notes and examples:

- `adult/`
  - Transformed datasets: e.g., `transformed_adult_race_category_flip.csv`
  - Results: `adult_rf.csv`, `adult_xgb.csv`, plus transformed variants by feature and model.
- `bank/`
  - Transformed datasets: e.g., `transformed_bank_previous_scale_values.csv`, `transformed_bank_poutcome_replace_synonyms.csv`
  - Additional `grouped/` may be present for helper artifacts.
- `compas/`
  - Transformed datasets: e.g., `transformed_compas_score_text_add_noise.csv`
- `diabetic/`
  - Transformed datasets derived from medication or demographic features (e.g., `chlorpropamide_*` transformations)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Prerequisites

- Base dataset CSVs should exist in each subfolder’s `dataset/` directory (e.g., `dataset/adult.csv`).
- Transformed datasets should follow the naming convention `transformed_<dataset>_<feature>_<mutation>.csv` and be placed under `dataset/`.
- Python environment set up with project dependencies (see below).

<p align="right">(<a href="#readme-top">back to top</a>)</p>
