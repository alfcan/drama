<a id="readme-top"></a>

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

<br />
<div align="center">
  <h3 align="center">DRAMA: Data-oRiented fAirness Mutation Analysis</h3>

  <p align="center">
    A framework designed to identify biases in datasets used for training machine learning models.
    <br />
    <a href="https://github.com/alfcan/drama"><strong>Explore the documentation »</strong></a>
    <br />
    <br />
    <a href="https://github.com/alfcan/drama/issues/new?labels=bug&template=bug-report---.md">Report a bug</a>
    ·
    <a href="https://github.com/alfcan/drama/issues/new?labels=enhancement&template=feature-request---.md">Request a Function</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#framework-overview">Framework Overview</a></li>
    <li><a href="#architecture">Architecture</a></li>
    <li><a href="#workflow">Workflow</a></li>
    <li><a href="#mutation-operators">Mutation Operators</a></li>
    <li><a href="#fairness-symptoms">Fairness Symptoms</a></li>
    <li><a href="#outputs">Outputs</a></li>
    <li><a href="#statistical-analysis">Statistical Analysis</a></li>
    <li><a href="#results-structure">Results Structure</a></li>
    <li><a href="#limitations--notes">Limitations & Notes</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->

## About The Project

DRAMA is a framework designed to identify biases in datasets used for training machine learning (ML) models. By using mutation operators, DRAMA introduces variations in datasets and evaluates the impact of these changes on fairness metrics.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

This section lists the main technologies used to develop DRAMA:

- [![Python][python-shield]][python-url]
- [![Pandas][pandas-shield]][pandas-url]
- [![TensorFlow][tensorflow-shield]][tensorflow-url]
- [![Scikit-learn][scikit-learn-shield]][scikit-learn-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->

## Getting Started

Here is an example of how you can set up the project locally.
To get a local copy up and running, follow these simple steps.

### Prerequisites

There are no specific prerequisites for using DRAMA.

- Place your CSV dataset in `data/` (e.g., `data/adult.csv`).
- Ensure the target/outcome column is present (binary or numeric).
- The framework removes rows with missing values and treats `?` as missing.

### Installation

1. Clone the Repository:
   ```sh
   git clone https://github.com/alfcan/drama.git
   ```
2. Install Dependencies:
   ```sh
   cd drama
   pip install -r requirements.txt
   ```
3. Run the Framework (interactive):
   ```sh
   cd src
   python main.py
   ```
   - Enter the dataset file name (e.g., `adult.csv`) when prompted.
   - Select the `target_attribute` (printed list of columns).
   - Select the `positive_label` (default `1` if unsure).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Framework Overview

DRAMA applies a set of mutation operators to each non-target feature and evaluates how these changes affect fairness-related symptoms across all potential sensitive attributes. For every mutation, it computes a baseline (pre-mutation) and post-mutation set of symptoms, then exports a streamlined CSV for statistical analysis and the transformed datasets in a hierarchical structure.

- Focuses on dataset-level fairness characteristics (no model training).
- Works with both numeric and categorical/textual features.
- Provides standalone statistical analysis modules to assess significance.

## Architecture

- `src/data/loader.py`: Loads datasets from CSV.
- `src/data/preprocessor.py`: Required preprocessing:
  - Removes rows with missing values (`?` treated as missing).
  - Applies one-hot encoding to categorical/boolean columns (temporary or full).
- `src/analysis/symptom_calculator.py`: Computes fairness symptoms:
  - Distribution/diversity, relationship to target, and fairness metrics.
- `src/mutation/mutation_operator.py`: Implements mutation operators:
  - Numeric: increment/decrement, scaling.
  - Categorical/Textual: category flip, synonym replacement, noise injection.
- `src/utils/result_handlers.py`: Exports streamlined results CSV.
- `src/utils/file_utils.py`: Creates feature-specific directories and exports transformed datasets.
- `src/statistical_analysis/*`: Sign Test, Wilcoxon Test, and comprehensive analysis with consensus.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Project Structure

Top-level directories:

- `src/`: Core framework code for dataset-level fairness symptom analysis, mutation application, and statistical testing.
- `data/`: Example datasets used by the main framework (CSV).
- `results/`: Outputs produced by `src/main.py` and analysis modules:
  - Streamlined results CSVs, comprehensive analysis reports, and hierarchical transformed datasets per feature.
- `RQ2/`: Model-based fairness mutation experiments (complementary to `src/`):
  - Per-dataset subfolders: `adult/`, `bank/`, `compas/`, `diabetic/`.
  - Common contents:
    - `dataset/`: base and transformed CSVs (`transformed_<dataset>_<feature>_<mutation>.csv`)
    - `main.py`, `metrics.py`, `utils.py`: experiment scripts and metric utilities
    - `results/`: baseline (`<dataset>_rf.csv`, `<dataset>_xgb.csv`) and transformed results
    - `stats_two_sided/`, `stats_greater/`: Wilcoxon test outputs (`p-val`, `CLES`, `metric`)
    - `analysis.ipynb`: plots/tables; `imgs/` figures; `table_rf.tex`, `table_xgb.tex` summaries
  - Role: quantifies the effect of feature-level mutations on trained model fairness metrics, validating whether dataset-level symptoms translate to model-level changes.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Workflow

1. Load dataset and remove missing values (`clean_data_only`).
2. Prompt user for `target_attribute` and `positive_label`.
3. Compute baseline symptoms for all features except the target.
4. For each feature:
   - Determine applicable mutation operators (based on dtype).
   - Apply mutation and compute post-mutation symptoms for all potential sensitive attributes using temporary one-hot encoding.
   - Record per-symptom differences and print a change summary.
5. Export:
   - Streamlined results CSV for statistical analysis.
   - Transformed datasets in hierarchical directories.

Notes:

- Change reporting threshold: prints changes when `|Δ| > 0.01`, flags “SIGNIFICANT CHANGE” at `|Δ| > 0.1` (console only; statistical tests use p-values).

## Mutation Operators

Implemented in `src/mutation/mutation_operator.py`:

- `increment_decrement_feature` (numeric): Random ±5% of feature range on ~20% rows.
- `scale_values` (numeric): Multiply by factor in `[0.8, 1.2]` on ~20% rows, preserves integer dtype.
- `category_flip` (categorical): Reassign ~15% of instances in each category to another category.
- `replace_synonyms` (textual): In ~10% rows, replace ~15% of words with WordNet synonyms.
- `add_noise` (textual/categorical strings): Inject simple character-level noise into ~10% rows.
  - For categorical strings, applies consistent mutations per unique category (<=45 unique values).

Applicable operators are selected per column via `src/utils/analysis_helpers.py`.

## Fairness Symptoms

Calculated in `src/analysis/symptom_calculator.py`:

- Distribution/Diversity:
  - `Gini Index`, `Shannon Entropy`, `Simpson Diversity`, `Imbalance Ratio`.
  - `Kurtosis`, `Skewness` (numeric only).
- Relationship to target:
  - `Mutual Information`, `Normalized Mutual Information`, `Kendall Tau` (numeric), `Correlation Ratio`.
- Fairness metrics (group-based):
  - `APD` (Absolute Probability Difference)
  - `Statistical Parity` (difference of positive rates)
  - `Disparate Impact` (ratio of positive rates)
  - `Unprivileged Unbalance`, `Privileged Unbalance`
  - `Unprivileged Pos Prob`, `Privileged Pos Prob`, `Pos Probability Diff`

For one-hot encoded categorical features, fairness metrics are aggregated over binary indicator columns (0 vs 1 groups).

## Outputs

1. Streamlined results CSV (per symptom):
   - Path: `results/results_{datasetName}_{timestamp}.csv`
   - Columns:
     - `column`, `dataset`, `column_type`, `mutation_type`, `is_sensitive` (false), `sensitive_attr_analyzed`
     - `symptom_name`, `pre_symptom_value`, `post_symptom_value`
     - `symptom_difference`, `symptom_abs_difference`, `symptom_percent_change`
2. Transformed datasets (per mutated feature/operator):
   - Path pattern: `results/{dataset}/feature_{featureName}_{timestamp}/transformed_{dataset}_{column}_{mutation}.csv`
3. Statistical analysis outputs (see below):
   - Sign Test and Wilcoxon summaries (JSON).
   - Comprehensive analysis JSON and `analysis_report.txt`.
   - Symptom-level Wilcoxon summary CSV/JSON.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Statistical Analysis

Tools located in `src/statistical_analysis/` operate on the streamlined results CSV.

- Sign Test (`mutation_sign_test.py`)
  - Purpose: directional consistency (positive vs. negative differences).
  - Usage:
    ```bash
    cd src/statistical_analysis && python mutation_sign_test.py ../../results/results_<dataset>_<timestamp>.csv
    ```
  - Output: `<results>_sign_test_results.json`
- Wilcoxon Signed-Rank Test (`mutation_wilcoxon_test.py`)
  - Purpose: median difference ≠ 0 (considers magnitude, excludes zeros).
  - Usage:
    ```bash
    cd src/statistical_analysis && python mutation_wilcoxon_test.py
    ```
  - Output: `wilcoxon_results.json`
- Symptom-level Wilcoxon (`wilcoxon_by_symptom.py`)
  - Aggregates per-symptom differences across entire dataset.
  - Usage:
    ```bash
    cd src/statistical_analysis && python wilcoxon_by_symptom.py --csv ../../results/results.csv --outdir ../../results
    ```
  - Output: `results/symptom_wilcoxon_summary.csv`, `results/symptom_wilcoxon_summary.json`
- Comprehensive Analysis (`comprehensive_analysis.py`)
  - Combines Sign Test and Wilcoxon; computes consensus/agreements.
  - Usage:
    ```bash
    cd src/statistical_analysis && python comprehensive_analysis.py
    ```
  - Prompts for filename in `results/` (e.g., `results_adult_YYYYMMDD_HHMMSS.csv`).
  - Output: `results/comprehensive_analysis_results.json`, `results/analysis_report.txt`

Interpretation guidelines (significance thresholds and effect size notes) are detailed in `src/statistical_analysis/README.md`.

## Results Structure

Outputs are organized under `results/`:

- `results/<dataset>/feature_<feature>_<timestamp>/transformed_<dataset>_<column>_<mutation>.csv`
- `results/results_<dataset>_<timestamp>.csv` (streamlined analysis input)
- Analysis:
  - `results/symptom_wilcoxon_summary.csv|.json`
  - `results/comprehensive_analysis_results.json`
  - `results/analysis_report.txt`

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Limitations & Notes

- Preprocessing strictly removes rows with missing values and treats `?` as missing.
- Group inference may fail on degenerate splits (e.g., empty halves); such metrics return `None`.
- Console “SIGNIFICANT CHANGE” uses a heuristic (`|Δ| > 0.1`); statistical significance relies on tests (p-values).
- Symptom calculations are dataset-level (no model predictions).
- Mutation parameters are configurable but have sensible defaults aligned with dataset realism.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTING -->

## Contributing

Contributions are what makes the open source community such an amazing place to learn, inspire and create. Any contributions you make will be **very much appreciated**.

If you have a suggestion that could improve this project, please fork the repository and create a pull request. You can also simply open an issue with the ‘enhancement’ tag.
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m ‘Add some AmazingFeature’`)
4. Push on the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->

## Contact

Alfonso Cannavale - [LinkedIn](https://www.linkedin.com/in/alfonso-cannavale-62150b229/)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[contributors-shield]: https://img.shields.io/github/contributors/alfcan/drama.svg?style=for-the-badge
[contributors-url]: https://github.com/alfcan/drama/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/alfcan/drama.svg?style=for-the-badge
[forks-url]: https://github.com/alfcan/drama/network/members
[stars-shield]: https://img.shields.io/github/stars/alfcan/drama.svg?style=for-the-badge
[stars-url]: https://github.com/alfcan/drama/stargazers
[issues-shield]: https://img.shields.io/github/issues/alfcan/drama.svg?style=for-the-badge
[issues-url]: https://github.com/alfcan/drama/issues
[license-shield]: https://img.shields.io/github/license/alfcan/drama.svg?style=for-the-badge
[license-url]: https://github.com/alfcan/drama/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/alfonso-cannavale-62150b229/
[python-shield]: https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54
[python-url]: https://www.python.org/
[pandas-shield]: https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white
[pandas-url]: https://pandas.pydata.org/
[tensorflow-shield]: https://img.shields.io/badge/tensorflow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white
[tensorflow-url]: https://www.tensorflow.org/
[scikit-learn-shield]: https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white
[scikit-learn-url]: https://scikit-learn.org/
