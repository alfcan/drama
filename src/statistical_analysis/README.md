# Statistical Analysis for Mutation Testing

This directory contains statistical analysis tools for evaluating the impact of mutation operators on fairness symptoms in machine learning models. The analysis helps identify which mutation operators produce statistically significant changes in fairness metrics.

## Overview

The statistical analysis framework consists of three main components:

1. **Sign Test** (`mutation_sign_test.py`) - Non-parametric test for detecting directional changes
2. **Wilcoxon Signed-Rank Test** (`mutation_wilcoxon_test.py`) - Non-parametric test considering magnitude of changes
3. **Comprehensive Analysis** (`comprehensive_analysis.py`) - Combined analysis with consensus evaluation

## Data Format

The analysis works with CSV files containing mutation results with the following structure:

```csv
column,dataset,mutation_type,symptom_name,symptom_difference,symptom_abs_difference,symptom_percent_change
race,adult,category_flip,APD,-0.123456,0.123456,-12.34
sex,adult,add_noise,Mutual Information,0.045678,0.045678,4.56
```

### Key Columns:
- `column`: Sensitive attribute being mutated (e.g., 'race', 'sex')
- `mutation_type`: Type of mutation applied (e.g., 'category_flip', 'add_noise')
- `symptom_name`: Fairness metric being measured (e.g., 'APD', 'Mutual Information')
- `symptom_difference`: Change in symptom value (post - pre mutation)
- `symptom_abs_difference`: Absolute value of the difference
- `symptom_percent_change`: Percentage change in symptom value

## Statistical Tests

### 1. Sign Test

**Purpose**: Determines if mutations consistently increase or decrease fairness symptoms.

**How it works**:
- Counts positive vs. negative changes in symptom values
- Tests if the proportion of positive changes differs significantly from 50%
- Uses binomial test with null hypothesis: P(positive change) = 0.5

**When to use**:
- When you want to detect consistent directional bias
- When the magnitude of change is less important than direction
- Robust to outliers and doesn't assume normal distribution

**Interpretation**:
- Significant result indicates consistent directional impact
- P-value < 0.05 suggests the mutation operator has a systematic effect
- Positive bias ratio > 0.5 means mutations tend to increase symptoms
- Negative bias ratio < 0.5 means mutations tend to decrease symptoms

### 2. Wilcoxon Signed-Rank Test

**Purpose**: Tests if the median difference in symptom values is significantly different from zero.

**How it works**:
- Ranks the absolute differences (excluding zeros)
- Compares sum of ranks for positive vs. negative differences
- More sensitive to magnitude of changes than Sign Test

**When to use**:
- When magnitude of change matters
- When you want to detect subtle but consistent effects
- When data may have outliers but is roughly symmetric

**Interpretation**:
- Significant result indicates the mutation has a measurable impact
- P-value < 0.05 suggests systematic change in symptom values
- Considers both direction and magnitude of changes
- More powerful than Sign Test when assumptions are met

### 3. Consensus Analysis

The comprehensive analysis combines both tests to provide robust conclusions:

**Agreement Categories**:
- **Both Significant**: Strong evidence of mutation impact
- **One Significant**: Moderate evidence, investigate further
- **Neither Significant**: No detectable impact or insufficient data

## Usage

### Running Individual Tests

```bash
# Run Sign Test only
python mutation_sign_test.py

# Run Wilcoxon Test only
python mutation_wilcoxon_test.py
```

### Running Comprehensive Analysis

```bash
# Run both tests with consensus analysis
python comprehensive_analysis.py
```

### Programmatic Usage

```python
from mutation_sign_test import MutationSignTest
from mutation_wilcoxon_test import MutationWilcoxonTest
from comprehensive_analysis import ComprehensiveMutationAnalysis

# Initialize with your CSV file
csv_path = "path/to/your/results.csv"

# Run individual tests
sign_test = MutationSignTest(csv_path)
sign_results = sign_test.generate_summary()

wilcoxon_test = MutationWilcoxonTest(csv_path)
wilcoxon_results = wilcoxon_test.generate_summary()

# Run comprehensive analysis
analysis = ComprehensiveMutationAnalysis(csv_path)
combined_results = analysis.run_complete_analysis()
```

## Output Files

The analysis generates several output files:

1. **`sign_test_results.json`** - Detailed Sign Test results
2. **`wilcoxon_results.json`** - Detailed Wilcoxon Test results
3. **`comprehensive_analysis_results.json`** - Combined results with consensus
4. **`analysis_report.txt`** - Human-readable summary report

## Interpreting Results

### Statistical Significance

- **P-value < 0.01**: Strong evidence of effect
- **P-value < 0.05**: Moderate evidence of effect
- **P-value ≥ 0.05**: No significant evidence of effect

### Effect Size Indicators

- **Mean/Median Difference**: Average change magnitude
- **Standard Deviation**: Variability in changes
- **Percentage of Non-zero Changes**: How often mutations have any effect

### Practical Significance

Consider both statistical and practical significance:

1. **Statistical Significance**: Is the effect real?
2. **Effect Size**: How large is the effect?
3. **Consistency**: Do both tests agree?
4. **Context**: Is the effect meaningful for your use case?

## Mutation Operator Categories

Based on typical results, mutation operators can be categorized as:

### High Impact Operators
- Consistently significant in both tests
- Large effect sizes
- High percentage of non-zero changes
- Examples: `category_flip`, `replace_synonyms`

### Moderate Impact Operators
- Significant in one test or inconsistent results
- Medium effect sizes
- Moderate percentage of changes
- Examples: `add_noise`, `scale_values`

### Low Impact Operators
- Rarely significant
- Small effect sizes
- Many zero changes
- Examples: `increment_decrement_feature`

## Recommendations

### For Significant Results:
1. **Investigate the mechanism**: Why does this mutation affect fairness?
2. **Consider mitigation**: How can you make your model more robust?
3. **Prioritize testing**: Focus on high-impact mutations in future tests

### For Non-significant Results:
1. **Check sample size**: Do you have enough data?
2. **Examine data quality**: Are mutations actually changing the data?
3. **Consider alternative tests**: Maybe the effect exists but these tests can't detect it

### For Inconsistent Results:
1. **Examine data distribution**: Are there outliers or skewness?
2. **Check assumptions**: Do the test assumptions hold?
3. **Consider effect size**: Small effects might be inconsistently detected

## Limitations

### Test Assumptions:
- **Sign Test**: Assumes independent observations
- **Wilcoxon Test**: Assumes symmetric distribution of differences
- **Both**: Assume paired observations (pre/post mutation)

### Data Requirements:
- Need sufficient non-zero differences for meaningful results
- Require paired pre/post mutation measurements
- Assume mutations are applied independently

### Interpretation Caveats:
- Statistical significance ≠ practical significance
- Multiple testing may inflate Type I error rates
- Results depend on mutation implementation quality

## Troubleshooting

### Common Issues:

1. **"Insufficient non-zero differences"**
   - Solution: Check if mutations are actually changing the data
   - Verify mutation implementation

2. **All p-values are 1.0**
   - Solution: Examine data for actual differences
   - Check CSV format and column names

3. **Inconsistent results between tests**
   - Solution: Examine data distribution and outliers
   - Consider effect size alongside significance

4. **Memory or performance issues**
   - Solution: Process data in chunks or filter to relevant subsets
   - Consider sampling for very large datasets

## Dependencies

```
pandas>=1.3.0
numpy>=1.20.0
scipy>=1.7.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

Install with:
```bash
pip install pandas numpy scipy matplotlib seaborn
```

## Contact and Support

For questions about the statistical analysis:
1. Check this README for common issues
2. Examine the generated report files for detailed results
3. Review the JSON output files for programmatic access to results

## Version History

- **v1.0**: Initial implementation with Sign Test and Wilcoxon Test
- **v1.1**: Added comprehensive analysis and consensus evaluation
- **v1.2**: Enhanced reporting and visualization capabilities