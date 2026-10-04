"""
Examples of how to use different statistical test alternatives (two-sided, greater, less)

In your configuration files (e.g., behavioral_configuration.py, cognitive_configuration.py, etc.),
you can now specify the type of statistical test using the 'alternative' parameter:

Options:
- 'two-sided' (default): Tests if there's any difference between groups
  - For Continuous/Ordinal: H1: Low ≠ High
  - For Categorical: H1: There is an association (uses chi-square test)

- 'greater': Tests directional difference
  - For Continuous/Ordinal: H1: Low > High (Low group is higher)
  - For Categorical: Parameter ignored, always uses two-sided chi-square test

- 'less': Tests directional difference  
  - For Continuous/Ordinal: H1: Low < High (High group is higher)
  - For Categorical: Parameter ignored, always uses two-sided chi-square test

Note: For categorical tests, the 'alternative' parameter is ignored - chi-square tests are inherently two-sided.

Examples:
==========

# Example 1: Two-sided test (default)
{
    'label': 'Hemoglobin A1C',
    'col': 'Hemoglobin_A1C',
    'test_type': 'Continuous',
    'threshold': None,
    'threshold_label': None,
    'threshold_test_type': None,
    'alternative': 'two-sided'  # Optional, this is the default
}

# Example 2: One-sided greater test (testing if Low biomarker group has higher values)
{
    'label': 'CRP',
    'col': 'CRP',
    'test_type': 'Continuous',
    'threshold': None,
    'threshold_label': None,
    'threshold_test_type': None,
    'alternative': 'greater'  # Tests if Low biomarker group has higher CRP
}

# Example 3: One-sided less test (testing if High biomarker group has higher values)
{
    'label': 'MOCA score',
    'col': MOCA_COL,
    'test_type': 'Continuous',
    'threshold': None,
    'threshold_label': None,
    'threshold_test_type': None,
    'alternative': 'less'  # Tests if High biomarker group has higher MOCA scores
}

# Example 4: With threshold test (using different alternatives)
{
    'label': 'Some Test',
    'col': 'some_column',
    'test_type': 'Ordinal',
    'threshold': 2,
    'threshold_label': 'Some Test Threshold',
    'threshold_test_type': 'Categorical',
    'alternative': 'less',  # Main test: tests if Low < High
    'threshold_alternative': 'two-sided'  # Threshold test: ignored for categorical, always two-sided chi-square
}

# Example 5: Categorical test (alternative parameter is ignored - always two-sided)
{
    'label': 'Heart disease',
    'col': 'Heart_disease',
    'test_type': 'Categorical',
    'threshold': None,
    'threshold_label': None,
    'threshold_test_type': None
    # Note: 'alternative' parameter is ignored for categorical tests
    # Chi-square tests are inherently two-sided
}

Understanding the direction:
============================
- Low biomarker group = group1
- High biomarker group = group2

When you use 'greater':
- Tests if group1 (Low) > group2 (High)
- This tests if the LOW biomarker group has HIGHER values in your variable

When you use 'less':
- Tests if group1 (Low) < group2 (High)  
- This tests if the HIGH biomarker group has HIGHER values in your variable

Example Scenario:
=================
If you're testing CRP (inflammatory marker) levels:
- Alternative: 'greater' means you're testing if people with LOW biomarker have HIGHER CRP
- Alternative: 'less' means you're testing if people with HIGH biomarker have HIGHER CRP
- Alternative: 'two-sided' tests if there's any difference (either direction)
"""

# Example configurations:

# For blood tests configuration (if you expect High biomarker group to have worse values):
BLOOD_TEST_EXAMPLE_LESS = {
    'label': 'Hemoglobin A1C',
    'col': 'Hemoglobin_A1C',
    'test_type': 'Continuous',
    'threshold': None,
    'threshold_label': None,
    'threshold_test_type': None,
    'alternative': 'less'  # Testing if High biomarker group has higher A1C
}

# For cognitive tests (if you expect High biomarker group to have worse scores):
COGNITIVE_TEST_EXAMPLE_LESS = {
    'label': 'MOCA score',
    'col': MOCA_COL,
    'test_type': 'Continuous',
    'threshold': None,
    'threshold_label': None,
    'threshold_test_type': None,
    'alternative': 'less'  # Testing if High biomarker group has lower MOCA scores
}

# For behavioral tests (typical two-sided):
BEHAVIORAL_TEST_EXAMPLE_TWO_SIDED = {
    'label': 'MBI-C',
    'col': MBI_C_COL,
    'test_type': 'Ordinal',
    'threshold': 6,
    'threshold_label': 'MBI-C Threshold',
    'threshold_test_type': 'Categorical',
    'alternative': 'two-sided',  # Default: test for any difference
    'threshold_alternative': 'two-sided'
}
