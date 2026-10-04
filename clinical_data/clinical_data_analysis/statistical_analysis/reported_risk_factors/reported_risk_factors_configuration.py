import sys
sys.path.append('../../')
from settings import *

RISK_FACTORS_TESTS = [
    {
        'label': 'Smoking',
        'col': 'Smoking',
        'test_type': 'Categorical',
        'threshold': None,
        'threshold_label': None,
        'threshold_test_type': None,
        'alternative': 'less' # high smoking score is worse
    },
    {
        'label': 'Anxiety/Depression',
        'col': ANEXITY_COL,
        'test_type': 'Categorical',
        'threshold': None,
        'threshold_label': None,
        'threshold_test_type': None,
        'alternative': 'less' # high anxiety/depression score is worse
    },
    {
        'label': 'BMI',
        'col': 'BMI',
        'test_type': 'Continuous',
        'threshold': None,
        'threshold_label': None,
        'threshold_test_type': None,
        'alternative': 'greater' # high bmi score is better in AD
    },
    {
        'label': 'Heart disease',
        'col': 'Heart_disease',
        'test_type': 'Categorical',
        'threshold': None,
        'threshold_label': None,
        'threshold_test_type': None,
        'alternative': 'less' # high heart disease score is worse
    },
    {
        'label': 'Diabetes mellitus',
        'col': 'Diabetes',
        'test_type': 'Categorical',
        'threshold': None,
        'threshold_label': None,
        'threshold_test_type': None,
        'alternative': 'less' # high diabetes score is worse
    },
    {
        'label': 'Hypertension',
        'col': 'Hypertension',
        'test_type': 'Categorical',
        'threshold': None,
        'threshold_label': None,
        'threshold_test_type': None,
        'alternative': 'less' # high hypertension score is worse
    },
    {
        'label': 'Hyperlipidemia',
        'col': 'Hyperlipidemia',
        'test_type': 'Categorical',
        'threshold': None,
        'threshold_label': None,
        'threshold_test_type': None
    },

    {
        'label': 'Menopause age',
        'col': 'Menopause age',
        'test_type': 'Continuous',
        'threshold': None,
        'threshold_label': None,
        'threshold_test_type': None,
        'alternative': 'less' # high menopause age score is worse
    }, 
    {
        'label': 'Periodontal disease',
        'col': 'Periodontal disease',
        'test_type': 'Categorical',
        'threshold': None,
        'threshold_label': None,
        'threshold_test_type': None
    },
    {
        'label': 'Systemic inflammatory disease',
        'col': 'Systemic_inflammatory_disease',
        'test_type': 'Categorical',
        'threshold': None,
        'threshold_label': None,
        'threshold_test_type': None,
        'alternative': 'less' # high systemic inflammatory disease score is worse
    },
    {
        'label': 'Alcohol consumption',
        'col': 'Alcohol_consumption',
        'test_type': 'Categorical',
        'threshold': None,
        'threshold_label': None,
        'threshold_test_type': None
    },
    {
        'label': 'OSA',
        'col': 'OSA',
        'test_type': 'Categorical',
        'threshold': None,
        'threshold_label': None,
        'threshold_test_type': None
    },
    {
        'label': 'Severity of OSA',
        'col': 'severity of OSA',
        'test_type': 'Ordinal',
        'threshold': None,
        'threshold_label': None,
        'threshold_test_type': None
    },

] 