import sys
sys.path.append('../../')
from BCG.clinical_data.clinical_data_analysis.statistical_analysis.settings import *

RISK_FACTORS_TESTS = [
    {
        'label': 'Number of risk factors',
        'col': 'Number_of_risk_factors',
        'test_type': 'Ordinal',
        'threshold': None,
        'threshold_label': None,
        'threshold_test_type': None
    },
    {
        'label': 'Anxiety/Depression',
        'col': ANEXITY_COL,
        'test_type': 'Categorical',
        'threshold': None,
        'threshold_label': None,
        'threshold_test_type': None
    },
    {
        'label': 'BMI',
        'col': 'BMI',
        'test_type': 'Continuous',
        'threshold': None,
        'threshold_label': None,
        'threshold_test_type': None
    },
    {
        'label': 'Heart disease',
        'col': 'Heart_disease',
        'test_type': 'Categorical',
        'threshold': None,
        'threshold_label': None,
        'threshold_test_type': None
    },
    {
        'label': 'Diabetes',
        'col': 'Diabetes',
        'test_type': 'Categorical',
        'threshold': None,
        'threshold_label': None,
        'threshold_test_type': None
    },
    {
        'label': 'Hypertension',
        'col': 'Hypertension',
        'test_type': 'Categorical',
        'threshold': None,
        'threshold_label': None,
        'threshold_test_type': None
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
        'label': 'Periodontal disease',
        'col': 'Periodontal disease',
        'test_type': 'Categorical',
        'threshold': None,
        'threshold_label': None,
        'threshold_test_type': None
    },
    {
        'label': 'Recurrent UTI/pneumonia/other infection',
        'col': 'Recurrent_UTI/_pneumonia/other_infection',
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
        'threshold_test_type': None
    },
    {
        'label': 'Smoking',
        'col': 'Smoking',
        'test_type': 'Categorical',
        'threshold': None,
        'threshold_label': None,
        'threshold_test_type': None
    },
    {
        'label': 'Pack years',
        'col': 'Pack_years',
        'test_type': 'Continuous',
        'threshold': None,
        'threshold_label': None,
        'threshold_test_type': None
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
        'label': 'Insomnia',
        'col': 'Insomnia',
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
    {
        'label': 'Physical activity',
        'col': 'Physical activity',
        'test_type': 'Ordinal',
        'threshold': None,
        'threshold_label': None,
        'threshold_test_type': None
    },
    {
        'label': 'Herpes zoster vaccine',
        'col': 'Herpes_zoster_vaccine',
        'test_type': 'Categorical',
        'threshold': None,
        'threshold_label': None,
        'threshold_test_type': None
    },
] 