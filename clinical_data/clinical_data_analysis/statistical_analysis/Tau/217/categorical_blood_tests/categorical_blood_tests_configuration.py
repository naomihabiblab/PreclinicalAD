import sys
sys.path.append('../../')
from BCG.clinical_data.clinical_data_analysis.statistical_analysis.settings import *

CATEGORICAL_BLOOD_TESTS = [
    {
        'label': 'Hemoglobin A1C (Categorical)',
        'col': 'HA1C_CAT',
        'test_type': 'Ordinal',
        'threshold': None,
        'threshold_label': None,
        'threshold_test_type': None
    },
    {
        'label': 'Cholesterol (Categorical)',
        'col': 'CHOL_CAT',
        'test_type': 'Ordinal',
        'threshold': None,
        'threshold_label': None,
        'threshold_test_type': None
    },
    {
        'label': 'Triglycerides (Categorical)',
        'col': 'TRG_CAT',
        'test_type': 'Ordinal',
        'threshold': None,
        'threshold_label': None,
        'threshold_test_type': None
    },
    {
        'label': 'LDL (Categorical)',
        'col': 'LDL_CAT',
        'test_type': 'Ordinal',
        'threshold': None,
        'threshold_label': None,
        'threshold_test_type': None
    },
    {
        'label': 'CRP (Categorical)',
        'col': 'CRP_CAT',
        'test_type': 'Ordinal',
        'threshold': None,
        'threshold_label': None,
        'threshold_test_type': None
    },
    {
        'label': 'Lymphocytes (Categorical)',
        'col': 'LYMPH_CAT',
        'test_type': 'Ordinal',
        'threshold': None,
        'threshold_label': None,
        'threshold_test_type': None
    },
] 