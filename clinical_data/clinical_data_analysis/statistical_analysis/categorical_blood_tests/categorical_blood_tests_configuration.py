import sys
sys.path.append('../../')
from settings import *

CATEGORICAL_BLOOD_TESTS = [
    {
        'label': 'Hemoglobin A1C (Categorical)',
        'col': 'HA1C_CAT',
        'test_type': 'Ordinal',
        'threshold': None,
        'threshold_label': None,
        'threshold_test_type': None,
        'alternative': 'less' # high hemoglobin a1c score is worse
    },
    {
        'label': 'Cholesterol (Categorical)',
        'col': 'CHOL_CAT',
        'test_type': 'Ordinal',
        'threshold': None,
        'threshold_label': None,
        'threshold_test_type': None,
        'alternative': 'less' # high cholesterol score is worse
    },
    {
        'label': 'Triglycerides (Categorical)',
        'col': 'TRG_CAT',
        'test_type': 'Ordinal',
        'threshold': None,
        'threshold_label': None,
        'threshold_test_type': None,
        'alternative': 'less' # high triglycerides score is worse
    },
    {
        'label': 'LDL (Categorical)',
        'col': 'LDL_CAT',
        'test_type': 'Ordinal',
        'threshold': None,
        'threshold_label': None,
        'threshold_test_type': None,
        'alternative': 'less' # high ldl score is worse
    },
    {
        'label': 'CRP (Categorical)',
        'col': 'CRP_CAT',
        'test_type': 'Ordinal',
        'threshold': None,
        'threshold_label': None,
        'threshold_test_type': None,
        'alternative': 'less' # high crp score is worse
    },
    {
        'label': 'Lymphocytes (Categorical)',
        'col': 'LYMPH_CAT',
        'test_type': 'Ordinal',
        'threshold': None,
        'threshold_label': None,
        'threshold_test_type': None,
        'alternative': 'less' # high lymphocytes score is worse
    },
] 