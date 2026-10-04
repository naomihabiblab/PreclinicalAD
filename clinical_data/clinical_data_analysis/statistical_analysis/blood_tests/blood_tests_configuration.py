import sys
sys.path.append('../../')
from settings import *

BLOOD_TESTS = [
    {
        'label': 'Hemoglobin A1C',
        'col': 'Hemoglobin_A1C',
        'test_type': 'Continuous',
        'threshold': None,
        'threshold_label': None,
        'threshold_test_type': None,
        'alternative': 'less' # high hemoglobin a1c score is worse
    },
    {
        'label': 'Cholesterol',
        'col': 'Cholesterol',
        'test_type': 'Continuous',
        'threshold': None,
        'threshold_label': None,
        'threshold_test_type': None,
        'alternative': 'less' # high cholesterol score is worse
    },
    {
        'label': 'Triglycerides',
        'col': 'Triglycerides',
        'test_type': 'Continuous',
        'threshold': None,
        'threshold_label': None,
        'threshold_test_type': None,
        'alternative': 'less' # high triglycerides score is worse
    },
    {
        'label': 'LDL',
        'col': 'LDL',
        'test_type': 'Continuous',
        'threshold': None,
        'threshold_label': None,
        'threshold_test_type': None,
        'alternative': 'less' # high ldl score is worse
    },
    {
        'label': 'HDL',
        'col': 'HDL',
        'test_type': 'Continuous',
        'threshold': None,
        'threshold_label': None,
        'threshold_test_type': None,
        'alternative': 'greater' # high hdl score is better
    },
    # {
    #     'label': 'real_GFR',
    #     'col': 'real_GFR',
    #     'test_type': 'Continuous',
    #     'threshold': None,
    #     'threshold_label': None,
    #     'threshold_test_type': None,
    #     'alternative': 'greater' # high real_gfr score is better
    # },
] 