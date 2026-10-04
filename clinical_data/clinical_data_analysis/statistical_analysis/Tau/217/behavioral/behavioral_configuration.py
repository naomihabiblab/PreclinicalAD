import sys
sys.path.append('../../')
from BCG.clinical_data.clinical_data_analysis.statistical_analysis.settings import *

BEHAVIORAL_TESTS = [
    # {
    #     'label': 'TMT (Trail B)',
    #     'col': TMT_B_COL,
    #     'test_type': 'Continuous',
    #     'threshold': None,
    #     'threshold_label': None,
    #     'threshold_test_type': None
    # },
    {
        'label': 'MBI-C',
        'col': MBI_C_COL,
        'test_type': 'Ordinal',
        'threshold': 6,
        'threshold_label': 'MBI-C Threshold',
        'threshold_test_type': 'Categorical'
    },
    # {
    #     'label': 'IST',
    #     'col': IST_COL,
    #     'test_type': 'Continuous',
    #     'threshold': None,
    #     'threshold_label': None,
    #     'threshold_test_type': None
    # },
    {
        'label': 'Interest, Motivation, and Drive',
        'col': 'interest, motivation, and drive',
        'test_type': 'Ordinal',
        'threshold': 2,
        'threshold_label': 'Interest, Motivation, and Drive Threshold',
        'threshold_test_type': 'Categorical'
    },
    {
        'label': 'Impulse Control',
        'col': 'Impulse control',
        'test_type': 'Ordinal',
        'threshold': 2,
        'threshold_label': 'Impulse Control Threshold',
        'threshold_test_type': 'Categorical'
    },
    {
        'label': 'Following Societal Norms',
        'col': 'following societal norms',
        'test_type': 'Ordinal',
        'threshold': 0,
        'threshold_label': 'Following Societal Norms Threshold',
        'threshold_test_type': 'Categorical'
    },
    {
        'label': 'Strongly Held Beliefs and Sensory',
        'col': 'strongly held beliefs and sensory',
        'test_type': 'Ordinal',
        'threshold': 0,
        'threshold_label': 'Strongly Held Beliefs Threshold',
        'threshold_test_type': 'Categorical'
    },
    {
        'label': 'iADL',
        'col': 'iADL',
        'test_type': 'Ordinal',
        'threshold': None,
        'threshold_label': None,
        'threshold_test_type': None
    },
]