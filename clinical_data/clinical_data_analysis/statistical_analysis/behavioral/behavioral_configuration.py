import sys
sys.path.append('../../')
from settings import *

BEHAVIORAL_TESTS = [
    {
        'label': 'MBI-C',
        'col': MBI_C_COL,
        'test_type': 'Ordinal',
        'threshold': 6,
        'threshold_label': 'MBI-C Threshold',
        'threshold_test_type': 'Categorical',
        'alternative': 'less',  # high mbi-c score is worse
        'threshold_alternative': 'less'  # high mbi-c score is worse
    },
    {
        'label': 'Interest, Motivation, and Drive',
        'col': 'interest, motivation, and drive',
        'test_type': 'Ordinal',
        'threshold': 2,
        'threshold_label': 'Interest, Motivation, and Drive Threshold',
        'threshold_test_type': 'Categorical',
        'alternative': 'less',  # high interest, motivation, and drive score is worse
        'threshold_alternative': 'less'  # high interest, motivation, and drive score is worse
    },
    {
        'label': 'Impulse Control',
        'col': 'Impulse control',
        'test_type': 'Ordinal',
        'threshold': 2,
        'threshold_label': 'Impulse Control Threshold',
        'threshold_test_type': 'Categorical',
        'alternative': 'less',  # high impulse control score is worse
        'threshold_alternative': 'less'  # high impulse control score is worse
    },
    {
        'label': 'Following Societal Norms',
        'col': 'following societal norms',
        'test_type': 'Ordinal',
        'threshold': 0,
        'threshold_label': 'Following Societal Norms Threshold',
        'threshold_test_type': 'Categorical',
        'alternative': 'less',  # high following societal norms score is worse
        'threshold_alternative': 'less'  # high following societal norms score is worse
    },
    {
        'label': 'Strongly Held Beliefs and Sensory',
        'col': 'strongly held beliefs and sensory',
        'test_type': 'Ordinal',
        'threshold': 0,
        'threshold_label': 'Strongly Held Beliefs Threshold',
        'threshold_test_type': 'Categorical',
        'alternative': 'less',  # high strongly held beliefs and sensory score is worse
        'threshold_alternative': 'less'  # high strongly held beliefs and sensory score is worse
    },
    {
        'label': 'iADL',
        'col': 'iADL',
        'test_type': 'Ordinal',
        'threshold': None,
        'threshold_label': None,
        'threshold_test_type': None,
        'alternative': 'less',  # high iadl score is worse
        'threshold_alternative': 'less'  # high iadl score is worse
    },
    {
        'label': 'mood or anxiety symptoms',
        'col': 'mood or anxiety symptoms',
        'test_type': 'Ordinal',
        'threshold': None,
        'threshold_label': None,
        'threshold_test_type': None,
        'alternative': 'less',  # high mood or anxiety symptoms score is worse
        'threshold_alternative': 'less'  # high mood or anxiety symptoms score is worse
    },
]