import sys
sys.path.append('../../')
from settings import *

COGNITIVE_TESTS = [
    {
        'label': 'MOCA score',
        'col': MOCA_COL,
        'test_type': 'Continuous',
        'threshold': None,
        'threshold_label': None,
        'threshold_test_type': None,
        'alternative': 'greater' # high moca score is better
    },
    {
        'label': 'TMT (Trail B)',
        'col': TMT_B_COL,
        'test_type': 'Continuous',
        'threshold': None,
        'threshold_label': None,
        'threshold_test_type': None,
        'alternative': 'less' # high tmt score is worse
    },
    {
        'label': 'IST',
        'col': IST_COL,
        'test_type': 'Continuous',
        'threshold': None,
        'threshold_label': None,
        'threshold_test_type': None,
        'alternative': 'greater' # high ist score is better
    },

] 