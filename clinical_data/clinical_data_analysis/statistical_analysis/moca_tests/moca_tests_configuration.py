import sys
sys.path.append('../../')
from settings import *

MOCA_TESTS = [    
    
    {
        'label': 'Attention and concentration',
        'col': 'Attention_and_concentration',
        'test_type': 'Ordinal',
        'threshold': None,
        'threshold_label': None,
        'threshold_test_type': None,
        'alternative': 'greater' # high attention and concentration score is better
    },
    {
        'label': 'Executive',
        'col': 'Executive',
        'test_type': 'Ordinal',
        'threshold': None,
        'threshold_label': None,
        'threshold_test_type': None,
        'alternative': 'greater' # high executive score is better
    },
    {
        'label': 'Spatial',
        'col': 'Spatial',
        'test_type': 'Ordinal',
        'threshold': None,
        'threshold_label': None,
        'threshold_test_type': None,
        'alternative': 'greater' # high spatial score is better
    },
    {
        'label': 'Memory',
        'col': 'Memory',
        'test_type': 'Ordinal',
        'threshold': None,
        'threshold_label': None,
        'threshold_test_type': None,
        'alternative': 'greater' # high memory score is better
    },
    {
        'label': 'Language',
        'col': 'Language',
        'test_type': 'Ordinal',
        'threshold': None,
        'threshold_label': None,
        'threshold_test_type': None,
        'alternative': 'greater' # high language score is better
    },
    {
        'label': 'Orientation',
        'col': 'Orientation',
        'test_type': 'Ordinal',
        'threshold': None,
        'threshold_label': None,
        'threshold_test_type': None,
        'alternative': 'greater' # high orientation score is better
    },
    {
        'label': 'Frontal memory score (#1)',
        'col': 'Frontal memory score (#1)',
        'test_type': 'Ordinal',
        'threshold': None,
        'threshold_label': None,
        'threshold_test_type': None,
        'alternative': 'less' # high frontal memory score (#1) is worse 
    },
    {
        'label': 'Hipocampal memory score (#2)',
        'col': 'Hipocampal memory score (#2)',
        'test_type': 'Ordinal',
        'threshold': None,
        'threshold_label': None,
        'threshold_test_type': None,
        'alternative': 'less' # high hipocampal memory score (#2) is worse
    },
] 
