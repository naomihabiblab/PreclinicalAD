import sys
sys.path.append('../../')
from BCG.clinical_data.clinical_data_analysis.statistical_analysis.settings import *

COGNITIVE_TESTS = [
    {
        'label': 'Attention and concentration',
        'col': 'Attention_and_concentration',
        'test_type': 'Ordinal',
        'threshold': None,
        'threshold_label': None,
        'threshold_test_type': None
    },
    {
        'label': 'Executive',
        'col': 'Executive',
        'test_type': 'Ordinal',
        'threshold': None,
        'threshold_label': None,
        'threshold_test_type': None
    },
    {
        'label': 'Spatial',
        'col': 'Spatial',
        'test_type': 'Ordinal',
        'threshold': None,
        'threshold_label': None,
        'threshold_test_type': None
    },
    {
        'label': 'Memory',
        'col': 'Memory',
        'test_type': 'Ordinal',
        'threshold': None,
        'threshold_label': None,
        'threshold_test_type': None
    },
    {
        'label': 'Language',
        'col': 'Language',
        'test_type': 'Ordinal',
        'threshold': None,
        'threshold_label': None,
        'threshold_test_type': None
    },
    {
        'label': 'Orientation',
        'col': 'Orientation',
        'test_type': 'Ordinal',
        'threshold': None,
        'threshold_label': None,
        'threshold_test_type': None
    },
    {
        'label': 'Frontal memory score (#1)',
        'col': 'Frontal memory score (#1)',
        'test_type': 'Ordinal',
        'threshold': None,
        'threshold_label': None,
        'threshold_test_type': None
    },
    {
        'label': 'Hipocampal memory score (#2)',
        'col': 'Hipocampal memory score (#2)',
        'test_type': 'Ordinal',
        'threshold': None,
        'threshold_label': None,
        'threshold_test_type': None
    },
    {
        'label': 'MOCA score',
        'col': MOCA_COL,
        'test_type': 'Continuous',
        'threshold': None,
        'threshold_label': None,
        'threshold_test_type': None
    },
    {
        'label': 'TMT (Trail B)',
        'col': TMT_B_COL,
        'test_type': 'Continuous',
        'threshold': None,
        'threshold_label': None,
        'threshold_test_type': None
    },
    {
        'label': 'IST',
        'col': IST_COL,
        'test_type': 'Continuous',
        'threshold': None,
        'threshold_label': None,
        'threshold_test_type': None
    },

] 