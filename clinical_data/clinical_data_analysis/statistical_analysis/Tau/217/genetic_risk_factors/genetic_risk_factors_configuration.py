import sys
sys.path.append('../../')
from BCG.clinical_data.clinical_data_analysis.statistical_analysis.settings import *

GENETIC_RISK_FACTORS_TESTS = [
    {
        'label': 'APOE_genotype',
        'col': 'APOE_genotype',
        'test_type': 'Ordinal',
        'threshold': None,
        'threshold_label': None,
        'threshold_test_type': None
    },
    {
        'label': 'APOE_risk',
        'col': 'APOE_risk',
        'test_type': 'Categorical',
        'threshold': None,
        'threshold_label': None,
        'threshold_test_type': None
    },
    
] 