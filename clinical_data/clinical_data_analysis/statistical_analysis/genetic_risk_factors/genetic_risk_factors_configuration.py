import sys
sys.path.append('../../')
from settings import *


GENETIC_RISK_FACTORS_TESTS = [
    {
        'label': 'APOE_genotype',
        'col': 'APOE_genotype',
        'test_type': 'Ordinal',
        'threshold': None,
        'threshold_label': None,
        'threshold_test_type': None,
        'alternative': 'less' # high aPOE_genotype score is worse
    },
    {
        'label': 'APOE_risk',
        'col': 'APOE_risk',
        'test_type': 'Categorical',
        'threshold': None,
        'threshold_label': None,
        'threshold_test_type': None,
        'alternative': 'less' # high aPOE_risk score is worse
    },
    
] 