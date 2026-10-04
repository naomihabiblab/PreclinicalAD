import sys
sys.path.append('../../')
from settings import *

RISK_FACTORS_TESTS = [
    {
        'label': 'Number of risk factors',
        'col': 'Number_of_risk_factors',
        'test_type': 'Ordinal',
        'threshold': None,
        'threshold_label': None,
        'threshold_test_type': None,
        'alternative': 'less' # high number of risk factors is worse
    },
    {
        'label': 'heavy_smoking',
        'col': 'heavy_smoking',
        'test_type': 'Categorical',
        'threshold': None,
        'threshold_label': None,
        'threshold_test_type': None,
        'alternative': 'less' # high smoking score is worse
    },
    # {
    #     'label': 'Insomnia',
    #     'col': 'Insomnia',
    #     'test_type': 'Categorical',
    #     'threshold': None,
    #     'threshold_label': None,
    #     'threshold_test_type': None
    # },
    {
        'label': 'severe_osa',
        'col': 'severe_osa',
        'test_type': 'Categorical',
        'threshold': None,
        'threshold_label': None,
        'threshold_test_type': None,
        'alternative': 'less' # high sleep disorder score is worse
    },
    # {
    #     'label': 'OSA',
    #     'col': 'OSA',
    #     'test_type': 'Categorical',
    #     'threshold': None,
    #     'threshold_label': None,
    #     'threshold_test_type': None
    # },
    # {
    #     'label': 'Severity of OSA',
    #     'col': 'severity of OSA',
    #     'test_type': 'Ordinal',
    #     'threshold': None,
    #     'threshold_label': None,
    #     'threshold_test_type': None
    # },
    # {
    #     'label': 'Physical activity',
    #     'col': 'Physical activity',
    #     'test_type': 'Ordinal',
    #     'threshold': None,
    #     'threshold_label': None,
    #     'threshold_test_type': None
    # },
    {
        'label': 'Herpes zoster vaccine',
        'col': 'Herpes_zoster_vaccine',
        'test_type': 'Categorical',
        'threshold': None,
        'threshold_label': None,
        'threshold_test_type': None,
        'alternative': 'greater' # high herpes zoster vaccine score is better
    },
    {
        'label': 'lipid_metaboloisim_disorder',
        'col': 'lipid_metaboloisim_disorder',
        'test_type': 'Categorical',
        'threshold': None,
        'threshold_label': None,
        'threshold_test_type': None,
        'alternative': 'less' # high lipid metabolism disorder score is worse
    },

] 