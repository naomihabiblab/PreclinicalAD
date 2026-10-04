import sys

sys.path.append("../../")
from settings import *  # noqa: F401,F403

SYSTEMIC_FACTORS_TESTS = [
    {
        "label": "SBP",
        "col": "SBP",
        "test_type": "Continuous",
        "threshold": None,
        "threshold_label": None,
        "threshold_test_type": None,
        "alternative": "less",  # high sbp score is worse
    },
    {
        "label": "DBP",
        "col": "DBP",
        "test_type": "Continuous",
        "threshold": None,
        "threshold_label": None,
        "threshold_test_type": None,
        "alternative": "less",  # high dbp score is worse
    },
    {
        "label": "HB",
        "col": "HB",
        "test_type": "Continuous",
        "threshold": None,
        "threshold_label": None,
        "threshold_test_type": None,
        "alternative": "greater",  # high hb score is better
    },
    {
        "label": "GFR",
        "col": "GFR",
        "test_type": "Continuous",
        "threshold": None,
        "threshold_label": None,
        "threshold_test_type": None,
        "alternative": "greater",  # high gfr score is better
    },
    {
        "label": "Creatinine",
        "col": "Creatinine",
        "test_type": "Continuous",
        "threshold": None,
        "threshold_label": None,
        "threshold_test_type": None,
        "alternative": "less",  # high creatinine score is worse
    },
    {
        "label": "RBC",
        "col": "RBC",
        "test_type": "Continuous",
        "threshold": None,
        "threshold_label": None,
        "threshold_test_type": None,
        "alternative": "greater",  # high rbc score is better
    },
    {
        "label": "WBC",
        "col": "WBC",
        "test_type": "Continuous",
        "threshold": None,
        "threshold_label": None,
        "threshold_test_type": None,
        "alternative": "less",  # high wbc score is worse
    },
    {
        "label": "Lymphocytes",
        "col": "Lymphocytes",
        "test_type": "Continuous",
        "threshold": None,
        "threshold_label": None,
        "threshold_test_type": None,
        "alternative": "less",  # high lymphocytes score is worse
    },    
    {
        'label': 'CRP',
        'col': 'CRP',
        'test_type': 'Continuous',
        'threshold': None,
        'threshold_label': None,
        'threshold_test_type': None,
        'alternative': 'less' # high crp score is worse
    },
]

