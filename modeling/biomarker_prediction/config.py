"""
Configuration constants for tau-217 classification pipeline.
All constants and configuration parameters are centralized here.
"""

# Participant exclusions are maintained in the private analysis configuration.
KIDNEY_OUTLIERS_IDS = []
OUTLIERS_IDS = []

# Tau-217 thresholds
HIGH_TAU_217_THRESHOLD = 0.445
INTR_TAU_217_THRESHOLD = 0.444
TAU_217 = 'p-tau217'

# MOCA score constants for baseline classifier
MOCA_COL = 'MOCA_score'
MCI_THR = 26

# Feature column definitions
COGNITIVE_COLUMNS = [
    'Trail_B_(second)', 'IST', 'MOCA_score', 'Executive', 'Spatial ',
    'Attention_and_concentration', 'Memory', 'Language', 'Orientation',
    'Frontal memory score (#1)', 'Hipocampal memory score (#2)'
]

BEHVIOURAL_COLUMNS = [
    'Mild_behavioral_Impairment_Checklist',
    'interest, motivation, and drive', 'mood or anxiety symptoms',
    'Impulse control', 'following societal norms',
    'strongly held beliefs and sensory', 'iADL'
]

AGE_GENDER_COLUMNS = ['Age', 'Gender']
HIGHT_WIGHT_COLUMNS = ['height', 'Weight']
DEMOGRAPHIC_COLUMNS = ['Years_of_education']

RISK_FACTORS_COLUMNS = [
    'Heart_disease', 'Diabetes', 'Hypertension', 'Hyperlipidemia',
    'Periodontal disease', 'Recurrent_UTI/_pneumonia/other_infection', 'Herpes_zoster_vaccine',
    'Systemic_inflammatory_disease', 'Smoking', 'Pack_years',
    'Alcohol_consumption', 'Insomnia', 'OSA', 'Trearment_for_OSA',
    'Physical activity', 'Time_Physical_activity_(hours/week)',
    'Anxiety/depression', 'Cognitive_activity', 'AD_in_the_family',
    'Number_of_risk_factors', 'Menopause age', 'sleep disorders',
    'systemic Infections/ inflammatory disorders', 'lipid_metaboloisim_disorder'
]

GENETIC_RISK_FACTORS_COLUMNS = ['APOE']

BLOOD_TESTS = [
    'Hemoglobin_A1C', 'Cholesterol', 'Triglycerides', 'LDL', 'HDL', 'CRP',
    'HB', 'RBC', 'WBC', 'Lymphocytes', 'GFR', 'Creatinine'
]

# Combined informative features for classification
INFORMATIVE_FEATURES = (
    COGNITIVE_COLUMNS +
    ['Herpes_zoster_vaccine', 'BMI', 'Diabetes_controlled', 'Diabetes', 'Hyperlipidemia', 'Smoking'] +
    # RISK_FACTORS_COLUMNS +
    BEHVIOURAL_COLUMNS +
    AGE_GENDER_COLUMNS +
    BLOOD_TESTS
)


# Hyperparameter optimization settings
N_TRIALS = 15  # Number of Optuna trials for HPO

# Data split settings
TRAIN_TEST_SPLIT = 0.3
VAL_TEST_SPLIT = 0.5  # Applied to the test portion from train_test_split
RANDOM_STATE = 42

