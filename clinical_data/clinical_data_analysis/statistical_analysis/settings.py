# Graphs Settings
healthy_color = '#b4cde3' # blue
intermediate_low_cplor = '#CBC3E3' # purple
intermediate_high_color = '#fed9a6' # orange
risk_color = '#fbb4ae' # red
pie_colors = [healthy_color, intermediate_low_cplor, intermediate_high_color, risk_color] # blue, green, orange, red
graph_colors = [healthy_color, risk_color] # blue, red
vline_color = 'red'

AGE_LOWER_THR = 65
AGE_UPPER_THR = 80

TAU_217 = 'p-tau217'
TAU_217_CAT = 'tau_217_category'
TAU_217_POPULATION_THR = 0.444
# HIGH_TAU_217_THR = 0.53
HIGH_TAU_217_THR = 0.444
# INT_HIGH_TAU_217_THR = 0.449
# INT_HIGH_TAU_217_THR = 0.445
INT_HIGH_TAU_217_THR = 0.4439
# INT_LOW_TAU_217_THR = 0.395
INT_LOW_TAU_217_THR = 0.4438

GFAP = 'GFAP'
GFAP_CAT = 'gfap_category'
GFAP_POPULATION_THR = 166
HIGH_GFAP_THR = 230
INT_GFAP_THR = 166

NFL = 'NFL'
NFL_CAT = 'nfl_category'
HIGH_NFL_65_69_THR = 11.9
HIGH_NFL_70_74_THR = 15.3
HIGH_NFL_75_80_THR = 17.7

TMT_B_COL = 'Trail_B_(second)'
MBI_C_COL = 'Mild_behavioral_Impairment_Checklist'
MBI_C_THR = 7
IST_COL = 'IST'
ANEXITY_COL = 'Anxiety/depression'

MCI_THR = 26
MOCA_COL = 'MOCA_score'
MCI = 'MCI'
MIN_MOCA_SCORE = 19

# Participant exclusions are maintained in the private analysis configuration.
OUTLIERS = []

# Blood tests
HA1C_INT, HA1C_HIGH = 5.7, 6.4
CHOL_INT, CHOL_HIGH = 180, 240
TRG_INT, TRG_HIGH = 150, 200
LDL_INT, LDL_HIGH = 100, 130
CRP_INT, CRP_HIGH, CRP_VERY_HIGH = 0.5, 1, 2
WBC_LOW, WBC_HIGH = 4.5, 11
LYMPH_LOW, LYMPH_HIGH = 1.3, 4.7

DEMOGRAPHIC_COLS = ['Age', 'Gender', 'Years_of_education', 'Family_status', 'Country_of_origin', 'height', 'weight', 'BMI']
RISK_FACTOR_COLS = ['Heart_disease', 'Diabetes', 'Hypertension', 'Hyperlipidemia', 'Periodontal disease', 'Menopause age', 'ESTROGEN'
                    'Recurrent_UTI/_pneumonia/other_infection', 'Systemic_inflammatory_disease', 'Smoking', 'Pack_years', 
                    'Alcohol_consumption', 'Insomnia', 'OSA', 'Trearment_for_OSA', 'severity of OSA', 'Physical activity', 
                    'Time_Physical_activity_(hours/week)', 'Anxiety/depression', 'Cognitive_activity', 'AD_in_the_family', 
                    'Herpes_zoster_vaccine', 'Number_of_risk_factors']
COGNITIVE_COLS = ['MOCA_score', 'Executive', 'Spatial', 'Attention_and_concentration', 'Memory', 'Language', 'Orientation', 
                  'Frontal memory score (#1)', 'Hipocampal memory score (#2)', 'Trail_B_(second)', 'IST']
BEHAVIORAL_COLS = ['Mild_behavioral_Impairment_Checklist', 'interest, motivation, and drive', 'mood or anxiety symptoms', 
                   'Impulse control', 'following societal norms', 'strongly held beliefs and sensory', 'iADL']
BLOOD_TEST_COLS = ['Hemoglobin_A1C', 'Cholesterol', 'Triglycerides', 'LDL', 'HDL', 'CRP', 'HB', 'RBC', 'WBC', 'Lymphocytes', 
                   'Creatinine', 'GFR']
PLASMA_COLS = ['p-tau217', 'GFAP', 'NFL']
GENETIC_COLS = ['APOE-SNP']

