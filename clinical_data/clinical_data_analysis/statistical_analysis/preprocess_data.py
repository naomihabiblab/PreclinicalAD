import pandas as pd
import numpy as np
import sys
from settings import *
import os


# Map biomarker type to settings
BIOMARKER_SETTINGS = {
    'tau_217': {
        'col': TAU_217,
        'cat': TAU_217_CAT,
        'high_thr': HIGH_TAU_217_THR,
        'int_low_thr': INT_LOW_TAU_217_THR,
        'int_high_thr': INT_HIGH_TAU_217_THR,
        'labels': ['Low', 'Intermediate_low', 'Intermediate_high', 'High'],
        'folder': 'Tau/217'
    },
    'gfap': {
        'col': GFAP,
        'cat': GFAP_CAT,
        'high_thr': HIGH_GFAP_THR,
        'int_low_thr': INT_GFAP_THR,
        'int_high_thr': None,  # Only one intermediate threshold for GFAP
        'labels': ['Low', 'Intermediate', 'High'],
        'folder': 'GFAP'
    },
    'nfl': {
        'col': NFL,
        'cat': NFL_CAT,
        'high_thr': None,  # Age-dependent thresholds
        'int_low_thr': None,  # Age-dependent thresholds
        'int_high_thr': None,  # Age-dependent thresholds
        'labels': ['Low', 'High'],
        'folder': 'NFL'
    }
}


DATA_PATH = '/ems/elsc-labs/habib-n/yuval.rom/BCG/clinical_data/Data/fightAD_general_data_table.xlsx'
SAVE_DIR = 'Biomarkers/'


def get_nfl_threshold(age):
    """Get NFL threshold based on age group"""
    if age < 70:
        return HIGH_NFL_65_69_THR
    elif age < 75:
        return HIGH_NFL_70_74_THR
    elif age >= 75:
        return HIGH_NFL_75_80_THR


def cockcroft_gault_gfr_bsa(
    age,
    creatinine_mg_dl,
    weight_kg,
    height_cm,
    is_female,
):
    """
    Cockcroft–Gault creatinine clearance (mL/min), adjusted to 1.73 m² body surface area
    using the Du Bois BSA formula. Expects serum creatinine in mg/dL.

    CKD-EPI (or other) eGFR can remain in column ``GFR``; this is a separate estimate in
    ``real_GFR``.
    """
    age = pd.to_numeric(age, errors='coerce')
    scr = pd.to_numeric(creatinine_mg_dl, errors='coerce')
    w = pd.to_numeric(weight_kg, errors='coerce')
    h = pd.to_numeric(height_cm, errors='coerce')
    is_female = np.asarray(is_female, dtype=bool)

    crcl = ((140 - age) * w) / (72 * scr)
    crcl = np.where(is_female, crcl * 0.85, crcl)
    bsa = 0.007184 * np.power(w, 0.425) * np.power(h, 0.725)
    cg_bsa = crcl * (1.73 / bsa)

    invalid = (
        scr.isna()
        | (scr <= 0)
        | w.isna()
        | (w <= 0)
        | h.isna()
        | (h <= 0)
        | age.isna()
        | (age <= 0)
    )
    cg_bsa = pd.Series(cg_bsa, index=age.index)
    cg_bsa[invalid] = np.nan
    return cg_bsa


def blood_tests_preprocess(df):
    df['HA1C_CAT'] = pd.cut(df['Hemoglobin_A1C'], bins=[0, HA1C_INT, HA1C_HIGH, np.inf], labels=['Low', 'Intermediate', 'High'])
    df['CHOL_CAT'] = pd.cut(df['Cholesterol'],  bins=[0, CHOL_INT, CHOL_HIGH, np.inf], labels=['Low', 'Intermediate', 'High'])
    df['Triglycerides'].replace({'38..8': 38.8}, inplace=True)
    df['TRG_CAT'] = pd.cut(df['Triglycerides'], bins=[0, TRG_INT, TRG_HIGH, np.inf], labels=['Low', 'Intermediate', 'High'])
    df['LDL_CAT'] = pd.cut(df['LDL'], bins=[0, LDL_INT, LDL_HIGH, np.inf], labels=['Low', 'Intermediate', 'High'])
    df['CRP'].replace({'<0.4': 0, '<0.5': 0, '<0.6':0}, inplace=True)
    df['SBP'] = df['BP'].str.split('/').str[0].astype(float)
    df['DBP'] = df['BP'].str.split('/').str[1].astype(float)
    # df = df[df['CRP'] != '<1']
    df['CRP'].replace({'<1': np.nan}, inplace=True)
    df['CRP_CAT'] = pd.cut(df['CRP'], bins=[0, CRP_INT, CRP_HIGH, CRP_VERY_HIGH, np.inf], labels=['Low', 'Intermediate', 'High', 'Very High'])
    df['WBC_CAT'] = pd.cut(df['WBC'], bins=[0, WBC_LOW, WBC_HIGH, np.inf], labels=['Low', 'Intermediate', 'High'])
    df['LYMPH_CAT'] = pd.cut(df['Lymphocytes'], bins=[0, LYMPH_LOW, LYMPH_HIGH, np.inf], labels=['Low', 'Intermediate', 'High'])
    # # CKD-EPI eGFR stays in GFR; Cockcroft–Gault scaled to 1.73 m² BSA (requires Creatinine mg/dL, kg, cm)
    # is_female = df['Gender'].astype(str).str.lower().eq('f')
    # df['real_GFR'] = cockcroft_gault_gfr_bsa(
    #     df['Age'],
    #     df['Creatinine'],
    #     df['Weight'],
    #     df['height'],
    #     is_female,
    # )
    return df



def genetic_risk_factors(df):
    df['APOE_genotype'] = df['APOE-SNP'].replace({'APOE3/3':0, 'APO3/3':0, 'APO3/4':1, 'APOE3/4':1, 'APOE4/4':2})
    df['APOE_risk'] = df['APOE-SNP'].replace({'APOE3/3':0, 'APO3/3':0, 'APO3/4':1, 'APOE3/4':1, 'APOE4/4':1})
    df['Gender'].replace({'f':1, 'm':0}, inplace=True)
    
    return df


def risk_factors_preprocess(df):
    df["Physical activity"].replace({'NO': 0, 'LOW': 1, 'MEDIUM': 2, 'HIGH': 3}, inplace=True)
    df["Physical activity"].replace({'NO ': 0, 'LOW ': 1, 'MEDIUM ': 2, 'HIGH ': 3}, inplace=True)
    df['Smoking'].replace({'in _the_ past':'YES', 'in_the_past': 'YES'}, inplace=True)
    # # Remove rows with values in Pack_years that can't be converted to numbers
    # df = df[pd.to_numeric(df['Pack_years'], errors='coerce').notna()]
    # df['Pack_years'] = df['Pack_years'].astype(float)
    # annotate smoking with more than 10 pack years a day
    df['heavy_smoking'] = (df['Smoking'] == 'YES') & (df['Pack_years'] >= 10)
    df['Physical activity'].replace({'NO': 0, 'LOW': 1, 'MEDIUM': 2, 'HIGH': 3}, inplace=True)
    df['Diabetes'].replace({'NO': 0, 'YES': 1}, inplace=True)
    df.loc[df['Hemoglobin_A1C'] >= 6.5, 'Diabetes'] = 1
    
    # Convert Hyperlipidemia to boolean and fix type mismatch
    hyperlipidemia_bool = (df['Hyperlipidemia'] == 'YES') | (df['Hyperlipidemia'] == 1)
    df['lipid_metaboloisim_disorder'] = (
        hyperlipidemia_bool | 
        (df['HDL'] < 45) | 
        (df['LDL'] > 100) | 
        (df['Triglycerides'] > 150)
    )
    df['Hypertension'] = df['Hypertension'].replace({'NO': 0, 'YES': 1})
    df['Heart_disease'] = df['Heart_disease'].replace({'NO': 0, 'NO ': 0, 'YES': 1})
    df['BP'].fillna("0/0", inplace=True)
    df['Hypertension'] = (df['Hypertension'] == 'YES') | (df['BP'].apply(lambda x: int(x.split('/')[0]) > 140) | (df['BP'].apply(lambda x: int(x.split('/')[1]) > 90)))
    df['Hypertension'] = df['Hypertension'].astype(int)
    # Add systemic infections/inflammations conditions
    df['systemic Infections/ inflammatory disorders'] = ((df['Systemic_inflammatory_disease'] == 'YES') | 
                                                        (df['Periodontal disease'] == 'YES') | 
                                                        (df['Recurrent_UTI/_pneumonia/other_infection'] == 'YES') | 
                                                        (df['CRP'] >= CRP_INT) | 
                                                        (df['WBC'] >= 10))
    # Add sleep disorders
    df['sleep disorders'] = ((df['Insomnia'] == 'YES') | (df['OSA'] == 'YES'))
    df['Anxiety/depression'] = ((df['Anxiety/depression'] == 'YES') | (df[MBI_C_COL] >= 5))
    df['severe_osa'] = (df['OSA'] == 'YES') & (df['severity of OSA'] >= 3)
    # replace samples with severity of 1 with np.nan
    df['severe_osa'][df['severity of OSA'] == 1] = np.nan

    return df

def age_preprocess(df):
    # filter for age between AGE_LOWER_THR and AGE_UPPER_THR
    df = df[df['Age'] >= AGE_LOWER_THR]
    df = df[df['Age'] <= AGE_UPPER_THR]
    return df

def load_and_preprocess_data(biomarker_type='tau_217'):
    biomarker = BIOMARKER_SETTINGS[biomarker_type]
    df = pd.read_excel(DATA_PATH, sheet_name='1')
    df.columns = df.loc[0]
    df = df.drop(0)
    df = df.iloc[:,:-2]
    df.rename(columns={'#':'ID',
                        'תאריך בדיקות דם': 'blood_test_date',
                        'שעת לקיחת דם':'blood_taken_time',
                        'מספר מנות':'num_of_injections',
                        'גיל הפסקת מחזור':'menopause',
                        'אסטרוגן':'estorgen',
                        'Frontal memory score (#1)- כמה מילים זכר/ה ברמז הראשון. נקודה אם זכר/ה רמז ראשון ושתי נקודות אם לא.':'Frontal memory score (#1)',
                        'Hipocampal memory score (#2)- כמה מילים לא זכר/ה בכלל':'Hipocampal memory score (#2)',
                        'p-tau_ new_#1':'p-tau_new_#1',
                        'ביקורת אחרי שנה':'checkup after a year',
                        'חיסון':'vaccine',
                        'תרופות':'medications',
                        'הערות':'notes',
                        'Number_of_risk_faktors':'Number_of_risk_factors',
                        'P-tau217':'p-tau217',
                        'צרבונין':'Cerebonin',
                        'Spatial ': 'Spatial'
                        }, inplace=True)
    df = df.apply(pd.to_numeric, errors='ignore')
    df = df.drop(df[df['ID'].isin(OUTLIERS)].index)
    df = df[df[MOCA_COL] != '?']

    df['MCI'] = df[MOCA_COL] <= MCI_THR
    df = df[df[MOCA_COL].notna()]
    df = df[df[MOCA_COL] > MIN_MOCA_SCORE] # Removing samples with extremlly low MOCA (19)
    print(f"After MOCA preprocess: {df.shape}")
    df = age_preprocess(df)
    print(f"After age preprocess: {df.shape}")
    df = blood_tests_preprocess(df)
    print(f"After blood tests preprocess: {df.shape}")
    df = risk_factors_preprocess(df)
    print(f"After risk factors preprocess: {df.shape}")
    df = genetic_risk_factors(df)
    print(f"After genetic risk factors preprocess: {df.shape}")

    # Biomarker categorization
    if biomarker_type == 'tau_217':
        df[biomarker['cat']] = pd.cut(df[biomarker['col']], bins=[0, biomarker['int_low_thr'], biomarker['int_high_thr'], biomarker['high_thr'], np.inf], labels=biomarker['labels'])
        df = df[df[biomarker['col']].notna()]
        df_high_vs_low = df[(df[biomarker['col']] > biomarker['high_thr']) | (df[biomarker['col']] < biomarker['int_low_thr'])]
        df_high_vs_low[biomarker['cat']] = df_high_vs_low[biomarker['cat']].cat.remove_categories(['Intermediate_low', 'Intermediate_high'])
    
    elif biomarker_type == 'gfap':
        df[biomarker['cat']] = pd.cut(df[biomarker['col']], bins=[0, biomarker['int_low_thr'], biomarker['high_thr'], np.inf], labels=biomarker['labels'])
        df = df[df[biomarker['col']].notna()]

        df_high_vs_low = df[(df[biomarker['col']] > biomarker['high_thr']) | (df[biomarker['col']] < biomarker['int_low_thr'])]
        df_high_vs_low[biomarker['cat']] = df_high_vs_low[biomarker['cat']].cat.remove_categories(['Intermediate'])
    
    elif biomarker_type == 'nfl':
        # NFL has age-dependent thresholds
        df = df[df[biomarker['col']].notna()]
        df = df[df['Age'].notna()]  # Need age for NFL thresholds
        
        # Create age-dependent categories
        df[biomarker['cat']] = 'Low'  # Default to Low
        for idx, row in df.iterrows():
            age = row['Age']
            nfl_value = row[biomarker['col']]
            threshold = get_nfl_threshold(age)
            if nfl_value > threshold:
                df.loc[idx, biomarker['cat']] = 'High'
        
        # For NFL, we only have Low vs High categories
        df_high_vs_low = df.copy()
        df_high_vs_low[biomarker['cat']] = df_high_vs_low[biomarker['cat']].astype('category')

    return df_high_vs_low, biomarker['cat'], biomarker['folder']


if __name__ == '__main__':
    biomarker_type = 'tau_217'
    if len(sys.argv) > 1:
        biomarker_type = sys.argv[1].lower()
        if biomarker_type not in BIOMARKER_SETTINGS:
            print(f"Unknown biomarker type '{biomarker_type}'. Using default 'tau_217'.")
            print(f"Available options: {list(BIOMARKER_SETTINGS.keys())}")
            biomarker_type = 'tau_217'

    df, biomarker_cat, biomarker_folder = load_and_preprocess_data(biomarker_type)
    print(df.shape)
    print(df.value_counts(subset=[biomarker_cat]))
    save_dir = biomarker_folder
    os.makedirs(save_dir, exist_ok=True)
    df.to_csv(os.path.join(save_dir, 'high_vs_low_data.csv'), index=False)