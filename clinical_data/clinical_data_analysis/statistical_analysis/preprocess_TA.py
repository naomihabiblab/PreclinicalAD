import pandas as pd
import numpy as np
import sys
from settings import *
import os

# Map tau type to settings
BIOMARKERS_SETTINGS = {
    'tau_217': {
        'col': TAU_217,
        'folder': 'Tau/217'
    },
    'gfap': {
        'col': 'GFAP',
        'folder': 'GFAP/'
    },
    'nfl': {
        'col': 'NFL',
        'folder': 'NFL/'
    }
}

DATA_PATH = '/ems/elsc-labs/habib-n/yuval.rom/BCG/clinical_data/Data/fightAD_general_data_table.xlsx'
SAVE_DIR = 'Tau/'

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
    df['Smoking'].replace({'in _the_ past':np.nan, 'in_the_past': np.nan}, inplace=True)
    # annotate smoking with more than 10 pack years a day
    df['heavy_smoking'] = (df['Smoking'] == 'YES') & (df['Pack_years'] >= 10)
    df['Physical activity'].replace({'NO': 0, 'LOW': 1, 'MEDIUM': 2, 'HIGH': 3}, inplace=True)
    df['Diabetes'].replace({'NO': 0, 'YES': 1}, inplace=True)
    df.loc[df['Hemoglobin_A1C'] >= 6.1, 'Diabetes'] = 1
    
    # Convert Hyperlipidemia to boolean and fix type mismatch
    hyperlipidemia_bool = (df['Hyperlipidemia'] == 'YES') | (df['Hyperlipidemia'] == 1)
    df['lipid_metaboloisim_disorder'] = (
        hyperlipidemia_bool | 
        (df['HDL'] < 45) | 
        (df['LDL'] > 100) | 
        (df['Triglycerides'] > 150)
    )
    df['Hypertension'] = df['Hypertension'].replace({'NO': 0, 'YES': 1})
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

def load_and_preprocess_data(biomarker='tau_217'):
    tau = BIOMARKERS_SETTINGS[biomarker]
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

    df['MCI'] = df[MOCA_COL] < MCI_THR
    df = df[df[MOCA_COL].notna()]
    df = df[df[MOCA_COL] > MIN_MOCA_SCORE] # Removing samples with extremlly low MOCA (19)

    df = age_preprocess(df)
    df = blood_tests_preprocess(df)
    df = risk_factors_preprocess(df)
    df = genetic_risk_factors(df)

    # Only filter for notna on tau biomarker
    df = df[df[tau['col']].notna()]

    return df, tau['folder']

if __name__ == '__main__':
    biomarker = 'tau_217'
    if len(sys.argv) > 1:
        biomarker = sys.argv[1].lower()
        if biomarker not in BIOMARKERS_SETTINGS:
            print(f"Unknown biomarker '{biomarker}'. Supported: {list(BIOMARKERS_SETTINGS.keys())}. Using default 'tau_217'.")
            biomarker = 'tau_217'

    df, biomarker_folder = load_and_preprocess_data(biomarker)
    print(df.shape)
    save_dir = biomarker_folder
    os.makedirs(save_dir, exist_ok=True)
    df.to_csv(os.path.join(save_dir, 'trait_association_data.csv'), index=False) 