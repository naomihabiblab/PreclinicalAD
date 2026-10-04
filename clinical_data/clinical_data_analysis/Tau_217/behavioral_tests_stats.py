import pandas as pd
import numpy as np
import sys
sys.path.append('../')
from BCG.clinical_data.clinical_data_analysis.statistical_analysis.settings import *
from utilis import *
import statsmodels.stats.multitest as smm
import csv

# Data loading and cleaning
DATA_PATH = '/ems/elsc-labs/habib-n/yuval.rom/BCG/clinical_data/Data/fightAD_general_data_table.xlsx'
SAVE_DIR = 'stats/behavioral/'
OUTPUT_FILE = SAVE_DIR + 'behavioral_tests_stats.txt'

def load_and_preprocess_data():
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
    print(df.shape)
    df = df[df[MOCA_COL] != '?']
    df['MCI'] = df[MOCA_COL] < MCI_THR
    df[TAU_CAT] = pd.cut(df[TAU_217], bins=[0, INT_LOW_TAU_217_THR, INT_HIGH_TAU_217_THR, HIGH_TAU_217_THR, np.inf], labels=['Low', 'Intermediate_low', 'Intermediate_high', 'High'])
    df = df[df[TAU_217].notna()]
    df = df[df[MOCA_COL] > MIN_MOCA_SCORE]

    df_high_vs_low = df[(df[TAU_217] > HIGH_TAU_217_THR) | (df[TAU_217] < INT_LOW_TAU_217_THR)]
    df_high_vs_low[TAU_CAT] = df_high_vs_low[TAU_CAT].cat.remove_categories(['Intermediate_low', 'Intermediate_high'])
    
    return df_high_vs_low

BEHAVIORAL_TESTS = [
    {
        'label': 'TMT (Trail B)',
        'col': TMT_B_COL,
        'test_type': 'Continuous',
        'threshold': None,
        'threshold_label': None,
        'threshold_test_type': None
    },
    {
        'label': 'MBI-C',
        'col': MBI_C_COL,
        'test_type': 'Ordinal',
        'threshold': 6,
        'threshold_label': 'MBI-C Threshold',
        'threshold_test_type': 'Categorical'
    },
    {
        'label': 'IST',
        'col': IST_COL,
        'test_type': 'Continuous',
        'threshold': None,
        'threshold_label': None,
        'threshold_test_type': None
    },
    {
        'label': 'Interest, Motivation, and Drive',
        'col': 'interest, motivation, and drive',
        'test_type': 'Ordinal',
        'threshold': 2,
        'threshold_label': 'Interest, Motivation, and Drive Threshold',
        'threshold_test_type': 'Categorical'
    },
    {
        'label': 'Impulse Control',
        'col': 'Impulse control',
        'test_type': 'Ordinal',
        'threshold': 2,
        'threshold_label': 'Impulse Control Threshold',
        'threshold_test_type': 'Categorical'
    },
    {
        'label': 'Following Societal Norms',
        'col': 'following societal norms',
        'test_type': 'Ordinal',
        'threshold': 0,
        'threshold_label': 'Following Societal Norms Threshold',
        'threshold_test_type': 'Categorical'
    },
    {
        'label': 'Strongly Held Beliefs and Sensory',
        'col': 'strongly held beliefs and sensory',
        'test_type': 'Ordinal',
        'threshold': 0,
        'threshold_label': 'Strongly Held Beliefs Threshold',
        'threshold_test_type': 'Categorical'
    },
    {
        'label': 'iADL',
        'col': 'iADL',
        'test_type': 'Ordinal',
        'threshold': None,
        'threshold_label': None,
        'threshold_test_type': None
    },
]

def run_behavioral_tests(df_high_vs_low, behavioral_tests):
    output_lines = []
    p_values = []
    risk_factors = []
    threshold_p_values = []
    threshold_risk_factors = []
    for test in behavioral_tests:
        print(f'Running {test["label"]} test...')
        col = test['col']
        label = test['label']
        test_type = test['test_type']
        threshold = test['threshold']
        threshold_label = test['threshold_label']
        threshold_test_type = test['threshold_test_type']

        df_test = df_high_vs_low.dropna(subset=[col]).copy()
        try:
            df_test[col] = df_test[col].astype(float)
        except Exception:
            print(f'{col} is not a number')
            pass
        group1 = df_test[df_test[TAU_CAT] == 'Low'][col]
        group2 = df_test[df_test[TAU_CAT] == 'High'][col]
        p = stat_test(test_type, group1, group2)
        output_lines.append(f'{label}: p-value = {p:.5f}')
        # output_lines.append(f'{label}: p-value = {p}')
        p_values.append(p)
        risk_factors.append(label)

        # Thresholded test
        if threshold is not None and threshold_label is not None and threshold_test_type is not None:
            thr_col = f"{col}_THR"
            df_test[thr_col] = (df_test[col] > threshold)
            group1_thr = df_test
            group2_thr = [TAU_CAT, thr_col]
            p_thr = stat_test(threshold_test_type, group1_thr, group2_thr)
            # output_lines.append(f'{threshold_label}: p-value = {p_thr:.5f}')
            output_lines.append(f'{threshold_label}: p-value = {p_thr}')
            threshold_p_values.append(p_thr)
            threshold_risk_factors.append(threshold_label)
    return output_lines, p_values, risk_factors, threshold_p_values, threshold_risk_factors

def correct_multiple_testing(p_values, risk_factors, alpha=0.05, method='fdr_bh'):
    reject, p_values_corrected, _, _ = smm.multipletests(p_values, alpha=alpha, method=method)
    lines = ['\nMultiple testing correction (BH-FDR):']
    csv_rows = []
    for i, p in enumerate(p_values):
        lines.append(f'{risk_factors[i]}: p-value = {p:.5f} -> adjusted p-value = {p_values_corrected[i]:.5f}')
        csv_rows.append([risk_factors[i], f'{p:.5f}', f'{p_values_corrected[i]:.5f}'])
    lines.append("\nSignificant features:")
    for i, p in enumerate(p_values):
        if p < alpha:
            lines.append(f'{risk_factors[i]}: p-value = {p:.5f} -> adjusted p-value = {p_values_corrected[i]:.5f}')
    return lines, csv_rows

def save_results(output_lines, output_file):
    with open(output_file, 'w') as f:
        for line in output_lines:
            f.write(line + '\n')

def save_csv(csv_rows, csv_file):
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Feature', 'p-value', 'adjusted p-value'])
        writer.writerows(csv_rows)

def main():
    df_high_vs_low = load_and_preprocess_data()
    print(df_high_vs_low.value_counts(subset=[TAU_CAT]))
    output_lines, p_values, risk_factors, threshold_p_values, threshold_risk_factors = run_behavioral_tests(df_high_vs_low, BEHAVIORAL_TESTS)
    all_p_values = p_values + threshold_p_values
    all_risk_factors = risk_factors + threshold_risk_factors
    all_p_values = [p for p in all_p_values if p is not None]
    correction_lines, csv_rows = correct_multiple_testing(all_p_values, all_risk_factors)
    output_lines.extend(correction_lines)
    save_results(output_lines, OUTPUT_FILE)
    save_csv(csv_rows, SAVE_DIR + 'behavioral_tests_stats.csv')

if __name__ == '__main__':
    print(f'######## Running behavioral tests with p-tau threshold = {HIGH_TAU_217_THR} ########\n\n\n\n')
    main() 