import pandas as pd
import numpy as np
import sys
import importlib
import os
# sys.path.append('../')
from settings import *
from utilis import *
import statsmodels.stats.multitest as smm
import csv
import argparse


# Map biomarker type to settings
BIOMARKER_SETTINGS = {
    'tau_217': {
        'cat': TAU_217_CAT,
        'high_thr': HIGH_TAU_217_THR,
        'data_path': 'Tau/217/high_vs_low_data.csv',
        'save_dir': 'Tau/217/'
    },
    'gfap': {
        'cat': GFAP_CAT,
        'high_thr': HIGH_GFAP_THR,
        'data_path': 'GFAP/high_vs_low_data.csv',
        'save_dir': 'GFAP/'
    },
    'nfl': {
        'cat': NFL_CAT,
        'high_thr': None,  # Age-dependent thresholds
        'data_path': 'NFL/high_vs_low_data.csv',
        'save_dir': 'NFL/'
    }
}


def run_statistical_tests(df_high_vs_low, tests_to_run, biomarker_cat):
    p_values = []
    risk_factors = []
    threshold_p_values = []
    threshold_risk_factors = []
    for test in tests_to_run:
        print(f'Running {test["label"]} test...')
        col = test['col']
        label = test['label']
        test_type = test['test_type']
        threshold = test['threshold']
        threshold_label = test['threshold_label']
        threshold_test_type = test['threshold_test_type']
        # Get alternative hypothesis type (default: 'two-sided')
        alternative = test.get('alternative', 'two-sided')
        threshold_alternative = test.get('threshold_alternative', alternative)

        df_test = df_high_vs_low.dropna(subset=[col]).copy()
        if test_type == 'Continuous':
            try:
                df_test[col] = df_test[col].astype(float)
            except Exception:
                print(f'{col} is not a number')
                df_test = df_test[pd.to_numeric(df_test[col], errors='coerce').notna()]
                df_test[col] = df_test[col].astype(float)
        elif test_type == 'Ordinal':
            # Map string categories to numbers if needed
            if df_test[col].dtype == object:
                mapping = {'Low': 0, 'Intermediate': 1, 'High': 2, 'Very High': 3}
                df_test[col] = df_test[col].map(mapping)
        if test_type == 'Categorical':
            group1 = df_test
            group2 = [biomarker_cat, col]
        else:
            group1 = df_test[df_test[biomarker_cat] == 'Low'][col]
            group2 = df_test[df_test[biomarker_cat] == 'High'][col]
        p = stat_test(test_type, group1, group2, alternative=alternative)
        p_values.append(p)
        risk_factors.append(label)

        # Thresholded test
        if threshold is not None and threshold_label is not None and threshold_test_type is not None:
            thr_col = f"{col}_THR"
            df_test[thr_col] = (df_test[col] > threshold)
            group1_thr = df_test
            group2_thr = [biomarker_cat, thr_col]
            p_thr = stat_test(threshold_test_type, group1_thr, group2_thr, alternative=threshold_alternative)
            threshold_p_values.append(p_thr)
            threshold_risk_factors.append(threshold_label)
    return p_values, risk_factors, threshold_p_values, threshold_risk_factors


def correct_multiple_testing(p_values, risk_factors, threshold_p_values=None, threshold_risk_factors=None, alpha=0.05, method='fdr_bh'):
    csv_rows = []
    # Correct main variables
    if p_values:
        reject, p_values_corrected, _, _ = smm.multipletests(p_values, alpha=alpha, method=method)
        for i, p in enumerate(p_values):
            csv_rows.append([risk_factors[i], f'{p:.5f}', f'{p_values_corrected[i]:.5f}'])
    # Correct threshold variables separately
    if threshold_p_values and threshold_risk_factors:
        reject_thr, p_values_corrected_thr, _, _ = smm.multipletests(threshold_p_values, alpha=alpha, method=method)
        for i, p in enumerate(threshold_p_values):
            csv_rows.append([threshold_risk_factors[i], f'{p:.5f}', f'{p_values_corrected_thr[i]:.5f}'])
    return csv_rows


def save_csv(csv_rows, csv_file):
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Feature', 'p-value', 'adjusted p-value'])
        writer.writerows(csv_rows)


def get_filtered_tests(config_name, sex=None):
    """Get tests configuration filtered by sex if needed."""
    # Dynamically import the configuration
    if config_name == 'behavioral':
        config_module = importlib.import_module('behavioral.behavioral_configuration')
        TESTS = getattr(config_module, 'BEHAVIORAL_TESTS')
    elif config_name == 'cognitive':
        config_module = importlib.import_module('cognitive.cognitive_configuration')
        TESTS = getattr(config_module, 'COGNITIVE_TESTS')
    elif config_name == 'risk_factors':
        config_module = importlib.import_module('risk_factors.risk_factors_configuration')
        TESTS = getattr(config_module, 'RISK_FACTORS_TESTS')

    elif config_name == 'blood_tests':
        config_module = importlib.import_module('blood_tests.blood_tests_configuration')
        TESTS = getattr(config_module, 'BLOOD_TESTS')
    elif config_name == 'systemic_factors':
        config_module = importlib.import_module('systemic_factors.systemic_factors_configuration')
        TESTS = getattr(config_module, 'SYSTEMIC_FACTORS_TESTS')
    elif config_name == 'categorical_blood_tests':
        config_module = importlib.import_module('categorical_blood_tests.categorical_blood_tests_configuration')
        TESTS = getattr(config_module, 'CATEGORICAL_BLOOD_TESTS')
    elif config_name == 'genetic_risk_factors':
        config_module = importlib.import_module('genetic_risk_factors.genetic_risk_factors_configuration')
        TESTS = getattr(config_module, 'GENETIC_RISK_FACTORS_TESTS')
    elif config_name == 'reported_risk_factors':
        config_module = importlib.import_module('reported_risk_factors.reported_risk_factors_configuration')
        TESTS = getattr(config_module, 'RISK_FACTORS_TESTS')
        # Filter out Menopause age if not analyzing females only
        if sex != 'f':
            TESTS = [test for test in TESTS if test['col'] != 'Menopause age']
    elif config_name == 'moca_tests':
        config_module = importlib.import_module('moca_tests.moca_tests_configuration')
        TESTS = getattr(config_module, 'MOCA_TESTS')
    else:
        print(f"Unknown config_name '{config_name}'. Use 'behavioral', 'cognitive', 'risk_factors', 'blood_tests', 'systemic_factors', 'categorical_blood_tests', 'genetic_risk_factors', 'reported_risk_factors', or 'moca_tests'.")
        sys.exit(1)
    
    return TESTS


def main(biomarker_type, config_name, sex=None, mci=None):
    biomarker = BIOMARKER_SETTINGS[biomarker_type]
    TESTS = get_filtered_tests(config_name, sex)

    SAVE_DIR = biomarker['save_dir'] + f'/{config_name}/'
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Read data
    df_high_vs_low = pd.read_csv(biomarker['data_path'])
    
    # Filter by sex if provided
    sex_str = ''
    if sex is not None:
        if sex.lower() == 'm':
            # Gender is encoded as 0 for male, 1 for female
            df_high_vs_low = df_high_vs_low[df_high_vs_low['Gender'] == 0]
            sex_str = '_males'
        elif sex.lower() == 'f':
            # Gender is encoded as 0 for male, 1 for female
            df_high_vs_low = df_high_vs_low[df_high_vs_low['Gender'] == 1]
            sex_str = '_females'
        else:
            print(f"Unknown sex argument '{sex}'. Use 'm' or 'f'.")
            sys.exit(1)
        print(f"Filtered by sex: {sex_str[1:].capitalize()} (n={len(df_high_vs_low)})")
    else:
        print(f'Cohort size: {len(df_high_vs_low)}')

    # Filter by MCI if provided
    mci_str = ''
    if mci is not None:
        if mci == 0 or mci == '0':
            df_high_vs_low = df_high_vs_low[df_high_vs_low['MCI'] == 0]
            mci_str = '_cognitively_healthy'
            print(f"Filtered by MCI: Cognitively Healthy (n={len(df_high_vs_low)})")
        elif mci == 1 or mci == '1':
            df_high_vs_low = df_high_vs_low[df_high_vs_low['MCI'] == 1]
            mci_str = '_cognitively_impaired'
            print(f"Filtered by MCI: Cognitively Impaired (n={len(df_high_vs_low)})")
        else:
            print(f"Unknown MCI argument '{mci}'. Use '0' for healthy or '1' for impaired.")
            sys.exit(1)

    OUTPUT_FILE = SAVE_DIR + f'high_vs_low_stats_{config_name}{sex_str}{mci_str}.csv'

    p_values, risk_factors, threshold_p_values, threshold_risk_factors = run_statistical_tests(df_high_vs_low, TESTS, biomarker['cat'])
    all_p_values = p_values + threshold_p_values
    all_risk_factors = risk_factors + threshold_risk_factors
    all_p_values = [p for p in all_p_values if p is not None]
    csv_rows = correct_multiple_testing(p_values, risk_factors, threshold_p_values, threshold_risk_factors)
    save_csv(csv_rows, OUTPUT_FILE)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run high vs low statistical analysis.')
    parser.add_argument('--biomarker_type', type=str, required=False, choices=['tau_217', 'gfap', 'nfl'], help='Biomarker type: tau_217, gfap, or nfl')
    parser.add_argument('--config_name', type=str, required=False, choices=['behavioral', 'cognitive', 'risk_factors', 'blood_tests', 'systemic_factors', 'categorical_blood_tests', 'genetic_risk_factors', 'reported_risk_factors', 'moca_tests'], help='Configuration name')
    parser.add_argument('--sex', type=str, choices=['m', 'f'], default=None, help='Sex: m or f (optional)')
    parser.add_argument('--MCI', type=str, choices=['0', '1'], default=None, help='MCI: 0 for healthy, 1 for impaired (optional)')
    parser.add_argument('--run_all', action='store_true', help='Set this flag to run all combinations of biomarker_type, config_name, sex, and MCI')
    args = parser.parse_args()

    if args.run_all:
        biomarker_types = ['tau_217', 'gfap', 'nfl']
        config_names = ['behavioral', 'cognitive', 'risk_factors', 'blood_tests', 'systemic_factors', 'categorical_blood_tests', 'genetic_risk_factors', 'reported_risk_factors', 'moca_tests']
        sexes = [None, 'm', 'f']
        mcis = [None, '0', '1']
        for biomarker_type in biomarker_types:
            for config_name in config_names:
                for sex in sexes:
                    for mci in mcis:
                        # Skip redundant None/None (already covered by default run)
                        print(f'######## Running high vs low tests with biomarker = {biomarker_type} ########')
                        print(f'Config: biomarker_type={biomarker_type}, config_name={config_name}, sex={sex}, MCI={mci}\n')
                        main(biomarker_type, config_name, sex, mci)
    else:
        if not args.biomarker_type or not args.config_name:
            parser.error('--biomarker_type and --config_name are required unless --run_all=true')
        print(f'######## Running high vs low tests with biomarker = {args.biomarker_type} ########\n\n\n\n')
        main(args.biomarker_type, args.config_name, args.sex, args.MCI) 