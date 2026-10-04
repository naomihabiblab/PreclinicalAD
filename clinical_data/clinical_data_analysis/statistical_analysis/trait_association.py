import pandas as pd
import numpy as np
import sys
import importlib
import os
from settings import *
from utilis import *
import statsmodels.api as sm
import statsmodels.stats.multitest as smm
import csv
import argparse

# Map biomarker to settings
BIOMARKER_SETTINGS = {
    'tau_217': {
        'col': 'p-tau217',
        'data_path': 'Tau/217/trait_association_data.csv',
        'save_dir': 'Tau/217/trait_association/'
    },
    'nfl': {
        'col': 'NFL',
        'data_path': 'NFL/trait_association_data.csv',
        'save_dir': 'NFL/trait_association/'

    },
    'gfap': {
        'col': 'GFAP',
        'data_path': 'GFAP/trait_association_data.csv',
        'save_dir': 'GFAP/trait_association/'
    }
}


def run_trait_association(df, biomarker_col, traits_config, covariates=['Age', 'Gender']):
    results = []
    for trait in traits_config:
        # trait can be a dict (from config) or a string
        if isinstance(trait, dict):
            trait_col = trait.get('col', trait.get('label', trait))
            test_type = trait.get('test_type', 'Continuous')
            label = trait.get('label', trait_col)
        else:
            raise Exception("problem in configuration file")
        cols = [trait_col] + covariates
        df_model = df.dropna(subset=[biomarker_col] + cols).copy()
        X = df_model[[trait_col] + covariates].copy()
        # Handle encoding based on test_type
        if test_type == 'Categorical':
            X = pd.get_dummies(X, columns=[trait_col], drop_first=True)
        elif test_type == 'Ordinal':
            # Try to map common ordinal categories if dtype is object
            if X[trait_col].dtype == object:
                mapping = {'Low': 0, 'Intermediate': 1, 'High': 2, 'Very High': 3}
                X[trait_col] = X[trait_col].map(mapping)
            # If still not numeric, try to convert to float
            if not np.issubdtype(X[trait_col].dtype, np.number):
                X[trait_col] = pd.to_numeric(X[trait_col], errors='coerce')
                X = X.dropna(subset=[trait_col])
        elif test_type == 'Continuous':
            # Convert to float, drop rows where conversion fails
            X[trait_col] = pd.to_numeric(X[trait_col], errors='coerce')
            X = X.dropna(subset=[trait_col])

        X = X.astype(float)
        X = sm.add_constant(X)
        # Align y with X after dropping rows beacuese of nan values in the trait
        y = df_model.loc[X.index, biomarker_col]
        try:
            model = sm.OLS(y, X).fit()
            # For categorical, take the first dummy column for the trait
            if test_type == 'Categorical':
                trait_dummies = [c for c in X.columns if c.startswith(trait_col + '_')]
                for dummy in trait_dummies:
                    coef = model.params.get(dummy, np.nan)
                    pval = model.pvalues.get(dummy, np.nan)
                    results.append({'trait': f'{label} ({dummy})', 'coef': coef, 'pval': pval})
                continue
            coef = model.params.get(trait_col, np.nan)
            pval = model.pvalues.get(trait_col, np.nan)
        except Exception as e:
            print(f'Exception during regression for {trait_col}: {e}')
            coef = np.nan
            pval = np.nan
        results.append({'trait': label, 'coef': coef, 'pval': pval})
    return results

def correct_multiple_testing(p_values, alpha=0.05, method='fdr_bh'):
    reject, pvals_corrected, _, _ = smm.multipletests(p_values, alpha=alpha, method=method)
    return pvals_corrected

def save_csv(results, output_file):
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Trait', 'Coefficient', 'p-value', 'adjusted p-value'])
        for row in results:
            writer.writerow([row['trait'], row['coef'], row['pval'], row['adj_pval']])

def main(biomarker, config_name, input_file=None, sex=None, mci=None):
    biomarker_info = BIOMARKER_SETTINGS[biomarker]
    # Dynamically import the configuration
    if config_name == 'behavioral':
        config_module = importlib.import_module('behavioral.behavioral_configuration')
        TRAITS = getattr(config_module, 'BEHAVIORAL_TESTS')
    elif config_name == 'cognitive':
        config_module = importlib.import_module('cognitive.cognitive_configuration')
        TRAITS = getattr(config_module, 'COGNITIVE_TESTS')
    elif config_name == 'risk_factors':
        config_module = importlib.import_module('risk_factors.risk_factors_configuration')
        TRAITS = getattr(config_module, 'RISK_FACTORS_TESTS')
    elif config_name == 'blood_tests':
        config_module = importlib.import_module('blood_tests.blood_tests_configuration')
        TRAITS = getattr(config_module, 'BLOOD_TESTS')
    elif config_name == 'systemic_factors':
        config_module = importlib.import_module('systemic_factors.systemic_factors_configuration')
        TRAITS = getattr(config_module, 'SYSTEMIC_FACTORS_TESTS')
    elif config_name == 'categorical_blood_tests':
        config_module = importlib.import_module('categorical_blood_tests.categorical_blood_tests_configuration')
        TRAITS = getattr(config_module, 'CATEGORICAL_BLOOD_TESTS')
    elif config_name == 'genetic_risk_factors':
        config_module = importlib.import_module('genetic_risk_factors.genetic_risk_factors_configuration')
        TRAITS = getattr(config_module, 'GENETIC_RISK_FACTORS_TESTS')
    elif config_name == 'reported_risk_factors':
        config_module = importlib.import_module('reported_risk_factors.reported_risk_factors_configuration')
        TRAITS = getattr(config_module, 'RISK_FACTORS_TESTS')
    elif config_name == 'moca_tests':
        config_module = importlib.import_module('moca_tests.moca_tests_configuration')
        TRAITS = getattr(config_module, 'MOCA_TESTS')
    else:
        print(f"Unknown config_name '{config_name}'. Use 'behavioral', 'cognitive', 'risk_factors', 'blood_tests', 'systemic_factors', 'categorical_blood_tests', 'genetic_risk_factors', 'reported_risk_factors', or 'moca_tests'.")
        sys.exit(1)

    SAVE_DIR = biomarker_info['save_dir']
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Determine input file path
    if input_file is not None and input_file != '':
        data_path = input_file
    else:
        data_path = biomarker_info.get('data_path', None)
        if not data_path or not os.path.exists(data_path):
            print(f"No input file provided and default data_path for biomarker '{biomarker}' does not exist: {data_path}")
            sys.exit(1)

    # Read data
    df = pd.read_csv(data_path)

    # Filter by sex if provided
    sex_str = ''
    if sex is not None:
        if sex == 0: # Males
            df = df[df['Gender'] == 0]
            sex_str = '_males'
        elif sex == 1: # Females
            df = df[df['Gender'] == 1]
            sex_str = '_females'
        else:
            print(f"Unknown sex argument '{sex}'. Use 'm' or 'f'.")
            sys.exit(1)
        print(f"Filtered by sex: {sex_str[1:].capitalize()} (n={len(df)})")
    else:
        print(f'Cohort size: {len(df)}')

    # Filter by MCI if provided
    mci_str = ''
    if mci is not None:
        if mci == 0 or mci == '0':
            df = df[df['MCI'] == 0]
            mci_str = '_cognitively_healthy'
            print(f"Filtered by MCI: Cognitively Healthy (n={len(df)})")
        elif mci == 1 or mci == '1':
            df = df[df['MCI'] == 1]
            mci_str = '_cognitively_impaired'
            print(f"Filtered by MCI: Cognitively Impaired (n={len(df)})")
        else:
            print(f"Unknown MCI argument '{mci}'. Use '0' for healthy or '1' for impaired.")
            sys.exit(1)

    OUTPUT_FILE = SAVE_DIR + f'trait_association_{config_name}{sex_str}{mci_str}.csv'

    # Run trait association
    results = run_trait_association(df, biomarker_info['col'], TRAITS)
    pvals = [r['pval'] for r in results]
    adj_pvals = correct_multiple_testing(pvals)
    for i, adj_pval in enumerate(adj_pvals):
        results[i]['adj_pval'] = adj_pval
    save_csv(results, OUTPUT_FILE)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run trait association analysis.')
    parser.add_argument('--biomarker', type=str, required=True, choices=['tau_217', 'nfl', 'gfap'], help='Biomarker to analyze')
    parser.add_argument('--config_name', type=str, required=False, choices=['behavioral', 'cognitive', 'risk_factors', 'blood_tests', 'systemic_factors', 'categorical_blood_tests', 'genetic_risk_factors', 'reported_risk_factors', 'moca_tests'], help='Configuration name')
    parser.add_argument('--input_file', type=str, required=False, default=None, help='Input CSV file. If not provided, will use the default data_path from biomarker settings.')
    parser.add_argument('--sex', type=str, choices=[0, 1], default=None, help='Sex: 0 (m) or 1 (f) (optional)')
    parser.add_argument('--MCI', type=str, choices=['0', '1'], default=None, help='MCI: 0 for healthy, 1 for impaired (optional)')
    parser.add_argument('--run_all', type=str, default='false', help='Set to true to run all combinations of config_name, sex, and MCI for the given biomarker')
    args = parser.parse_args()

    if args.run_all.lower() == 'true':
        config_names = ['behavioral', 'cognitive', 'risk_factors', 'blood_tests', 'systemic_factors', 'categorical_blood_tests', 'genetic_risk_factors', 'reported_risk_factors', 'moca_tests']
        sexes = [None, 0, 1]
        mcis = [None, '0', '1']
        for config_name in config_names:
            for sex in sexes:
                for mci in mcis:
                    print(f'######## Running trait association: biomarker={args.biomarker}, config_name={config_name}, sex={sex}, MCI={mci} ########')
                    main(args.biomarker, config_name, args.input_file, sex, mci)
    else:
        if not args.config_name:
            parser.error('--config_name is required unless --run_all=true')
        main(args.biomarker, args.config_name, args.input_file, args.sex, args.MCI)
