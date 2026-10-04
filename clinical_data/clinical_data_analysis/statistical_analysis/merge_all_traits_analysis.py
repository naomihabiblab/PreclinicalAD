import pandas as pd
import os

BIOMARKERS = ['tau_217', 'nfl', 'gfap']
CONFIGS = [
    'behavioral',
    'cognitive',
    'risk_factors',
    'blood_tests',
    'categorical_blood_tests',
    'genetic_risk_factors',
    'reported_risk_factors',
    'moca_tests'
]
SEXES = [('', 'All'), ('_males', 'Males'), ('_females', 'Females')]
MCI_STATUSES = [
    ('', 'All'),
    ('_cognitively_healthy', 'Cognitively Healthy'),
    ('_cognitively_impaired', 'Cognitively Impaired')
]

def main():
    all_dfs = []
    for biomarker in BIOMARKERS:
        if biomarker.startswith('tau_'):
            stats_dir = os.path.join('Tau', biomarker.split('_')[1], 'trait_association')
        else:
            stats_dir = os.path.join(biomarker.upper(), 'trait_association')
        for config in CONFIGS:
            for sex_suffix, sex_label in SEXES:
                for mci_suffix, mci_label in MCI_STATUSES:
                    fname = f'trait_association_{config}{sex_suffix}{mci_suffix}.csv'
                    fpath = os.path.join(stats_dir, fname)
                    if os.path.exists(fpath):
                        df = pd.read_csv(fpath)
                        df['Biomarker'] = biomarker
                        df['Config'] = config
                        df['Sex'] = sex_label
                        df['MCI_Status'] = mci_label
                        all_dfs.append(df)
                    else:
                        print(f"Warning: {fpath} not found.")
    if all_dfs:
        merged = pd.concat(all_dfs, ignore_index=True)
        # Add significance column
        merged['Significant'] = (merged['adjusted p-value'] < 0.05)
        # Reorder columns if present
        col_order = ['Biomarker', 'Config', 'Sex', 'MCI_Status', 'Trait', 'Coefficient', 'p-value', 'adjusted p-value', 'Significant']
        merged = merged[[c for c in col_order if c in merged.columns]]
        merged_csv = 'all_trait_association_results.csv'
        merged.to_csv(merged_csv, index=False)
        print(f'Merged table saved as {merged_csv}')
    else:
        print('No trait association files found to merge.')

if __name__ == '__main__':
    main()
