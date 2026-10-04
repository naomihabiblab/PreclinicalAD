import pandas as pd
import os
import sys

# Map biomarker type to settings
BIOMARKER_SETTINGS = {
    'tau_217': {
        'stats_dir': os.path.join('Tau', '217'),
    },
    'gfap': {
        'stats_dir': os.path.join('GFAP'),
    },
    'nfl': {
        'stats_dir': os.path.join('NFL'),
    }
}

def main():
    all_dfs = []
    for biomarker_type, biomarker in BIOMARKER_SETTINGS.items():
        stats_dir = biomarker['stats_dir']
        sources = [
            ('behavioral', os.path.join(stats_dir, 'behavioral', 'high_vs_low_stats_behavioral.csv'),
             os.path.join(stats_dir, 'behavioral', 'high_vs_low_stats_behavioral_males.csv'),
             os.path.join(stats_dir, 'behavioral', 'high_vs_low_stats_behavioral_females.csv')),
            ('cognitive', os.path.join(stats_dir, 'cognitive', 'high_vs_low_stats_cognitive.csv'),
             os.path.join(stats_dir, 'cognitive', 'high_vs_low_stats_cognitive_males.csv'),
             os.path.join(stats_dir, 'cognitive', 'high_vs_low_stats_cognitive_females.csv')),
            ('moca_tests', os.path.join(stats_dir, 'moca_tests', 'high_vs_low_stats_moca_tests.csv'),
             os.path.join(stats_dir, 'moca_tests', 'high_vs_low_stats_moca_tests_males.csv'),
             os.path.join(stats_dir, 'moca_tests', 'high_vs_low_stats_moca_tests_females.csv')),
            ('risk_factors', os.path.join(stats_dir, 'risk_factors', 'high_vs_low_stats_risk_factors.csv'),
             os.path.join(stats_dir, 'risk_factors', 'high_vs_low_stats_risk_factors_males.csv'),
             os.path.join(stats_dir, 'risk_factors', 'high_vs_low_stats_risk_factors_females.csv')),
            ('reported_risk_factors', os.path.join(stats_dir, 'reported_risk_factors', 'high_vs_low_stats_reported_risk_factors.csv'),
             os.path.join(stats_dir, 'reported_risk_factors', 'high_vs_low_stats_reported_risk_factors_males.csv'),
             os.path.join(stats_dir, 'reported_risk_factors', 'high_vs_low_stats_reported_risk_factors_females.csv')),
            ('blood_tests', os.path.join(stats_dir, 'blood_tests', 'high_vs_low_stats_blood_tests.csv'),
             os.path.join(stats_dir, 'blood_tests', 'high_vs_low_stats_blood_tests_males.csv'),
             os.path.join(stats_dir, 'blood_tests', 'high_vs_low_stats_blood_tests_females.csv')),
            ('systemic_factors', os.path.join(stats_dir, 'systemic_factors', 'high_vs_low_stats_systemic_factors.csv'),
             os.path.join(stats_dir, 'systemic_factors', 'high_vs_low_stats_systemic_factors_males.csv'),
             os.path.join(stats_dir, 'systemic_factors', 'high_vs_low_stats_systemic_factors_females.csv')),
            ('categorical_blood_tests', os.path.join(stats_dir, 'categorical_blood_tests', 'high_vs_low_stats_categorical_blood_tests.csv'),
             os.path.join(stats_dir, 'categorical_blood_tests', 'high_vs_low_stats_categorical_blood_tests_males.csv'),
             os.path.join(stats_dir, 'categorical_blood_tests', 'high_vs_low_stats_categorical_blood_tests_females.csv')),
            ('genetic_risk_factors', os.path.join(stats_dir, 'genetic_risk_factors', 'high_vs_low_stats_genetic_risk_factors.csv'),
             os.path.join(stats_dir, 'genetic_risk_factors', 'high_vs_low_stats_genetic_risk_factors_males.csv'),
             os.path.join(stats_dir, 'genetic_risk_factors', 'high_vs_low_stats_genetic_risk_factors_females.csv')),
        ]
        # Add MCI-specific sources
        mci_statuses = [
            ('cognitively_healthy', '_cognitively_healthy', 'Cognitively Healthy'),
            ('cognitively_impaired', '_cognitively_impaired', 'Cognitively Impaired'),
        ]
        for base, base_path, males_path, females_path in sources:
            # All subjects
            if os.path.exists(base_path):
                df = pd.read_csv(base_path)
                # Merge with males and females if available
                if males_path and os.path.exists(males_path):
                    df_m = pd.read_csv(males_path)
                    df = df.merge(df_m[['Feature', 'p-value', 'adjusted p-value']], on='Feature', how='left', suffixes=('', '_males'))
                    df.rename(columns={'p-value_males': 'p-value_males', 'adjusted p-value_males': 'adjusted p-value_males'}, inplace=True)
                else:
                    df['p-value_males'] = float('nan')
                    df['adjusted p-value_males'] = float('nan')
                if females_path and os.path.exists(females_path):
                    df_f = pd.read_csv(females_path)
                    df = df.merge(df_f[['Feature', 'p-value', 'adjusted p-value']], on='Feature', how='left', suffixes=('', '_females'))
                    df.rename(columns={'p-value_females': 'p-value_females', 'adjusted p-value_females': 'adjusted p-value_females'}, inplace=True)
                else:
                    df['p-value_females'] = float('nan')
                    df['adjusted p-value_females'] = float('nan')
                df['Source'] = base
                df['Biomarker_Type'] = biomarker_type
                df['MCI_Status'] = 'All'
                all_dfs.append(df)
            else:
                print(f"Warning: {base_path} not found.")
            # MCI-specific
            for mci_key, mci_suffix, mci_label in mci_statuses:
                mci_path = base_path.replace('.csv', f'{mci_suffix}.csv')
                if os.path.exists(mci_path):
                    df_mci = pd.read_csv(mci_path)
                    # Try to merge with sex-specific MCI files if they exist
                    males_mci_path = males_path.replace('.csv', f'{mci_suffix}.csv') if males_path else None
                    females_mci_path = females_path.replace('.csv', f'{mci_suffix}.csv') if females_path else None
                    if males_mci_path and os.path.exists(males_mci_path):
                        df_m = pd.read_csv(males_mci_path)
                        df_mci = df_mci.merge(df_m[['Feature', 'p-value', 'adjusted p-value']], on='Feature', how='left', suffixes=('', '_males'))
                        df_mci.rename(columns={'p-value_males': 'p-value_males', 'adjusted p-value_males': 'adjusted p-value_males'}, inplace=True)
                    else:
                        df_mci['p-value_males'] = float('nan')
                        df_mci['adjusted p-value_males'] = float('nan')
                    if females_mci_path and os.path.exists(females_mci_path):
                        df_f = pd.read_csv(females_mci_path)
                        df_mci = df_mci.merge(df_f[['Feature', 'p-value', 'adjusted p-value']], on='Feature', how='left', suffixes=('', '_females'))
                        df_mci.rename(columns={'p-value_females': 'p-value_females', 'adjusted p-value_females': 'adjusted p-value_females'}, inplace=True)
                    else:
                        df_mci['p-value_females'] = float('nan')
                        df_mci['adjusted p-value_females'] = float('nan')
                    df_mci['Source'] = base
                    df_mci['Biomarker_Type'] = biomarker_type
                    df_mci['MCI_Status'] = mci_label
                    all_dfs.append(df_mci)
                else:
                    print(f"Warning: {mci_path} not found.")

    if all_dfs:
        merged = pd.concat(all_dfs, ignore_index=True)
        # Add significance column
        merged['Significant'] = (merged['adjusted p-value'] < 0.05)
        # Reorder columns
        col_order = ['Biomarker_Type', 'Source', 'MCI_Status', 'Feature', 'p-value', 'adjusted p-value', 'p-value_males', 'adjusted p-value_males', 'p-value_females', 'adjusted p-value_females', 'Significant']
        merged = merged[col_order]
        merged_csv = 'all_high_vs_low_stats_all_biomarkers.csv'
        merged.to_csv(merged_csv, index=False)
        print(f'Merged table saved as {merged_csv}')
    else:
        print('No stats files found to merge.')

if __name__ == '__main__':
    main() 
