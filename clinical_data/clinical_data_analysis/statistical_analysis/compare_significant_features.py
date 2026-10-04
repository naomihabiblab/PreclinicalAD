import pandas as pd
import numpy as np

def load_and_prepare_data():
    """
    Load the two CSV files and prepare them for comparison
    """
    # Load the trait association results
    trait_df = pd.read_csv('all_trait_association_results.csv')
    
    # Load the high vs low stats results
    high_low_df = pd.read_csv('all_high_vs_low_stats_all_biomarkers.csv')
    
    return trait_df, high_low_df


def normalize_trait_names(trait_df):
    """
    Normalize trait names to match the high vs low analysis format
    Remove categorical value suffixes like (OSA_YES) to match original feature names
    """
    # Create a copy to avoid modifying the original
    trait_df_normalized = trait_df.copy()
    
    # Define patterns to clean up trait names
    # Remove patterns like (OSA_YES), (APOE_risk_1.0), etc.
    trait_df_normalized['Trait'] = trait_df_normalized['Trait'].str.replace(r'\s*\([^)]+\)', '', regex=True)
    
    # Handle specific cases where trait names have slight variations
    # TMT -> TMT (Trail B)
    trait_df_normalized['Trait'] = trait_df_normalized['Trait'].replace('TMT', 'TMT (Trail B)')
    
    # Frontal memory score -> Frontal memory score (#1)
    trait_df_normalized['Trait'] = trait_df_normalized['Trait'].replace('Frontal memory score', 'Frontal memory score (#1)')
    
    return trait_df_normalized


def filter_full_cohort_data(trait_df, high_low_df):
    """
    Filter both datasets to only include full cohort data (Sex == 'All' and MCI_Status == 'All')
    """
    # Filter trait association data for full cohort
    trait_full_cohort = trait_df[(trait_df['Sex'] == 'All') & (trait_df['MCI_Status'] == 'All')]
    # Filter high vs low data for full cohort
    high_low_full_cohort = high_low_df[(high_low_df['MCI_Status'] == 'All')]
    # Normalize trait names to match high vs low analysis
    trait_full_cohort = normalize_trait_names(trait_full_cohort)
    
    return trait_full_cohort, high_low_full_cohort


def create_comparison_key(df, file_type):
    """
    Create a unique key for each row to enable comparison between the two datasets
    """
    if file_type == 'trait':
        # For trait association results, create key from: Biomarker, Config, Trait
        df['comparison_key'] = df['Biomarker'] + '_' + df['Config'] + '_' + df['Trait']
    else:
        # For high vs low stats, create key from: Biomarker_Type, Source, Feature
        df['comparison_key'] = df['Biomarker_Type'] + '_' + df['Source'] + '_' + df['Feature']
    
    return df


def compare_significant_features():
    """
    Compare significant features between the two datasets (full cohort only)
    """
    # Load data
    trait_df, high_low_df = load_and_prepare_data()
    
    # Filter for full cohort data only
    trait_full_cohort, high_low_full_cohort = filter_full_cohort_data(trait_df, high_low_df)
    
    # Create comparison keys
    trait_full_cohort = create_comparison_key(trait_full_cohort, 'trait')
    high_low_full_cohort = create_comparison_key(high_low_full_cohort, 'high_low')
    
    # Get significant features from each dataset
    trait_significant = trait_full_cohort[trait_full_cohort['Significant'] == True]
    high_low_significant = high_low_full_cohort[high_low_full_cohort['Significant'] == True]
    
    # Get the comparison keys for significant features
    trait_significant_keys = set(trait_significant['comparison_key'])
    high_low_significant_keys = set(high_low_significant['comparison_key'])
    
    # Find overlaps and differences
    significant_in_both = trait_significant_keys.intersection(high_low_significant_keys)
    only_in_trait = trait_significant_keys - high_low_significant_keys
    only_in_high_low = high_low_significant_keys - trait_significant_keys
    
    return {
        'significant_in_both': significant_in_both,
        'only_in_trait': only_in_trait,
        'only_in_high_low': only_in_high_low,
        'trait_significant_df': trait_significant,
        'high_low_significant_df': high_low_significant,
        'trait_df': trait_full_cohort,
        'high_low_df': high_low_full_cohort
    }


def get_detailed_info(keys, trait_df, high_low_df, category_name):
    """
    Get detailed information about features in a specific category
    """
    results = []
    
    for key in keys:
        # Find the corresponding rows in both datasets
        trait_row = trait_df[trait_df['comparison_key'] == key]
        high_low_row = high_low_df[high_low_df['comparison_key'] == key]
        
        # Initialize with None values
        trait_info = None
        high_low_info = None
        
        # Get trait info if available
        if not trait_row.empty:
            trait_info = trait_row.iloc[0]
        
        # Get high_low info if available
        if not high_low_row.empty:
            high_low_info = high_low_row.iloc[0]
        
        # Create result dictionary
        result = {
            'comparison_key': key,
            'biomarker': None,
            'config/source': None,
            'trait/feature': None,
            'trait_coefficient': None,
            'trait_p_value': None,
            'trait_adjusted_p_value': None,
            'high_low_p_value': None,
            'high_low_adjusted_p_value': None
        }
        
        # Fill in trait information if available
        if trait_info is not None:
            result.update({
                'biomarker': trait_info.get('Biomarker', None),
                'config/source': trait_info.get('Config', None),
                'trait/feature': trait_info.get('Trait', None),
                'trait_coefficient': trait_info.get('Coefficient', None),
                'trait_p_value': trait_info.get('p-value', None),
                'trait_adjusted_p_value': trait_info.get('adjusted p-value', None)
            })
        
        # Fill in high_low information if available
        if high_low_info is not None:
            result.update({
                'biomarker': result['biomarker'] or high_low_info.get('Biomarker_Type', None),
                'config/source': result['config/source'] or high_low_info.get('Source', None),
                'trait/feature': result['trait/feature'] or high_low_info.get('Feature', None),
                'high_low_p_value': high_low_info.get('p-value', None),
                'high_low_adjusted_p_value': high_low_info.get('adjusted p-value', None)
            })
        
        results.append(result)
    
    # Create DataFrame and sort by biomarker
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values('biomarker')
    
    return df


def print_summary(comparison_results):
    """
    Print a comprehensive summary of the comparison
    """
    print("=" * 80)
    print("COMPARISON OF SIGNIFICANT FEATURES BETWEEN TWO DATASETS")
    print("(FULL COHORT DATA ONLY - Sex='All' and MCI_Status='All')")
    print("=" * 80)
    
    # Summary counts
    print(f"\nSUMMARY COUNTS:")
    print(f"Features significant in both datasets: {len(comparison_results['significant_in_both'])}")
    print(f"Features only significant in trait association: {len(comparison_results['only_in_trait'])}")
    print(f"Features only significant in high vs low comparison: {len(comparison_results['only_in_high_low'])}")
    
    # Detailed information for each category
    if comparison_results['significant_in_both']:
        print(f"\n{'='*60}")
        print("FEATURES SIGNIFICANT IN BOTH DATASETS")
        print(f"{'='*60}")
        both_df = get_detailed_info(
            comparison_results['significant_in_both'],
            comparison_results['trait_df'],
            comparison_results['high_low_df'],
            'both'
        )
        print(both_df.to_string(index=False))
        
        # Save to CSV
        both_df.to_csv('significant_in_both_datasets_full_cohort.csv', index=False)
        print(f"\nDetailed results saved to: significant_in_both_datasets_full_cohort.csv")
    
    if comparison_results['only_in_trait']:
        print(f"\n{'='*60}")
        print("FEATURES ONLY SIGNIFICANT IN TRAIT ASSOCIATION")
        print(f"{'='*60}")
        trait_only_df = get_detailed_info(
            comparison_results['only_in_trait'],
            comparison_results['trait_df'],
            comparison_results['high_low_df'],
            'trait_only'
        )
        print(trait_only_df.to_string(index=False))
        
        # Save to CSV
        trait_only_df.to_csv('significant_only_in_trait_association_full_cohort.csv', index=False)
        print(f"\nDetailed results saved to: significant_only_in_trait_association_full_cohort.csv")
    
    if comparison_results['only_in_high_low']:
        print(f"\n{'='*60}")
        print("FEATURES ONLY SIGNIFICANT IN HIGH VS LOW COMPARISON")
        print(f"{'='*60}")
        high_low_only_df = get_detailed_info(
            comparison_results['only_in_high_low'],
            comparison_results['trait_df'],
            comparison_results['high_low_df'],
            'high_low_only'
        )
        print(high_low_only_df.to_string(index=False))
        
        # Save to CSV
        high_low_only_df.to_csv('significant_only_in_high_vs_low_full_cohort.csv', index=False)
        print(f"\nDetailed results saved to: significant_only_in_high_vs_low_full_cohort.csv")


def main():
    """
    Main function to run the comparison
    """
    print("Starting comparison of significant features (full cohort data only)...")
    
    try:
        # Perform the comparison
        comparison_results = compare_significant_features()
        
        # Print summary
        print_summary(comparison_results)
        
        print("\n" + "="*80)
        print("COMPARISON COMPLETED SUCCESSFULLY")
        print("="*80)
        
    except Exception as e:
        print(f"Error during comparison: {str(e)}")
        raise


if __name__ == "__main__":
    main() 