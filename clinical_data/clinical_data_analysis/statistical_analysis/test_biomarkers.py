#!/usr/bin/env python3
"""Test script to verify biomarker preprocessing for tau_217, gfap, and nfl."""

import sys
import os
sys.path.append('.')

from preprocess_data import load_and_preprocess_data, BIOMARKER_SETTINGS

def test_biomarker_preprocessing():
    """Test preprocessing for all biomarkers"""
    
    print("Testing biomarker preprocessing...")
    print("=" * 50)
    
    for biomarker_type in ['tau_217', 'gfap', 'nfl']:
        print(f"\nTesting {biomarker_type.upper()}:")
        print("-" * 30)
        
        try:
            df, biomarker_cat, biomarker_folder = load_and_preprocess_data(biomarker_type)
            
            print(f"Data shape: {df.shape}")
            print(f"Biomarker category column: {biomarker_cat}")
            print(f"Save folder: {biomarker_folder}")
            
            # Show category distribution
            if biomarker_cat in df.columns:
                print("Category distribution:")
                print(df[biomarker_cat].value_counts())
            else:
                print(f"Warning: {biomarker_cat} column not found in dataframe")
                print(f"Available columns: {list(df.columns)}")
            
            # Show some basic stats
            biomarker_col = BIOMARKER_SETTINGS[biomarker_type]['col']
            if biomarker_col in df.columns:
                print(f"\n{biomarker_col} statistics:")
                print(f"cohort size: {df.shape[0]}")
                print(f"Mean: {df[biomarker_col].mean():.3f}")
                print(f"Std: {df[biomarker_col].std():.3f}")
                print(f"Min: {df[biomarker_col].min():.3f}")
                print(f"Max: {df[biomarker_col].max():.3f}")
                print(f"Non-null count: {df[biomarker_col].count()}")
            
            print(f"✓ {biomarker_type} preprocessing completed successfully")
            
        except Exception as e:
            print(f"✗ Error processing {biomarker_type}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("Testing completed!")

if __name__ == '__main__':
    test_biomarker_preprocessing()
