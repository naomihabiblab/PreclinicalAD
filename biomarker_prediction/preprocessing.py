"""
Data preprocessing functions for tau-217 classification.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from config import (
    OUTLIERS_IDS, HIGH_TAU_217_THRESHOLD, TAU_217,
    MOCA_COL, MCI_THR, INFORMATIVE_FEATURES
)


def remove_outliers(data):
    """Remove outliers from the dataset.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input dataframe with ID as index
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with outliers removed
    """
    return data[~data.index.isin(OUTLIERS_IDS)]


def filter_data_for_classification(data):
    """Filter and prepare data for classification.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input dataframe with all features
        
    Returns:
    --------
    tuple
        (features_df, result_vector) where:
        - features_df: DataFrame with selected features and MCI column
        - result_vector: Series with tau-217 values for matching indices
    """
    features = INFORMATIVE_FEATURES

    # Use only features that actually exist in the dataframe to avoid KeyError
    available_features = [f for f in features if f in data.columns]
    missing_features = [f for f in features if f not in data.columns]

    if missing_features:
        print(f"Warning: the following features are missing from input data and will be ignored: {missing_features}")

    df = data[available_features].dropna(axis=0)
    df['MCI'] = (data[MOCA_COL] <= MCI_THR).astype(int)
    result_vector = data[data.index.isin(df.index)][TAU_217]
    return df, result_vector


def categorize_data(tau_vector, biomarker="p-tau217"):
    """Categorize tau values into binary categories.
    
    Parameters:
    -----------
    tau_vector : pd.Series
        Continuous tau values
    biomarker : str, default="p-tau217"
        Type of tau measurement
        
    Returns:
    --------
    pd.DataFrame
        Binary classification labels (0 or 1)
    """
    tau_categories = np.zeros(tau_vector.shape, dtype=int)
    if biomarker == TAU_217:
        tau_categories[tau_vector > HIGH_TAU_217_THRESHOLD] = 1  # tau level is higher than the threshold
    tau_categories = pd.DataFrame(tau_categories, index=tau_vector.index, columns=[biomarker])
    return tau_categories


def filter_by_cognitive_status(df, mci_only=False, nci_only=False):
    """Filter dataframe by cognitive status.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    mci_only : bool, default=False
        If True, keep only MCI patients (MOCA_score < MCI_THR)
    nci_only : bool, default=False
        If True, keep only NCI patients (MOCA_score >= MCI_THR)
        
    Returns:
    --------
    pd.DataFrame
        Filtered dataframe
    """
    if mci_only:
        df = df[df[MOCA_COL] < MCI_THR]
    if nci_only:
        df = df[df[MOCA_COL] >= MCI_THR]
    return df


def scale_features(X_train, X_val=None, X_test=None, return_scaler=False):
    """Scale features using StandardScaler.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    X_val : pd.DataFrame, optional
        Validation features
    X_test : pd.DataFrame, optional
        Test features
    return_scaler : bool, default=False
        If True, return the fitted scaler
        
    Returns:
    --------
    tuple
        Scaled dataframes (and optionally the scaler)
    """
    # scaler = StandardScaler()
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train_norm = scaler.fit_transform(X_train)
    X_train_norm = pd.DataFrame(
        X_train_norm, 
        columns=X_train.columns, 
        index=X_train.index
    )
    
    result = [X_train_norm]
    
    if X_val is not None:
        X_val_norm = scaler.transform(X_val)
        X_val_norm = pd.DataFrame(
            X_val_norm,
            columns=X_val.columns,
            index=X_val.index
        )
        result.append(X_val_norm)
    
    if X_test is not None:
        X_test_norm = scaler.transform(X_test)
        X_test_norm = pd.DataFrame(
            X_test_norm,
            columns=X_test.columns,
            index=X_test.index
        )
        result.append(X_test_norm)
    
    if return_scaler:
        result.append(scaler)
    
    return tuple(result) if len(result) > 1 else result[0]


def calculate_scaled_moca_threshold(X_train):
    """Calculate scaled MOCA threshold for BaselineClassifier.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features (before scaling)
        
    Returns:
    --------
    float
        Scaled MOCA threshold
    """
    moca_mean = X_train[MOCA_COL].mean()
    moca_std = X_train[MOCA_COL].std()
    moca_threshold_scaled = (MCI_THR - moca_mean) / moca_std
    return moca_threshold_scaled

