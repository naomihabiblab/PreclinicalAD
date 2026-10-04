"""
TabPFN Model Explainability using SHAP, SHAP-IQ, and ALE

This script loads trained TabPFN classification models and generates multiple types of
explanations to understand feature importance and contributions to predictions:
- SHAP: Shapley Additive Explanations for individual predictions
- SHAP-IQ: Advanced SHAP with interaction support
- ALE: Accumulated Local Effects plots for global feature effects

"""

import pandas as pd
import numpy as np
import os
import argparse
import pickle
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns

from tabpfn_extensions.interpretability import shap as tabpfn_shap
from tabpfn_extensions.interpretability import shapiq as tabpfn_shapiq
from alibi.explainers import ALE
import shap

# Set style for plots (no grid)
plt.style.use('seaborn-v0_8')
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['axes.grid'] = False
sns.set_palette("husl")

DEFAULT_BAD_COLOR = '#fbb4ae'   # red  — high p-tau217 (class 1)
DEFAULT_GOOD_COLOR = '#b4cde3'  # blue — low p-tau217  (class 0)
TAU_CLASS_COLORS = {0: DEFAULT_GOOD_COLOR, 1: DEFAULT_BAD_COLOR}


def _align_ale_bins_and_values(feat_ale, feat_bins, feat_name=None):
    """Align ALE bin boundaries and values to same-length 1D arrays (bin centers + ale values).
    Returns (bins_1d, ale_1d). Optionally warn on length mismatch if feat_name is provided.
    For 2D feat_ale (n_bins x n_classes), only the first class column is used so bins and ale match.
    """
    feat_ale = np.asarray(feat_ale)
    feat_bins_flat = np.asarray(feat_bins).flatten() if np.asarray(feat_bins).ndim > 1 else np.asarray(feat_bins).ravel()
    # Use first class/slice so we have 1D ALE to align with bins (avoid mixing classes when trimming)
    if feat_ale.ndim > 1:
        feat_ale_flat = feat_ale[:, 0].ravel()
    else:
        feat_ale_flat = feat_ale.ravel()
    if len(feat_bins_flat) == len(feat_ale_flat) + 1:
        feat_bins_to_use = (feat_bins_flat[:-1] + feat_bins_flat[1:]) / 2
    elif len(feat_bins_flat) == len(feat_ale_flat):
        feat_bins_to_use = feat_bins_flat
    else:
        min_len = min(len(feat_bins_flat), len(feat_ale_flat))
        if feat_name is not None:
            print(f"  ⚠ Warning: Length mismatch for {feat_name}: bins={len(feat_bins_flat)}, ale={len(feat_ale_flat)}. Using first {min_len} values.")
        feat_bins_to_use = feat_bins_flat[:min_len]
        feat_ale_flat = feat_ale_flat[:min_len]
    return feat_bins_to_use, feat_ale_flat


def _shap_values_to_matrix(shap_values):
    """Convert SHAP result (Explanation or array) to 2D numpy array (n_samples, n_features)."""
    try:
        mat = shap_values.values if hasattr(shap_values, 'values') else np.asarray(shap_values)
        if mat.ndim == 3:
            mat = mat[:, :, 0]
        return np.asarray(mat)
    except Exception:
        mat = np.array(shap_values)
        return mat[:, :, 0] if mat.ndim == 3 else mat


def _load_precalculated_shap(path, X_test, feature_names, mci_info, n_samples):
    """Load pre-calculated SHAP from CSV; align features/indices; apply MCI filter if needed.
    Returns (shap_values_mat, X_test_subset, shap_values_for_plot).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pre-calculated SHAP values file not found: {path}")
    shap_values_df = pd.read_csv(path, index_col=0)
    shap_values_mat = shap_values_df.values
    if list(shap_values_df.columns) != feature_names:
        if set(shap_values_df.columns) == set(feature_names):
            shap_values_df = shap_values_df[feature_names]
            shap_values_mat = shap_values_df.values
        else:
            raise ValueError("Feature names in CSV do not match expected feature names")
    if isinstance(X_test, pd.DataFrame):
        common_indices = shap_values_df.index.intersection(X_test.index)
        if len(common_indices) > 0:
            X_test_subset = X_test.loc[common_indices]
            shap_values_mat = shap_values_df.loc[common_indices].values
            if mci_info is not None and isinstance(mci_info, pd.Series):
                available_mci = mci_info.loc[mci_info.index.intersection(common_indices)]
                if len(available_mci) > 0:
                    final_indices = common_indices.intersection(available_mci.index)
                    if len(final_indices) > 0:
                        X_test_subset = X_test_subset.loc[final_indices]
                        shap_values_mat = shap_values_df.loc[final_indices].values
        else:
            n_use = min(len(shap_values_df), len(X_test), n_samples or len(shap_values_df))
            X_test_subset = X_test.iloc[:n_use]
            shap_values_mat = shap_values_mat[:n_use]
    else:
        n_use = min(len(shap_values_df), len(X_test), n_samples or len(shap_values_df))
        X_test_subset = X_test[:n_use]
        shap_values_mat = shap_values_mat[:n_use]
    return shap_values_mat, X_test_subset, shap_values_mat


def apply_mci_filter(X_test, y_test, mci_info, filter_mci, filter_nci):
    """Apply MCI or NCI filter to test data. Returns (X_test, y_test, mci_info)."""
    if not filter_mci and not filter_nci:
        return X_test, y_test, mci_info
    if mci_info is None:
        raise ValueError("Cannot filter by MCI: MCI information not available and train data path not provided.")
    original_size = len(X_test)
    if filter_mci:
        mci_mask = (mci_info > 0)
        if mci_mask.sum() == 0:
            raise ValueError("No samples with MCI>0 found in the data")
        X_test = X_test.loc[mci_mask]
        y_test = y_test.loc[mci_mask] if hasattr(y_test, 'loc') else y_test[mci_mask]
        mci_info = mci_info.loc[mci_mask]
        print(f"\nFiltered to MCI>0: {len(X_test)} samples (from {original_size})")
    elif filter_nci:
        mci_mask = (mci_info < 0)
        if mci_mask.sum() == 0:
            raise ValueError("No samples with MCI==0 found in the data")
        X_test = X_test.loc[mci_mask]
        y_test = y_test.loc[mci_mask] if hasattr(y_test, 'loc') else y_test[mci_mask]
        mci_info = mci_info.loc[mci_mask]
        print(f"\nFiltered to MCI==0: {len(X_test)} samples (from {original_size})")
    return X_test, y_test, mci_info


def load_model_and_scaler(saved_models_dir, model_name='TabPFN', mci=False, nci=False):
    """Load trained TabPFN model and scaler from saved files.
    
    Args:
        saved_models_dir: Directory containing saved models
        model_name: Name of the model to load (default: 'TabPFN')
        mci: Whether MCI filter was used
        nci: Whether NCI filter was used
        
    Returns:
        tuple: (model, scaler, clinical_data) loaded objects
    """
    print(f"Loading model: {model_name}")
    
    # Load model
    model_path = os.path.join(saved_models_dir, f'{model_name}_MCI=False_NCI=False.pkl')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"  ✓ Loaded model from: {model_path}")
    
    # Load scaler
    scaler_path = os.path.join(saved_models_dir, f'scaler_MCI=False_NCI=False.pkl')
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print(f"  ✓ Loaded scaler from: {scaler_path}")
    
    # Load clinical_data
    clinical_data_path = os.path.join(saved_models_dir, f'individual_models_clinical_data_MCI=False_NCI=False.pkl')
    clinical_data = {}
    if os.path.exists(clinical_data_path):
        with open(clinical_data_path, 'rb') as f:
            clinical_data = pickle.load(f)
        print(f"  ✓ Loaded clinical_data from: {clinical_data_path}")
    else:
        print(f"  ⚠ Clinical_data not found, continuing without it")
    
    return model, scaler, clinical_data


def load_test_data(data_path, test_data_path=None, test_targets_path=None):
    """Load test data for explanation.
    
    Args:
        data_path: Path to full dataset (if test splits not available)
        test_data_path: Path to test features CSV (optional)
        test_targets_path: Path to test targets CSV (optional)
        
    Returns:
        tuple: (X_test, y_test, feature_names, mci_info)
            - mci_info: Series with MCI values indexed by sample ID, or None if not available
    """
    if test_data_path and test_targets_path and os.path.exists(test_data_path) and os.path.exists(test_targets_path):
        print(f"Loading test data from saved splits...")
        X_test = pd.read_csv(test_data_path, index_col=0)
        y_test = pd.read_csv(test_targets_path, index_col=0)
        print(f"  Test features shape: {X_test.shape}")
        print(f"  Test targets shape: {y_test.shape}")
        
        # Extract MCI information from test data if available
        mci_info = None
        if 'MCI' in X_test.columns:
            mci_info = X_test['MCI']
            # # Remove MCI from feature columns if it's there
            # X_test = X_test.drop(columns=['MCI'])
            print(f"  ✓ Found MCI column in test data")
        elif 'MCI' in y_test.columns:
            mci_info = y_test['MCI']
            print(f"  ✓ Found MCI column in test targets")
        
        return X_test, y_test, list(X_test.columns), mci_info
    
    # Otherwise load from full dataset
    print(f"Loading full dataset from: {data_path}")
    df = pd.read_csv(data_path, index_col='ID')
    
    # Use the same filtering as in classification script
    # Participant exclusions are maintained in the private analysis configuration.
    OUTLIERS_IDS = []
    df = df[~df.index.isin(OUTLIERS_IDS)]
    df = df[df['p-tau217'].notna()]
    
    # Use informative features
    COGNITIVE_COLUMNS = ['Trail_B_(second)', 'IST', 'MOCA_score', 'Executive', 'Spatial ',
           'Attention_and_concentration', 'Memory', 'Language', 'Orientation',
           'Frontal memory score (#1)', 'Hipocampal memory score (#2)']
    BEHVIOURAL_COLUMNS = ['Mild_behavioral_Impairment_Checklist',
           'interest, motivation, and drive', 'mood or anxiety symptoms',
           'Impulse control', 'following societal norms',
           'strongly held beliefs and sensory', 'iADL']
    AGE_GENDER_COLUMNS = ['Age', 'Gender']
    BLOOD_TESTS = ['Hemoglobin_A1C', 'Cholesterol', 'Triglycerides', 'LDL', 'HDL', 'CRP', 'HB', 'RBC', 'WBC',
           'Lymphocytes', 'Creatinine', 'GFR']
    
    INFORMATIVE_FEATURES = (COGNITIVE_COLUMNS + ['Herpes_zoster_vaccine', 'BMI', 'Diabetes_controlled', 
                           'Diabetes', 'Hyperlipidemia'] + BEHVIOURAL_COLUMNS + 
                           AGE_GENDER_COLUMNS + BLOOD_TESTS)
    
    X = df[INFORMATIVE_FEATURES].dropna(axis=0)
    y_tau = df.loc[X.index, 'p-tau217']
    
    # Categorize target
    HIGH_TAU_217_THRESHOLD = 0.445
    y = (y_tau > HIGH_TAU_217_THRESHOLD).astype(int)
    
    # Extract MCI information if available in full dataset (fallback)
    mci_info = df.loc[X.index, 'MCI'] if 'MCI' in df.columns else None
    
    return X, y, list(X.columns), mci_info


def load_train_data(args, feature_names, scaler, X_test_scaled, y_test, mci_info=None):
    """Load training data if available; otherwise return test subset for background.
    Returns (X_train_scaled, y_train, has_train_and_full, mci_info, X_train_raw).
    X_train_raw is unscaled train (feature_names only) when has_train_and_full, else None.
    When train is loaded and mci_info is None, may set mci_info from train's MCI column.
    """
    if not args.train_data_path or not os.path.exists(args.train_data_path):
        y_train = y_test.iloc[:100] if hasattr(y_test, 'iloc') else y_test[:100]
        return X_test_scaled[:100], y_train, False, mci_info, None
    X_train = pd.read_csv(args.train_data_path, index_col=0)
    if mci_info is None and 'MCI' in X_train.columns:
        common = X_train.index.intersection(X_test_scaled.index)
        if len(common) > 0:
            mci_info = X_train.loc[common, 'MCI']
    X_train = X_train[feature_names]
    X_train_scaled = scaler.transform(X_train)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_names, index=X_train.index)
    y_train_path = args.train_data_path.replace('train_data.csv', 'train_targets.csv')
    if not os.path.exists(y_train_path):
        return X_test_scaled[:100], y_test.iloc[:100] if hasattr(y_test, 'iloc') else y_test[:100], False, mci_info, None
    y_train = pd.read_csv(y_train_path, index_col=0)
    if y_train.shape[1] > 1:
        y_train = y_train.iloc[:, 0]
    return X_train_scaled, y_train, True, mci_info, X_train


def build_full_dataset(X_train, X_test, y_train, y_test, feature_names, scaler):
    """Concatenate train+test, scale, and return (X_full_scaled, y_full)."""
    X_full = pd.concat([X_train, X_test], axis=0)
    y_full = pd.concat([y_train, y_test], axis=0)
    if hasattr(y_full, 'ndim') and y_full.ndim > 1:
        y_full = y_full.iloc[:, 0]
    X_full_scaled = scaler.transform(X_full)
    X_full_scaled = pd.DataFrame(X_full_scaled, columns=feature_names, index=X_full.index)
    return X_full_scaled, y_full


def run_explainability_for_dataset(dataset_name, X_data, y_data, data_dir, args, model,
                                   X_train_scaled, y_train, feature_names, mci_info):
    """Run predictions and all requested explainers (SHAP, SHAP-IQ, ALE) for one dataset."""
    print(f"\n{'#'*80}\n# EXPLAINABILITY FOR: {dataset_name.upper()} (n={len(X_data)})\n{'#'*80}")
    y_pred = model.predict(X_data)
    y_flat = y_data.values.flatten() if hasattr(y_data, 'values') else np.asarray(y_data).ravel()
    predictions_df = pd.DataFrame({'true_label': y_flat, 'predicted_label': y_pred}, index=X_data.index)
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X_data)
        for c in range(proba.shape[1]):
            predictions_df[f'prob_class_{c}'] = proba[:, c]
    predictions_df.to_csv(os.path.join(data_dir, 'predictions.csv'))
    print(f"  ✓ Saved predictions to: {data_dir}")

    if args.use_shap:
        explain_with_shap(
            model=model, X_train=X_train_scaled, X_test=X_data, feature_names=feature_names,
            output_dir=data_dir, n_samples=args.n_shap_samples, algorithm=args.shap_algorithm,
            precalculated_shap_path=args.precalculated_shap_values if dataset_name == 'test' else None,
            mci_info=mci_info if dataset_name == 'test' else None, top_n_plot=args.top_n_plot
        )
    if args.use_shapiq:
        explain_with_shapiq(
            model=model, X_train=X_train_scaled, y_train=y_train, X_test=X_data,
            feature_names=feature_names, output_dir=data_dir,
            n_explain=args.n_shapiq_samples, n_model_evals=args.n_model_evals, index='SV', max_order=2
        )
        if args.use_interactions:
            print(f"\n{'='*80}\nCALCULATING SHAPLEY INTERACTION VALUES ({dataset_name.upper()})\n{'='*80}")
            explain_with_shapiq(
                model=model, X_train=X_train_scaled, y_train=y_train, X_test=X_data,
                feature_names=feature_names, output_dir=data_dir,
                n_explain=args.n_shapiq_samples, n_model_evals=args.n_model_evals, index='FSII', max_order=2
            )
    if args.use_ale:
        features_to_explain = [f.strip() for f in args.ale_features.split(',')] if args.ale_features else None
        explain_with_ale(
            model=model, X_train=X_data, X_test=X_data, feature_names=feature_names, output_dir=data_dir,
            target_class=args.ale_target_class, n_bins=args.ale_n_bins,
            features_to_explain=features_to_explain, top_n_plot=args.top_n_plot
        )


def explain_with_shap(model, X_train, X_test, feature_names, output_dir, n_samples=None, algorithm='permutation', precalculated_shap_path=None, mci_info=None, top_n_plot=15):
    """Generate SHAP explanations using standard SHAP.
    
    Args:
        model: Trained TabPFN model (only needed if precalculated_shap_path is None)
        X_train: Training data (used as background, only needed if precalculated_shap_path is None)
        X_test: Test data to explain
        feature_names: List of feature names
        output_dir: Directory to save outputs
        n_samples: Number of samples to explain; None = use all samples (only used if precalculated_shap_path is None)
        algorithm: SHAP algorithm to use ('permutation' or 'exact', only used if precalculated_shap_path is None)
        precalculated_shap_path: Path to CSV file containing pre-calculated SHAP values (optional)
        mci_info: Series with MCI values for filtering precalculated SHAP values (optional)
        top_n_plot: Number of top features to show in plots (full results still saved to CSV)
    """
    if precalculated_shap_path is not None:
        print(f"\n{'='*80}\nLOADING PRE-CALCULATED SHAP VALUES\n{'='*80}\nLoading from: {precalculated_shap_path}")
        shap_values_mat, X_test_subset, shap_values = _load_precalculated_shap(
            precalculated_shap_path, X_test, feature_names, mci_info, n_samples
        )
        print(f"  ✓ Loaded SHAP values matrix shape: {shap_values_mat.shape}")
    else:
        print(f"\n{'='*80}\nCALCULATING SHAP VALUES (Standard SHAP)\n{'='*80}")
        if n_samples is None:
            X_test_subset = X_test
        else:
            n_use = min(n_samples, len(X_test))
            X_test_subset = X_test.iloc[:n_use] if isinstance(X_test, pd.DataFrame) else X_test[:n_use]
        print(f"Explaining {len(X_test_subset)} samples...")
        shap_values = tabpfn_shap.get_shap_values(
            estimator=model,
            test_x=X_test_subset,
            attribute_names=feature_names,
            algorithm=algorithm,
            background=X_train[:100] if len(X_train) > 100 else X_train,
        )
        shap_values_mat = _shap_values_to_matrix(shap_values)
        print(f"  ✓ SHAP values matrix shape: {shap_values_mat.shape}")

    # Save SHAP values (full feature set)
    shap_values_df = pd.DataFrame(
        shap_values_mat,
        columns=feature_names,
        index=(X_test_subset.index if isinstance(X_test_subset, pd.DataFrame) else range(len(X_test_subset)))
    )
    shap_values_path = os.path.join(output_dir, 'shap_values.csv')
    shap_values_df.to_csv(shap_values_path)
    print(f"  ✓ Saved SHAP values (full) to: {shap_values_path}")

    # Calculate mean absolute SHAP values (feature importance) and save full rankings
    mean_abs_shap = np.abs(shap_values_mat).mean(axis=0)
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': mean_abs_shap
    }).sort_values('mean_abs_shap', ascending=False)
    importance_path_csv = os.path.join(output_dir, 'shap_feature_importance.csv')
    importance_df.to_csv(importance_path_csv, index=False)
    print(f"  ✓ Saved feature importance rankings (full) to: {importance_path_csv}")

    # Top N features for plotting (full results already saved above)
    n_plot = min(top_n_plot, len(feature_names))
    top_features = importance_df.head(n_plot)['feature'].tolist()
    top_indices = [feature_names.index(f) for f in top_features]
    shap_values_top = shap_values_mat[:, top_indices]
    feature_names_top = [feature_names[i] for i in top_indices]
    importance_df_top = importance_df.head(n_plot)

    # Create visualization (tabpfn_extensions.plot_shap returns None and handles plotting internally)
    print(f"Creating SHAP visualizations (top {n_plot} features)...")
    # Only call plot_shap if we have the original shap_values object, not just the matrix
    if precalculated_shap_path is None and not isinstance(shap_values, np.ndarray):
        tabpfn_shap.plot_shap(shap_values)
    else:
        # For pre-calculated values or numpy arrays, use the matrix directly (top N only)
        try:
            shap.plots.bar(shap_values=shap_values_top, show=False)
            plt.title("Aggregate feature importances across the test examples (top %d)" % n_plot)
            bar_path = os.path.join(output_dir, 'shap_bar_plot.pdf')
            plt.savefig(bar_path, format='pdf', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Saved SHAP bar plot to: {bar_path}")
        except Exception as e:
            print(f"  ⚠ Could not create SHAP bar plot: {e}")

    # Beeswarm (dot) plot for per-feature distributions (top N only)
    try:
        if isinstance(X_test_subset, pd.DataFrame):
            X_for_plot = X_test_subset[top_features]
        else:
            X_for_plot = pd.DataFrame(X_test_subset[:, top_indices], columns=feature_names_top)
        plt.figure(figsize=(10, max(8, len(feature_names_top) * 0.3)))
        shap.summary_plot(shap_values_top, X_for_plot, feature_names=feature_names_top, show=False)
        plt.tight_layout()
        beeswarm_path = os.path.join(output_dir, 'shap_beeswarm_plot.pdf')
        plt.savefig(beeswarm_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved SHAP beeswarm plot to: {beeswarm_path}")
    except Exception as e:
        print(f"  ⚠ Could not create SHAP beeswarm plot: {e}")
        import traceback
        traceback.print_exc()

    # Plot feature importance (top N only)
    plt.figure(figsize=(10, max(8, len(importance_df_top) * 0.3)))
    sns.barplot(data=importance_df_top, y='feature', x='mean_abs_shap',
                palette='viridis', rasterized=False)
    plt.xlabel('Mean |SHAP value|')
    plt.ylabel('Feature')
    plt.title('Feature Importance (Mean Absolute SHAP Values, top %d)' % n_plot)
    plt.tight_layout()
    importance_path = os.path.join(output_dir, 'shap_feature_importance.pdf')
    plt.savefig(importance_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved feature importance plot to: {importance_path}")

    return shap_values, importance_df


def explain_with_shapiq(model, X_train, y_train, X_test, feature_names, output_dir, 
                        n_explain=5, n_model_evals=100, index='SV', max_order=2):
    """Generate SHAP-IQ explanations (more advanced, supports interactions).
    
    Args:
        model: Trained TabPFN model
        X_train: Training data
        y_train: Training labels
        X_test: Test data to explain
        feature_names: List of feature names
        output_dir: Directory to save outputs
        n_explain: Number of samples to explain
        n_model_evals: Budget for model evaluations (higher = more accurate but slower)
        index: SHAP index type ('SV' for Shapley Value, 'FSII' for interactions)
        max_order: Maximum order of interactions (for FSII)
    """
    print(f"\n{'='*80}")
    print(f"CALCULATING SHAP-IQ VALUES (index={index})")
    print(f"{'='*80}")
    
    # Limit samples for computation time
    X_test_array = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
    X_train_array = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
    
    X_explain = X_test_array[:n_explain]
    
    print(f"Explaining {len(X_explain)} samples...")
    print(f"Using {n_model_evals} model evaluations per sample...")
    print("This may take a while...")
    
    # Get TabPFN explainer
    explainer = tabpfn_shapiq.get_tabpfn_explainer(
        model=model,
        data=X_train_array,
        labels=y_train.values if isinstance(y_train, pd.Series) or isinstance(y_train, pd.DataFrame) else y_train,
        index=index,
        max_order=max_order if index == 'FSII' else 2,
        verbose=True,
    )
    
    # Explain each sample
    all_explanations = []
    for i, x_sample in enumerate(X_explain):
        print(f"\nExplaining sample {i+1}/{len(X_explain)}...")
        explanation = explainer.explain(x=x_sample, budget=n_model_evals)
        all_explanations.append(explanation)
        
        # Plot force plot for this sample
        try:
            fig = explanation.plot_force(feature_names=feature_names, show=False)
            if fig is not None:
                output_path = os.path.join(output_dir, f'shapiq_force_plot_sample_{i}_index_{index}.pdf')
                fig.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
                plt.close(fig)
                print(f"  ✓ Saved force plot to: {output_path}")
        except Exception as e:
            print(f"  ⚠ Could not create force plot: {e}")
        
        # For interaction plots (FSII)
        if index == 'FSII' and max_order >= 2:
            try:
                fig = explanation.plot_upset(feature_names=feature_names, show=False)
                if fig is not None:
                    output_path = os.path.join(output_dir, f'shapiq_upset_plot_sample_{i}.pdf')
                    fig.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    print(f"  ✓ Saved upset plot to: {output_path}")
            except Exception as e:
                print(f"  ⚠ Could not create upset plot: {e}")
    
    print(f"\n✓ Completed SHAP-IQ explanations for {len(X_explain)} samples")
    return all_explanations


def create_model_predictor(model, feature_names, target_class=None):
    """
    Create a prediction function wrapper for ALE explainer.
    
    ALE requires a callable function that takes a numpy array of features and returns
    predictions. For classification models, we can return either:
    - Class probabilities (predict_proba) - recommended for ALE
    - Class predictions (predict) - simpler but less informative
    
    Args:
        model: Trained TabPFN model with predict/predict_proba methods
        feature_names: List of feature names (needed to convert numpy to DataFrame for TabPFN)
        target_class: For classification, which class probability to return.
                     If None, returns probabilities for all classes (shape: n_samples, n_classes)
                     If specified (0 or 1), returns probabilities for that class only (shape: n_samples,)
    
    Returns:
        callable: Function that takes X (numpy array) and returns predictions/probabilities
    """
    def predictor(X):
        """
        Prediction function for ALE.
        
        This function is called by ALE many times with different feature values.
        ALE will vary one feature at a time and see how predictions change.
        
        Args:
            X: numpy array of shape (n_samples, n_features)
        
        Returns:
            numpy array: Predictions or probabilities
                - If target_class is None: shape (n_samples, n_classes) - probabilities for all classes
                - If target_class is specified: shape (n_samples,) - probabilities for that class
        """
        # Convert to DataFrame if needed (TabPFN might expect DataFrame)
        # ALE passes numpy arrays, but TabPFN might need DataFrames
        # We'll try DataFrame first, then fall back to numpy if needed
        try:
            # Try with DataFrame (TabPFN often expects this)
            X_df = pd.DataFrame(X, columns=feature_names)
            if hasattr(model, 'predict_proba') and target_class is not None:
                # Return probabilities for specific class (1D array)
                proba = model.predict_proba(X_df)
                return proba[:, target_class]
            elif hasattr(model, 'predict_proba'):
                # Return all class probabilities (2D array)
                return model.predict_proba(X_df)
            else:
                # Fallback to class predictions
                return model.predict(X_df)
        except (TypeError, AttributeError, ValueError):
            # If DataFrame doesn't work, try with numpy array directly
            if hasattr(model, 'predict_proba') and target_class is not None:
                proba = model.predict_proba(X)
                return proba[:, target_class]
            elif hasattr(model, 'predict_proba'):
                return model.predict_proba(X)
            else:
                return model.predict(X)
    
    return predictor


def explain_with_ale(model, X_train, X_test, feature_names, output_dir,
                     target_class=None, n_bins=10, features_to_explain=None, top_n_plot=15):
    """
    Generate ALE (Accumulated Local Effects) plots using alibi package.
    
    WHAT IS ALE?
    ------------
    ALE plots show how a feature affects model predictions on average, while accounting
    for correlations with other features. Unlike Partial Dependence Plots (PDP), ALE:
    1. Divides each feature into bins (intervals)
    2. For each bin, calculates the average change in prediction when the feature
       value is varied within that bin
    3. Accumulates these effects to show the overall effect of the feature
    4. Handles correlated features better than PDP by using conditional distributions
    
    HOW IT WORKS:
    ------------
    For a feature X_j:
    1. Divide the range of X_j into bins [z_{k-1}, z_k]
    2. For each bin, compute the average change in prediction when X_j changes
       from z_{k-1} to z_k, while keeping other features at their observed values
    3. Accumulate these changes: ALE_j(x) = sum of effects up to x
    4. Center the result so it has mean zero (for interpretability)
    
    INTERPRETATION:
    ---------------
    - Positive ALE value: Higher feature values increase predictions (on average)
    - Negative ALE value: Higher feature values decrease predictions (on average)
    - Steep slope: Strong effect of the feature
    - Flat line: Weak effect of the feature
    - Non-linear curves: Complex relationships (e.g., thresholds, interactions)
    
    Args:
        model: Trained TabPFN model
        X_train: Training data (used to determine feature ranges and distributions)
        X_test: Test data (can be used for additional analysis, but ALE is global)
        feature_names: List of feature names
        output_dir: Directory to save outputs
        target_class: For classification, which class to explain (0 or 1, or None for all classes)
        n_bins: Number of bins to divide each feature into (more bins = more detail but slower)
        features_to_explain: List of feature indices or names to explain.
                            If None, explains all features (can be slow for many features)
        top_n_plot: Number of top features to show in plots (full results still saved to CSV)

    Returns:
        dict: Dictionary containing ALE explainer object and results
    """

    print(f"\n{'='*80}")
    print("CALCULATING ALE (ACCUMULATED LOCAL EFFECTS)")
    print(f"{'='*80}")
    print("\nALE Explanation:")
    print("  - Shows how each feature affects predictions on average")
    print("  - Accounts for feature correlations (better than Partial Dependence Plots)")
    print("  - Provides global interpretability (not per-sample)")
    print(f"  - Using {n_bins} bins per feature")
    
    # Convert to numpy arrays (ALE expects numpy)
    # X_train is used to determine feature distributions and ranges
    if isinstance(X_train, pd.DataFrame):
        X_train_array = X_train.values
    else:
        X_train_array = X_train
    
    # Determine which features to explain
    if features_to_explain is None:
        # Explain all features
        feature_indices = list(range(len(feature_names)))
        print(f"  - Explaining all {len(feature_names)} features")
    else:
        # Convert feature names to indices if needed
        feature_indices = []
        for feat in features_to_explain:
            if isinstance(feat, str):
                if feat in feature_names:
                    feature_indices.append(feature_names.index(feat))
                else:
                    print(f"  ⚠ Warning: Feature '{feat}' not found, skipping")
            else:
                feature_indices.append(feat)
        print(f"  - Explaining {len(feature_indices)} selected features")
    
    # Create prediction function wrapper
    # For classification, we typically want class probabilities (not hard predictions)
    # target_class=None means we'll get probabilities for all classes
    print(f"\nCreating model predictor function...")
    if target_class is not None:
        print(f"  - Focusing on class {target_class} probabilities")
    else:
        print(f"  - Using probabilities for all classes")
    
    predictor = create_model_predictor(model, feature_names, target_class=target_class)
    
    # Determine target names for classification
    target_names = [f"Class_{i}" for i in range(len(model.classes_))] if hasattr(model, 'classes_') else None
    print(f"\nInitializing ALE explainer (data shape: {X_train_array.shape})...")
    # Try to create explainer with grid_size parameter (some versions support it in constructor)
    try:
        ale_explainer = ALE(
            predictor=predictor,  # Our wrapped prediction function
            feature_names=feature_names,  # For labeling plots
            target_names=target_names,  # For multi-class classification
            grid_size=n_bins  # Number of bins per feature (may not be supported in all versions)
        )
        print(f"  ✓ Created ALE explainer with {n_bins} bins per feature")
    except TypeError:
        # If grid_size is not supported in constructor, create without it
        ale_explainer = ALE(
            predictor=predictor,  # Our wrapped prediction function
            feature_names=feature_names,  # For labeling plots
            target_names=target_names  # For multi-class classification
        )
        print(f"  ✓ Created ALE explainer (using default bins)")
    
    # Explain features
    # The explain() method computes the ALE values for each feature
    # Unlike some explainers, ALE doesn't need a separate fit() call
    # It analyzes the data and computes ALE values in one step
    print(f"\nComputing ALE values...")
    print("  - This may take a while for many features or large datasets")
    print("  - ALE evaluates the model many times to compute average effects")
    print("  - ALE analyzes feature distributions and computes effects in one step")
    
    # Explain only selected features
    # The 'features' parameter specifies which features to explain
    # The grid_size parameter (if supported) controls the number of bins
    try:
        ale_explanations = ale_explainer.explain(
            X_train_array,  # Data to explain (ALE uses this to understand feature distributions)
            features=feature_indices,  # Which features to explain
            grid_size=n_bins  # Number of bins per feature (may not be supported in all versions)
        )
    except TypeError:
        # If grid_size is not supported in explain(), try without it
        # The bins might have been set in constructor or use default
        print(f"  ⚠ Note: grid_size parameter not supported in explain(). Using bins from constructor or default.")
        ale_explanations = ale_explainer.explain(
            X_train_array,  # Data to explain (ALE uses this to understand feature distributions)
            features=feature_indices  # Which features to explain
        )
    
    print("  ✓ ALE computation complete")
    
    # The ale_explanations object contains:
    # - ale_values: The ALE values for each feature (shape depends on target_class)
    # - ale0: The baseline (centering constant)
    # - feature_values: The bin boundaries for each feature
    # - feature_names: Feature names
    
    # Save ALE values to CSV
    print(f"\nSaving ALE results...")
    
    # Extract ALE values
    # Shape depends on whether we're explaining one class or all classes
    ale_values = ale_explanations.ale_values
    
    # Create output directory for ALE plots
    ale_output_dir = os.path.join(output_dir, 'ale_plots')
    os.makedirs(ale_output_dir, exist_ok=True)
    
    # Save ALE values for each feature (per-class when 2D so alignment is valid)
    for i, feat_idx in enumerate(feature_indices):
        feat_name = feature_names[feat_idx]
        feat_ale = ale_values[i]
        feat_bins = ale_explanations.feature_values[i]
        feat_ale_arr = np.asarray(feat_ale)
        if feat_ale_arr.ndim > 1:
            for class_idx in range(feat_ale_arr.shape[1]):
                bins_1d, ale_1d = _align_ale_bins_and_values(feat_ale_arr[:, class_idx], feat_bins, feat_name)
                ale_df = pd.DataFrame({'feature_value': bins_1d, 'ale_value': ale_1d})
                ale_df.to_csv(os.path.join(ale_output_dir, f'ale_{feat_name}_class{class_idx}.csv'), index=False)
        else:
            bins_1d, ale_1d = _align_ale_bins_and_values(feat_ale, feat_bins, feat_name)
            ale_df = pd.DataFrame({'feature_value': bins_1d, 'ale_value': ale_1d})
            ale_df.to_csv(os.path.join(ale_output_dir, f'ale_{feat_name}.csv'), index=False)
    
    print(f"  ✓ Saved ALE values to: {ale_output_dir}")

    # Compute importance (needed for top-N plotting) and save full rankings
    ale_importance = []
    for i, feat_idx in enumerate(feature_indices):
        feat_name = feature_names[feat_idx]
        if isinstance(ale_values, list):
            feat_ale = ale_values[i]
        else:
            feat_ale = ale_values[i]
        if feat_ale.ndim > 1:
            variance = np.max([np.var(feat_ale[:, j]) for j in range(feat_ale.shape[1])])
            range_val = np.max([np.max(feat_ale[:, j]) - np.min(feat_ale[:, j])
                              for j in range(feat_ale.shape[1])])
        else:
            variance = np.var(feat_ale)
            range_val = np.max(feat_ale) - np.min(feat_ale)
        ale_importance.append({
            'feature': feat_name,
            'ale_variance': variance,
            'ale_range': range_val
        })
    importance_df = pd.DataFrame(ale_importance).sort_values('ale_variance', ascending=False)
    importance_path = os.path.join(ale_output_dir, 'ale_feature_importance.csv')
    importance_df.to_csv(importance_path, index=False)
    print(f"  ✓ Saved ALE feature importance (full) to: {importance_path}")

    # Top N features for plotting (full results already saved above)
    n_plot = min(top_n_plot, len(feature_indices))
    importance_df_top = importance_df.head(n_plot)
    top_feature_names = importance_df_top['feature'].tolist()
    # Indices into feature_indices/ale_values for the top N (by importance order)
    top_plot_positions = [feature_indices.index(feature_names.index(f)) for f in top_feature_names]

    # Create visualizations (top N only)
    print(f"\nCreating ALE plots (top {n_plot} features)...")
    
    # Plot individual ALE plots for top features only
    n_features = n_plot
    
    # Determine grid size for subplots
    n_cols = min(3, n_features)  # Max 3 columns
    n_rows = (n_features + n_cols - 1) // n_cols  # Ceiling division
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_features == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Build interpretable class labels for legend (Class 0/1 = target labels 0/1 = low/high p-tau217)
    if hasattr(model, 'classes_') and model.classes_ is not None:
        cl = np.asarray(model.classes_)
        if len(cl) == 2 and set(cl) <= {0, 1}:
            legend_labels = ['Low p-tau217 (0)', 'High p-tau217 (1)']
        else:
            legend_labels = [f'Class {c}' for c in cl]
    else:
        n_cl = np.asarray(ale_values[0]).shape[1] if np.asarray(ale_values[0]).ndim > 1 else 1
        cl = np.arange(n_cl)
        legend_labels = [f'Class {i}' for i in range(n_cl)]
    
    # Plot each of the top features
    for plot_idx, pos in enumerate(top_plot_positions):
        feat_idx = feature_indices[pos]
        feat_name = feature_names[feat_idx]
        feat_ale = ale_values[pos]
        feat_bins = ale_explanations.feature_values[pos]
        feat_bins_to_use, feat_ale_flat = _align_ale_bins_and_values(feat_ale, feat_bins)
        ax = axes[plot_idx]
        if np.asarray(feat_ale).ndim > 1:
            for class_idx in range(np.asarray(feat_ale).shape[1]):
                bins_c, ale_c = _align_ale_bins_and_values(feat_ale[:, class_idx], feat_bins)
                label = legend_labels[class_idx] if class_idx < len(legend_labels) else f'Class {class_idx}'
                class_label = int(cl[class_idx]) if class_idx < len(cl) else class_idx
                ax.plot(
                    bins_c, ale_c, label=label, linewidth=2,
                    color=TAU_CLASS_COLORS.get(class_label),
                    rasterized=False,
                )
            ax.legend()
        else:
            ax.plot(feat_bins_to_use, feat_ale_flat, linewidth=2,
                    color=TAU_CLASS_COLORS.get(target_class, 'steelblue'),
                    rasterized=False)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, rasterized=False)
        ax.set_xlabel(feat_name, fontsize=10)
        ax.set_ylabel('ALE Effect', fontsize=10)
        ax.set_title(f'ALE Plot: {feat_name}', fontsize=11, fontweight='bold')
    
    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    # Save combined plot (top N only)
    combined_path = os.path.join(ale_output_dir, 'ale_plots_combined.pdf')
    plt.savefig(combined_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved combined ALE plots to: {combined_path}")

    # Plot feature importance (top N only) on log scale
    fig, ax = plt.subplots(figsize=(10, max(8, len(importance_df_top) * 0.3)))
    sns.barplot(data=importance_df_top, y='feature', x='ale_variance',
                palette='viridis', rasterized=False, ax=ax)
    ax.set_xscale('log')
    ax.set_xlabel('ALE Variance (Feature Importance, log scale)')
    ax.set_ylabel('Feature')
    ax.set_title('Feature Importance Based on ALE Variance (top %d)\n(Higher variance = stronger effect)' % n_plot)
    plt.tight_layout()
    importance_plot_path = os.path.join(ale_output_dir, 'ale_feature_importance.pdf')
    plt.savefig(importance_plot_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved ALE importance plot to: {importance_plot_path}")
    
    print(f"\n{'='*80}")
    print("ALE EXPLANATIONS COMPLETE")
    print(f"{'='*80}")
    print(f"All ALE outputs saved to: {ale_output_dir}")
    
    return {
        'explainer': ale_explainer,
        'explanations': ale_explanations,
        'importance_df': importance_df
    }


def main(args):
    """Main function to load models and generate explanations."""
    print("="*80)
    print("TABPFN MODEL EXPLAINABILITY")
    print("="*80)
    
    # Setup output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Setup saved models directory
    saved_models_dir = os.path.join(args.saved_models_dir, 'saved_models')
    if not os.path.exists(saved_models_dir):
        raise FileNotFoundError(f"Saved models directory not found: {saved_models_dir}")
    
    # Load model and scaler
    model, scaler, clinical_data = load_model_and_scaler(
        saved_models_dir,
        model_name=args.model_name,
        mci=args.mci,
        nci=args.nci
    )
    
    # Get feature names from clinical_data or test data
    if 'feature_names' in clinical_data:
        feature_names = clinical_data['feature_names']
        print(f"  ✓ Using feature names from clinical_data ({len(feature_names)} features)")
    else:
        feature_names = None  # Will get from data
    
    # Load test data
    X_test, y_test, data_feature_names, mci_info = load_test_data(
        args.data_path,
        test_data_path=args.test_data_path,
        test_targets_path=args.test_targets_path
    )
    
    if feature_names is None:
        feature_names = data_feature_names

    X_test, y_test, mci_info = apply_mci_filter(X_test, y_test, mci_info, args.mci, args.nci)
    print("\nApplying feature scaling...")
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=feature_names, index=X_test.index
    )

    print(f"\nLoading training data from: {args.train_data_path}" if (args.train_data_path and os.path.exists(args.train_data_path)) else "\n  ⚠ Training data not found; will only run explainability on test set")
    X_train_scaled, y_train, has_train_and_full, mci_info, X_train = load_train_data(
        args, feature_names, scaler, X_test_scaled, y_test, mci_info
    )
    if has_train_and_full:
        print(f"  ✓ Train set: {len(X_train_scaled)} samples")
        X_full_scaled, y_full = build_full_dataset(X_train, X_test, y_train, y_test, feature_names, scaler)
        print(f"  ✓ Full dataset: {len(X_full_scaled)} samples (train + test)")

    test_dir = os.path.join(output_dir, 'test')
    train_dir = os.path.join(output_dir, 'train')
    full_dir = os.path.join(output_dir, 'full')
    os.makedirs(test_dir, exist_ok=True)
    if has_train_and_full:
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(full_dir, exist_ok=True)
    print(f"\nOutput subdirs: test -> {test_dir}" + (f" train -> {train_dir} full -> {full_dir}" if has_train_and_full else " (train/full skipped: no train data)"))

    datasets = [('test', X_test_scaled, y_test, test_dir)]
    if has_train_and_full:
        datasets += [('train', X_train_scaled, y_train, train_dir), ('full', X_full_scaled, y_full, full_dir)]
    for dataset_name, X_data, y_data, data_dir in datasets:
        run_explainability_for_dataset(
            dataset_name, X_data, y_data, data_dir, args, model,
            X_train_scaled, y_train, feature_names, mci_info
        )

    print(f"\n{'='*80}\nEXPLANATIONS COMPLETE\n{'='*80}\nOutputs saved per dataset: {output_dir} (test/, train/, full/)")


def parse_args():
    """Build argument parser and return validated args."""
    parser = argparse.ArgumentParser(
        description='Generate SHAP, SHAP-IQ, and ALE explanations for trained TabPFN classification models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic SHAP explanation
  python tabpfn_explainability.py --saved_models_dir biomarker_prediction/figures/classification
  # All methods (plot top 15; full results in CSVs)
  python tabpfn_explainability.py --saved_models_dir biomarker_prediction/figures/classification --use_shap --use_ale --use_shapiq --top_n_plot 15
        """
    )
    parser.add_argument('--saved_models_dir', type=str,
                        default='/ems/elsc-labs/habib-n/yuval.rom/BCG/modeling/biomarker_prediction/figures/classification',
                        help='Directory containing saved_models subdirectory')
    parser.add_argument('--output_dir', type=str,
                        default='/ems/elsc-labs/habib-n/yuval.rom/BCG/modeling/biomarker_prediction/figures/classification/explainability',
                        help='Directory to save explanation outputs')
    parser.add_argument('--model_name', type=str, default='TabPFN', help='Name of the model to explain')
    parser.add_argument('--mci', action='store_true', help='Filter samples to MCI==1')
    parser.add_argument('--nci', action='store_true', help='Filter samples to MCI==0')
    parser.add_argument('--data_path', type=str,
                        default='/ems/elsc-labs/habib-n/yuval.rom/BCG/clinical_data/Data/model_preprocessed_data.csv',
                        help='Path to full dataset (if test splits not provided)')
    parser.add_argument('--test_data_path', type=str,
                        default='/ems/elsc-labs/habib-n/yuval.rom/BCG/modeling/biomarker_prediction/figures/classification/test_data.csv',
                        help='Path to test features CSV')
    parser.add_argument('--test_targets_path', type=str,
                        default='/ems/elsc-labs/habib-n/yuval.rom/BCG/modeling/biomarker_prediction/figures/classification/test_targets.csv',
                        help='Path to test targets CSV')
    parser.add_argument('--train_data_path', type=str,
                        default='/ems/elsc-labs/habib-n/yuval.rom/BCG/modeling/biomarker_prediction/figures/classification/train_data.csv',
                        help='Path to training data CSV')
    parser.add_argument('--use_shap', action='store_true', help='Use standard SHAP')
    parser.add_argument('--use_shapiq', action='store_true', help='Use SHAP-IQ')
    parser.add_argument('--use_interactions', action='store_true', help='Calculate Shapley interactions (FSII) with SHAP-IQ')
    parser.add_argument('--use_ale', action='store_true', help='Use ALE')
    parser.add_argument('--n_shap_samples', type=int, default=None, help='Samples to explain with SHAP; default: all')
    parser.add_argument('--n_shapiq_samples', type=int, default=3, help='Samples to explain with SHAP-IQ')
    parser.add_argument('--n_model_evals', type=int, default=100, help='Model evals per sample (SHAP-IQ)')
    parser.add_argument('--shap_algorithm', type=str, default='permutation', choices=['permutation', 'exact'])
    parser.add_argument('--precalculated_shap_values', type=str, default=None, help='Path to pre-calculated SHAP CSV')
    parser.add_argument('--ale_n_bins', type=int, default=10, help='Bins for ALE')
    parser.add_argument('--ale_target_class', type=int, default=None, help='Class to explain with ALE (0 or 1)')
    parser.add_argument('--ale_features', type=str, default=None, help='Comma-separated feature names for ALE')
    parser.add_argument('--top_n_plot', type=int, default=15, help='Top N features in plots; full results in CSVs')
    args = parser.parse_args()
    if args.mci and args.nci:
        raise ValueError("Cannot specify both --mci and --nci.")
    return args


if __name__ == '__main__':
    main(parse_args())

