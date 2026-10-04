"""
Main script for tau-217 classification model training and evaluation.
This script orchestrates the entire pipeline using modular components.
"""

import os
import pickle
import argparse
import warnings
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from tabpfn import TabPFNClassifier

# Import modular components
from config import (
    TAU_217, MOCA_COL, MCI_THR, N_TRIALS,
    TRAIN_TEST_SPLIT, VAL_TEST_SPLIT, RANDOM_STATE
)
from preprocessing import (
    remove_outliers, filter_data_for_classification, categorize_data,
    filter_by_cognitive_status, scale_features, calculate_scaled_moca_threshold
)
from visualization import (
    plot_confusion_matrix, plot_model_comparison, plot_pred_proba_vs_ptau217,
    plot_combined_confusion_matrices
)
from hyperparameter_optimization import (
    optimize_random_forest, optimize_catboost
)
from model_training import train_and_evaluate_model, compute_class_weights
from bootstrap_evaluation import run_bootstrap_evaluation
from utils import evaluate_metrics
from baseline_classifier import BaselineClassifier
from ensemble import (
    create_ensemble_models,
    features_for_model,
    plot_ensemble_comparison,
    plot_ensemble_confusion_matrices,
    save_ensemble_results,
    save_ensemble_models
)

warnings.filterwarnings('ignore')
np.random.seed(RANDOM_STATE)

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'


def initialize_cuda(use_gpu):
    """Initialize CUDA for TabPFN/TabFM if GPU is requested.
    
    Parameters:
    -----------
    use_gpu : bool
        Whether to use GPU
    """
    if use_gpu:
        try:
            if torch.cuda.is_available():
                torch.zeros(1).cuda()
                print("CUDA initialized successfully")
            else:
                print("Warning: GPU requested but not available. Using CPU instead.")
        except Exception as e:
            print(f"Warning: Could not initialize CUDA: {e}")


def create_models(best_rf_params, best_cb_params_f1, best_cb_params_ber,
                  class_weights_combined, moca_threshold_scaled, use_gpu,
                  use_tabfm=True, use_tabfm_ensemble=False):
    """Create model instances with optimized parameters.
    
    Parameters:
    -----------
    best_rf_params : dict
        Best RandomForest hyperparameters
    best_cb_params_f1 : dict
        Best CatBoost F1 hyperparameters
    best_cb_params_ber : dict
        Best CatBoost BER hyperparameters
    class_weights_combined : dict
        Class weights for combined training set
    moca_threshold_scaled : float
        Scaled MOCA threshold for BaselineClassifier
    use_gpu : bool
        Whether to use GPU for TabPFN/TabFM
    use_tabfm : bool
        Whether to include TabFM in the model list
    use_tabfm_ensemble : bool
        Whether to use TabFM's stronger ensemble preset
        
    Returns:
    --------
    list
        List of (model_name, model) tuples
    """
    models = [
        ("LogisticRegression", LogisticRegression(
            random_state=RANDOM_STATE,
            class_weight='balanced',
            n_jobs=-1
        )),
        ("RandomForest", RandomForestClassifier(
            random_state=RANDOM_STATE,
            **best_rf_params,
            bootstrap=True,
            class_weight='balanced',
            n_jobs=-1
        )),
        ("CatBoost_F1", CatBoostClassifier(
            random_state=RANDOM_STATE,
            **best_cb_params_f1,
            verbose=0,
            thread_count=-1,
            eval_metric='F1',
            allow_writing_files=False,
            class_weights=class_weights_combined,
            task_type='CPU'
        )),
        ("CatBoost_BER", CatBoostClassifier(
            random_state=RANDOM_STATE,
            **best_cb_params_ber,
            verbose=0,
            thread_count=-1,
            eval_metric='BalancedAccuracy',
            allow_writing_files=False,
            class_weights=class_weights_combined,
            task_type='CPU'
        )),
        ("CatBoost_Default", CatBoostClassifier(
            random_state=RANDOM_STATE,
            verbose=0,
            thread_count=-1,
            eval_metric='F1',
            class_weights=class_weights_combined,
            task_type='CPU'
        )),
        ("TabPFN", TabPFNClassifier(
            device='cuda' if (use_gpu and torch.cuda.is_available()) else 'cpu',
            balance_probabilities=True,
            random_state=RANDOM_STATE,
        )),
    ]
    if use_tabfm:
        from tabfm_model import create_tabfm_classifier
        models.append((
            "TabFM",
            create_tabfm_classifier(
                use_gpu=use_gpu,
                use_ensemble_preset=use_tabfm_ensemble,
                random_state=RANDOM_STATE,
            ),
        ))
    models.append(
        ("Baseline", BaselineClassifier(
            moca_col=MOCA_COL,
            mci_thr=MCI_THR,
            scaled_threshold=moca_threshold_scaled
        )),
    )
    return models


def save_data_splits(X_train_combined, y_train_combined, X_test_norm, y_test, output_dir):
    """Save train and test data splits to CSV files.
    
    Parameters:
    -----------
    X_train_combined : pd.DataFrame
        Combined training features
    y_train_combined : pd.DataFrame
        Combined training targets
    X_test_norm : pd.DataFrame
        Normalized test features
    y_test : pd.DataFrame
        Test targets
    output_dir : str
        Output directory
    """
    X_train_combined.to_csv(os.path.join(output_dir, 'train_data.csv'))
    y_train_combined.to_csv(os.path.join(output_dir, 'train_targets.csv'))
    print(f"Train + validation features saved to: {os.path.join(output_dir, 'train_data.csv')}")
    print(f"Train + validation targets saved to: {os.path.join(output_dir, 'train_targets.csv')}")
    print(f"Train + validation features shape: {X_train_combined.shape}")
    print(f"Train + validation targets shape: {y_train_combined.shape}")

    X_test_norm.to_csv(os.path.join(output_dir, 'test_data.csv'))
    y_test.to_csv(os.path.join(output_dir, 'test_targets.csv'))
    print(f"Test features saved to: {os.path.join(output_dir, 'test_data.csv')}")
    print(f"Test targets saved to: {os.path.join(output_dir, 'test_targets.csv')}")
    print(f"Test features shape: {X_test_norm.shape}")
    print(f"Test targets shape: {y_test.shape}")


def save_model_results(best_results, best_rf_params, best_cb_params_f1, best_cb_params_ber,
                       X_train_combined, y_train_combined, y_test, scaler, output_dir, args):
    """Save model results, metrics, and models.
    
    Parameters:
    -----------
    best_results : list
        List of model result tuples
    best_rf_params : dict
        Best RandomForest parameters
    best_cb_params_f1 : dict
        Best CatBoost F1 parameters
    best_cb_params_ber : dict
        Best CatBoost BER parameters
    X_train_combined : pd.DataFrame
        Combined training features
    y_train_combined : pd.DataFrame
        Combined training targets
    y_test : pd.DataFrame
        Test targets
    scaler : StandardScaler
        Fitted scaler
    output_dir : str
        Output directory
    args : argparse.Namespace
        Command line arguments
    """
    # Save best parameters and metrics
    with open(os.path.join(output_dir, f'best_paramaters_MCI={args.mci}_NCI={args.nci}.txt'), 'w') as f:
        for result_tuple in best_results:
            if len(result_tuple) == 6:
                name, _, _, _, train_metrics, test_metrics = result_tuple
                train_auc_str = f"{train_metrics[4]:.4f}" if train_metrics[4] is not None else "N/A"
                train_prauc_str = f"{train_metrics[5]:.4f}" if train_metrics[5] is not None else "N/A"
                test_auc_str = f"{test_metrics[4]:.4f}" if test_metrics[4] is not None else "N/A"
                test_prauc_str = f"{test_metrics[5]:.4f}" if test_metrics[5] is not None else "N/A"
                f.write(f"{name}:\n")
                f.write(f"  Train: Accuracy={train_metrics[0]:.4f}, Precision={train_metrics[1]:.4f}, "
                       f"Recall={train_metrics[2]:.4f}, F1={train_metrics[3]:.4f}, "
                       f"AUC={train_auc_str}, PRAUC={train_prauc_str}\n")
                f.write(f"  Test: Accuracy={test_metrics[0]:.4f}, Precision={test_metrics[1]:.4f}, "
                       f"Recall={test_metrics[2]:.4f}, F1={test_metrics[3]:.4f}, "
                       f"AUC={test_auc_str}, PRAUC={test_prauc_str}\n")
            else:
                name, _, _, _, test_metrics = result_tuple[:5]
                test_auc_str = f"{test_metrics[4]:.4f}" if test_metrics[4] is not None else "N/A"
                test_prauc_str = f"{test_metrics[5]:.4f}" if test_metrics[5] is not None else "N/A"
                f.write(f"{name}: Accuracy={test_metrics[0]:.4f}, Precision={test_metrics[1]:.4f}, "
                       f"Recall={test_metrics[2]:.4f}, F1={test_metrics[3]:.4f}, "
                       f"AUC={test_auc_str}, PRAUC={test_prauc_str}\n")
        f.write(f"CatBoost F1 best params: {best_cb_params_f1}\n")
        f.write(f"CatBoost BER best params: {best_cb_params_ber}\n")
    
    # Save classification reports
    results_summary_path = os.path.join(output_dir, f'results_summary_MCI={args.mci}_NCI={args.nci}.txt')
    with open(results_summary_path, 'w') as f:
        for result_tuple in best_results:
            if len(result_tuple) == 6:
                model_name, model, y_train_pred, y_test_pred, train_metrics, test_metrics = result_tuple
            else:
                model_name, model, y_train_pred, y_test_pred, test_metrics = result_tuple[:5]
                train_metrics = None
            
            f.write(f"{model_name} Training Metrics:\n")
            f.write(classification_report(y_train_combined, y_train_pred))
            if train_metrics and train_metrics[4] is not None:
                f.write(f"Train AUC: {train_metrics[4]:.4f}\n")
            f.write("\n")
            f.write(f"{model_name} Test Metrics:\n")
            f.write(classification_report(y_test, y_test_pred))
            if test_metrics[4] is not None:
                f.write(f"Test AUC: {test_metrics[4]:.4f}\n")
            f.write("\n" + "="*80 + "\n\n")
    
    # Save individual models
    print("\n" + "="*80)
    print("SAVING INDIVIDUAL MODELS")
    print("="*80)
    
    models_dir = os.path.join(output_dir, 'saved_models')
    os.makedirs(models_dir, exist_ok=True)
    
    for result_tuple in best_results:
        model_name = result_tuple[0]
        model = result_tuple[1]
        model_path = os.path.join(models_dir, f'{model_name}_MCI={args.mci}_NCI={args.nci}.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Saved {model_name} to: {model_path}")
    
    # Save scaler (needed for inference)
    scaler_path = os.path.join(models_dir, f'scaler_MCI={args.mci}_NCI={args.nci}.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Saved scaler to: {scaler_path}")
    
    # Save model clinical_data (best parameters, etc.)
    clinical_data_path = os.path.join(models_dir, f'individual_models_clinical_data_MCI={args.mci}_NCI={args.nci}.pkl')
    individual_clinical_data = {
        'best_rf_params': best_rf_params,
        'best_cb_params_f1': best_cb_params_f1,
        'best_cb_params_ber': best_cb_params_ber,
        'model_names': [result_tuple[0] for result_tuple in best_results],
        'test_metrics': {result_tuple[0]: (result_tuple[5] if len(result_tuple) == 6 else result_tuple[4]) 
                         for result_tuple in best_results},
        'feature_names': list(X_train_combined.columns),
    }
    with open(clinical_data_path, 'wb') as f:
        pickle.dump(individual_clinical_data, f)
    print(f"Saved model clinical_data to: {clinical_data_path}")


def save_top_model_predictions(best_results, X_test_norm, y_test, df, output_dir, args):
    """Save predictions from the top-performing single model on the test set.
    
    Parameters:
    -----------
    best_results : list
        List of model result tuples
    X_test_norm : pd.DataFrame
        Normalized test features (to get indices)
    y_test : pd.DataFrame
        Test targets
    df : pd.DataFrame
        Original dataframe with p-tau217 values
    output_dir : str
        Output directory
    args : argparse.Namespace
        Command line arguments
    """
    # Find top-performing model by F1 score and extract needed variables
    best_f1 = -1
    best_model_name = None
    best_model = None
    best_y_test_pred = None
    
    for idx, result_tuple in enumerate(best_results):
        if len(result_tuple) == 6:
            model_name, model, _, y_test_pred, _, test_metrics = result_tuple
        else:
            model_name, model, _, y_test_pred, test_metrics = result_tuple[:5]
        
        f1_score = test_metrics[3]  # F1 is at index 3
        if f1_score > best_f1:
            best_f1 = f1_score
            best_model_name = model_name
            best_model = model
            best_y_test_pred = y_test_pred
    
    print(f"\nTop-performing model: {best_model_name} (F1: {best_f1:.4f})")
    
    # Get IDs from test set (indices are preserved in normalized DataFrame)
    test_ids = X_test_norm.index
    # Get p-tau217 values
    ptau217_values = df.loc[test_ids, TAU_217]
    
    # Extract ground truth values (handle DataFrame format)
    # Note: y_test and X_test_norm come from the same train_test_split, so order is preserved
    if isinstance(y_test, pd.DataFrame):
        ground_truth = y_test.iloc[:, 0].values
    else:
        ground_truth = np.array(y_test) if not isinstance(y_test, np.ndarray) else y_test
    
    # Create results DataFrame
    predictions_df = pd.DataFrame({
        'ID': test_ids,
        'p-tau217': ptau217_values.values,
        'predicted_class': best_y_test_pred,
        'ground_truth': ground_truth
    })
    
    # Save to CSV
    output_path = os.path.join(
        output_dir, 
        f'top_model_predictions_{best_model_name}_MCI={args.mci}_NCI={args.nci}.csv'
    )
    predictions_df.to_csv(output_path, index=False)
    print(f"Saved top model predictions to: {output_path}")
    print(f"Predictions shape: {predictions_df.shape}")


def print_ensemble_summary(best_results, ensemble_results):
    """Print ensemble model summary statistics.
    
    Parameters:
    -----------
    best_results : list
        List of model result tuples
    ensemble_results : dict
        Ensemble results dictionary
    """
    print("\n" + "="*80)
    print("ENSEMBLE MODEL SUMMARY")
    print("="*80)
    individual_f1 = []
    individual_test_auc = []
    for result_tuple in best_results:
        if len(result_tuple) == 6:
            _, _, _, _, _, test_metrics = result_tuple
        else:
            _, _, _, _, test_metrics = result_tuple[:5]
        individual_f1.append(test_metrics[3])
        if test_metrics[4] is not None:
            individual_test_auc.append(test_metrics[4])
    print(f"Best Individual Model F1: {max(individual_f1):.4f}")
    print(f"Average Individual Model F1: {np.mean(individual_f1):.4f}")
    if individual_test_auc:
        print(f"Best Individual Model Test AUC: {max(individual_test_auc):.4f}")
        print(f"Average Individual Model Test AUC: {np.mean(individual_test_auc):.4f}")
    print(f"Majority Vote Ensemble F1: {ensemble_results['majority_metrics'][3]:.4f}")
    print(f"Weighted Vote Ensemble F1: {ensemble_results['weighted_metrics'][3]:.4f}")
    print(f"Ensemble Improvement (Majority): {ensemble_results['majority_metrics'][3] - np.mean(individual_f1):.4f}")
    print(f"Ensemble Improvement (Weighted): {ensemble_results['weighted_metrics'][3] - np.mean(individual_f1):.4f}")


def main(args):
    """Main pipeline for tau-217 classification.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command line arguments
    """
    # Load and preprocess data
    print("Loading data...")
    df = pd.read_csv(args.data_path, index_col='ID')
    print(f"Initial data shape: {df.shape}")
    
    df = remove_outliers(df)
    print(f"Data shape after removing outliers: {df.shape}")
    
    df = df[df[TAU_217].notna()]
    df = filter_by_cognitive_status(df, mci_only=args.mci, nci_only=args.nci)
    print(f"Data shape after filtering: {df.shape}, arg.mci={args.mci}, arg.nci={args.nci}")

    # Prepare features and target
    X, y = filter_data_for_classification(df)
    y = categorize_data(y)
    print(f"Features shape: {X.shape}, Target shape: {y.shape}")
    print("\nClass distribution before balancing:")
    print(y[y.columns[0]].value_counts(normalize=True))
    
    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Split data into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=TRAIN_TEST_SPLIT, random_state=RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=VAL_TEST_SPLIT, random_state=RANDOM_STATE, stratify=y_temp
    )
    
    # Scale features
    X_train_norm, X_val_norm, X_test_norm, scaler = scale_features(
        X_train, X_val, X_test, return_scaler=True
    )
    
    # Calculate scaled threshold for BaselineClassifier
    moca_threshold_scaled = calculate_scaled_moca_threshold(X_train)
    print(f"Original MOCA threshold: {MCI_THR}")
    print(f"Calculated scaled threshold: {moca_threshold_scaled:.4f}")

    # Compute class weights
    class_weights = compute_class_weights(y_train)

    # Combine training and validation sets for final model training
    X_train_combined = pd.concat([X_train_norm, X_val_norm], ignore_index=True)
    X_train_combined_raw = pd.concat([X_train, X_val], ignore_index=True)
    y_train_combined = pd.concat([y_train, y_val], ignore_index=True)
    
    # Save data splits
    save_data_splits(X_train_combined, y_train_combined, X_test_norm, y_test, output_dir)

    # Recompute class weights for combined training set
    class_weights_combined = compute_class_weights(y_train_combined)

    # Hyperparameter Optimization
    print("\n--- Running Hyperparameter Optimization with Optuna ---")
    best_rf_params, best_rf_f1 = optimize_random_forest(
        X_train_norm, y_train, X_val_norm, y_val, n_trials=N_TRIALS
    )
    print(f"Best RandomForest params: {best_rf_params}, Best Validation F1: {best_rf_f1:.4f}")

    best_cb_params_f1, best_cb_f1 = optimize_catboost(
        X_train_norm, y_train, X_val_norm, y_val, class_weights, args.use_gpu, n_trials=N_TRIALS
    )
    print(f"Best CatBoost params (F1): {best_cb_params_f1}, Best Validation F1: {best_cb_f1:.4f}")

    best_cb_params_ber = best_cb_params_f1
    print(f"Using F1-optimized params for CatBoost (BER) as well.")

    # Initialize CUDA
    initialize_cuda(args.use_gpu)

    # Create and train models
    models = create_models(
        best_rf_params, best_cb_params_f1, best_cb_params_ber,
        class_weights_combined, moca_threshold_scaled, args.use_gpu,
        use_tabfm=not args.skip_tabfm,
        use_tabfm_ensemble=args.tabfm_ensemble,
    )

    best_results = []
    for model_name, model in models:
        X_train_model = features_for_model(
            model_name, X_train_combined, X_train_combined_raw
        )
        X_test_model = features_for_model(model_name, X_test_norm, X_test)
        m, y_train_pred, y_test_pred, train_metrics, test_metrics = train_and_evaluate_model(
            model, X_train_model, X_test_model, y_train_combined, y_test,
            use_gpu=args.use_gpu,
            use_tabfm_ensemble=args.tabfm_ensemble,
        )
        f1 = metrics.f1_score(y_test, y_test_pred)
        print(f"F1 for {model_name}: {f1:.4f}")
        
        train_f1 = metrics.f1_score(y_train_combined, y_train_pred)
        overfitting_gap = train_f1 - f1
        print(f"Overfitting gap (Train-Test F1): {overfitting_gap:.4f}")
        best_results.append((model_name, m, y_train_pred, y_test_pred, train_metrics, test_metrics))

    # Plot model comparison
    fig = plot_model_comparison(best_results, X_train_combined, X_test_norm, y_train_combined, y_test)
    fig.savefig(os.path.join(output_dir, f'model_comparison_MCI={args.mci}_NCI={args.nci}.pdf'),
                format='pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Plot individual confusion matrices
    print("\nPlotting individual confusion matrices...")
    for result_tuple in best_results:
        if len(result_tuple) == 6:
            model_name, model, y_train_pred, y_test_pred, train_metrics, test_metrics = result_tuple
        else:
            model_name, model, y_train_pred, y_test_pred, test_metrics = result_tuple[:5]
            train_metrics = None
        
        model_name_full = f'{model_name}_MCI={args.mci}_NCI={args.nci}'
        X_train_model = features_for_model(
            model_name, X_train_combined, X_train_combined_raw
        )
        X_test_model = features_for_model(model_name, X_test_norm, X_test)
        
        # Plot raw and normalized confusion matrices (recall and precision)
        plot_confusion_matrix(model, X_train_model, y_train_combined,
                             output_dir=output_dir, model_name=model_name_full,
                             test=False, y_pred=y_train_pred, normalized=False)
        plot_confusion_matrix(model, X_test_model, y_test,
                             output_dir=output_dir, model_name=model_name_full,
                             test=True, y_pred=y_test_pred, normalized=False)
        plot_confusion_matrix(model, X_train_model, y_train_combined,
                             output_dir=output_dir, model_name=model_name_full,
                             test=False, y_pred=y_train_pred, normalized=True)
        plot_confusion_matrix(model, X_test_model, y_test,
                             output_dir=output_dir, model_name=model_name_full,
                             test=True, y_pred=y_test_pred, normalized=True)
        plot_confusion_matrix(model, X_train_model, y_train_combined,
                             output_dir=output_dir, model_name=model_name_full,
                             test=False, y_pred=y_train_pred, normalized='column')
        plot_confusion_matrix(model, X_test_model, y_test,
                             output_dir=output_dir, model_name=model_name_full,
                             test=True, y_pred=y_test_pred, normalized='column')
        
        # Plot combined confusion matrices
        plot_combined_confusion_matrices(
            y_train_combined, y_test, y_train_pred, y_test_pred,
            model_name, output_dir, args.mci, args.nci
        )
        print(f"Saved confusion matrix plots for {model_name}")

    # Plot predicted probability vs. p-tau217
    print("\nPlotting predicted probability vs. p-tau217...")
    y_test_ptau217 = df.loc[X_test.index, TAU_217]
    for result_tuple in best_results:
        if len(result_tuple) == 6:
            model_name, model, _, _, _, _ = result_tuple
        else:
            model_name, model, _, _, _ = result_tuple[:5]
        plot_pred_proba_vs_ptau217(
            model,
            features_for_model(model_name, X_test_norm, X_test),
            y_test_ptau217,
            model_name,
            output_dir,
        )

    # Save model results
    save_model_results(
        best_results, best_rf_params, best_cb_params_f1, best_cb_params_ber,
        X_train_combined, y_train_combined, y_test, scaler, output_dir, args
    )

    # Save top model predictions
    print("\n" + "="*80)
    print("SAVING TOP MODEL PREDICTIONS")
    print("="*80)
    save_top_model_predictions(best_results, X_test_norm, y_test, df, output_dir, args)

    # Create and evaluate ensemble models
    print("\n" + "="*80)
    print("CREATING ENSEMBLE MODELS")
    print("="*80)
    
    ensemble_results = create_ensemble_models(
        best_results, X_test_norm, y_test, X_test_raw=X_test
    )
    plot_ensemble_comparison(ensemble_results, best_results, X_test_norm, y_test, output_dir)
    plot_ensemble_confusion_matrices(ensemble_results, y_test, output_dir, args)
    save_ensemble_results(ensemble_results, best_results, output_dir, args)
    save_ensemble_models(ensemble_results, output_dir, args)
    
    # Print ensemble summary
    print_ensemble_summary(best_results, ensemble_results)

    # Run bootstrap evaluation
    best_params = {
        'RandomForest': best_rf_params,
        'CatBoost_F1': best_cb_params_f1,
        'CatBoost_BER': best_cb_params_ber
    }
    run_bootstrap_evaluation(args, best_params, n_bootstrap=args.n_bootstrap)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train tau-217 classification models')
    parser.add_argument('--data_path', type=str,
                        default='/ems/elsc-labs/habib-n/yuval.rom/BCG/clinical_data/Data/model_preprocessed_data.csv',
                        # default='/ems/elsc-labs/habib-n/yuval.rom/BCG/clinical_data/Data/model_preprocessed_data_new_old.csv',
                        help='Path to the input data CSV file')
    parser.add_argument('--output_dir', type=str,
                        default="/ems/elsc-labs/habib-n/yuval.rom/BCG/modeling/biomarker_prediction/figures/classification",
                        help='Directory to save output plots (required)')
    parser.add_argument('--mci', action='store_true', default=False,
                        help='keep mci in the data')
    parser.add_argument('--nci', action='store_true', default=False,
                        help='keep non-cognitive inclined in the data')
    parser.add_argument('--use_gpu', action='store_true', default=True,
                        help='Use GPU for TabPFN/TabFM if available')
    parser.add_argument('--skip_tabfm', action='store_true', default=False,
                        help='Skip TabFM (useful if tabfm is not installed)')
    parser.add_argument('--tabfm_ensemble', action='store_true', default=False,
                        help='Use TabFM ensemble preset (slower, often better)')
    parser.add_argument('--n_bootstrap', type=int, default=5,
                        help='Number of bootstrap runs (default: 5)')
    args = parser.parse_args()
    main(args)
