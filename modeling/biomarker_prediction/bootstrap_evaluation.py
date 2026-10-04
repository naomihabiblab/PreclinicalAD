"""
Bootstrap evaluation functions for model validation.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from tabpfn import TabPFNClassifier
import torch

from config import MOCA_COL, MCI_THR, TAU_217
from preprocessing import (
    remove_outliers, filter_data_for_classification, categorize_data,
    filter_by_cognitive_status, calculate_scaled_moca_threshold
)
from model_training import compute_class_weights
from visualization import (
    plot_pred_proba_vs_ptau217, plot_model_comparison, plot_bootstrap_boxplots
)
from utils import evaluate_metrics
from baseline_classifier import BaselineClassifier
from ensemble import features_for_model

import matplotlib.pyplot as plt

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'


def create_bootstrap_models(best_params, run_seed, class_weights, moca_threshold_scaled,
                              use_gpu, use_tabfm=True, use_tabfm_ensemble=False):
    """Create model instances for bootstrap evaluation.
    
    Parameters:
    -----------
    best_params : dict
        Best hyperparameters for each model
    run_seed : int
        Random seed for this bootstrap run
    class_weights : dict
        Class weights for balancing
    moca_threshold_scaled : float
        Scaled MOCA threshold for BaselineClassifier
    use_gpu : bool
        Whether to use GPU for TabPFN/TabFM
    use_tabfm : bool
        Whether to include TabFM
    use_tabfm_ensemble : bool
        Whether to use TabFM ensemble preset
        
    Returns:
    --------
    list
        List of (model_name, model) tuples
    """
    best_rf_params = best_params['RandomForest']
    best_cb_default_params = best_params.get('CatBoost_Default', {})
    
    models = [
        ("RandomForest", RandomForestClassifier(
            random_state=run_seed, **best_rf_params, bootstrap=True,
            class_weight='balanced', n_jobs=-1
        )),
        ("CatBoost_Default", CatBoostClassifier(
            random_state=run_seed,
            **best_cb_default_params,
            verbose=0,
            thread_count=-1,
            eval_metric='F1',
            class_weights=class_weights,
            early_stopping_rounds=15,
            allow_writing_files=False,
            task_type='CPU',
        )),
        ("LogisticRegression", LogisticRegression(
            random_state=run_seed,
            class_weight='balanced',
            max_iter=5000,
            solver='liblinear',
        )),
        ("TabPFN", TabPFNClassifier(
            device='cuda' if (use_gpu and torch.cuda.is_available()) else 'cpu',
            balance_probabilities=True,
            random_state=run_seed,
        )),
    ]
    if use_tabfm:
        from tabfm_model import create_tabfm_classifier
        models.append((
            "TabFM",
            create_tabfm_classifier(
                use_gpu=use_gpu,
                use_ensemble_preset=use_tabfm_ensemble,
                random_state=run_seed,
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


def run_bootstrap_evaluation(args, best_params, n_bootstrap=20, test_size=0.2):
    """Run bootstrap evaluation for model validation.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command line arguments
    best_params : dict
        Best hyperparameters for each model
    n_bootstrap : int, default=10
        Number of bootstrap runs
    """
    print("Loading data...")
    df = pd.read_csv(args.data_path, index_col='ID')
    print(f"Initial data shape: {df.shape}")
    df = remove_outliers(df)
    print(f"Data shape after removing outliers: {df.shape}")
    df = df[df[TAU_217].notna()]
    df = filter_by_cognitive_status(df, mci_only=args.mci, nci_only=args.nci)
    
    X_full, y_full = filter_data_for_classification(df)
    y_full = categorize_data(y_full)
    print(f"Features shape: {X_full.shape}, Target shape: {y_full.shape}")

    all_results = []
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    for run_seed in range(n_bootstrap):
        print(f"\n=== Bootstrap Run {run_seed+1}/{n_bootstrap} ===")
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X_full, y_full, test_size=test_size, random_state=run_seed, stratify=y_full
        )

        # Scale features
        scaler_full = StandardScaler()
        X_train_full_norm = scaler_full.fit_transform(X_train_full)
        X_test_full_norm = scaler_full.transform(X_test)
        X_train_full_norm = pd.DataFrame(
            X_train_full_norm, columns=X_train_full.columns, index=X_train_full.index
        )
        X_test_full_norm = pd.DataFrame(
            X_test_full_norm, columns=X_test.columns, index=X_test.index
        )

        # Calculate scaled threshold for BaselineClassifier
        moca_threshold_scaled_bootstrap = calculate_scaled_moca_threshold(X_train_full)
        class_weights = compute_class_weights(y_train_full)

        # Create models
        models = create_bootstrap_models(
            best_params, run_seed, class_weights,
            moca_threshold_scaled_bootstrap, args.use_gpu,
            use_tabfm=not getattr(args, 'skip_tabfm', False),
            use_tabfm_ensemble=getattr(args, 'tabfm_ensemble', False),
        )
        
        y_test_ptau217 = df.loc[X_test.index, TAU_217]
        
        for model_name, model in models:
            X_train_model = features_for_model(model_name, X_train_full_norm, X_train_full)
            X_test_model = features_for_model(model_name, X_test_full_norm, X_test)
            model.fit(X_train_model, np.array(y_train_full).flatten())
            
            train_metrics = evaluate_metrics(
                model, X_train_model, y_train_full, test=False
            )
            test_metrics = evaluate_metrics(
                model, X_test_model, y_test, test=True
            )

            all_results.append({
                'run': run_seed, 'model': model_name, 'set': 'train',
                'accuracy': train_metrics[0], 'precision': train_metrics[1],
                'recall': train_metrics[2], 'f1': train_metrics[3],
                'auc': train_metrics[4], 'prauc': train_metrics[5]
            })
            all_results.append({
                'run': run_seed, 'model': model_name, 'set': 'test',
                'accuracy': test_metrics[0], 'precision': test_metrics[1],
                'recall': test_metrics[2], 'f1': test_metrics[3],
                'auc': test_metrics[4], 'prauc': test_metrics[5]
            })
            
            plot_pred_proba_vs_ptau217(
                model,
                features_for_model(model_name, X_test_full_norm, X_test),
                y_test_ptau217,
                model_name,
                output_dir,
            )
    
    # Save results
    results_df = pd.DataFrame(all_results)
    summary = results_df.groupby(['model', 'set']).agg(['mean', 'std']).reset_index()
    summary.columns = ['_'.join([str(i) for i in col if i]) for col in summary.columns.values]
    excel_path = os.path.join(
        output_dir, f'bootstrap_results_MCI={args.mci}_NCI={args.nci}.xlsx'
    )
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        results_df.to_excel(writer, sheet_name='All Runs', index=False)
        summary.to_excel(writer, sheet_name='Summary', index=False)
    print(f"Results saved to {excel_path}")
    
    # Plot boxplots
    plot_bootstrap_boxplots(results_df, output_dir)

    # Plot confusion matrices for best models
    _plot_best_bootstrap_confusion_matrices(
        results_df, X_full, y_full, best_params, args, test_size
    )


def _plot_best_bootstrap_confusion_matrices(results_df, X_full, y_full, best_params, args, test_size):
    """Plot confusion matrices for best bootstrap runs.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        Bootstrap results dataframe
    X_full : pd.DataFrame
        Full feature set
    y_full : pd.DataFrame
        Full target set
    best_params : dict
        Best hyperparameters
    args : argparse.Namespace
        Command line arguments
    test_size : float
        Test set size
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    best_models = {}
    for model_name in results_df['model'].unique():
        test_f1s = results_df[
            (results_df['model'] == model_name) & (results_df['set'] == 'test')
        ]
        mean_f1 = test_f1s.groupby('run')['f1'].mean()
        best_run = mean_f1.idxmax()
        best_models[model_name] = best_run
    
    output_dir = args.output_dir
    
    for model_name, best_run in best_models.items():
        X_train, X_test, y_train, y_test = train_test_split(
            X_full, y_full, test_size=test_size, random_state=best_run, stratify=y_full
        )
        
        scaler = StandardScaler()
        X_train_norm = scaler.fit_transform(X_train)
        X_test_norm = scaler.transform(X_test)
        X_train_norm = pd.DataFrame(
            X_train_norm, columns=X_train.columns, index=X_train.index
        )
        X_test_norm = pd.DataFrame(
            X_test_norm, columns=X_test.columns, index=X_test.index
        )
        
        class_weights = compute_class_weights(y_train)
        moca_threshold_scaled = calculate_scaled_moca_threshold(X_train)
        
        # Create and train model
        if model_name == 'RandomForest':
            model = RandomForestClassifier(
                random_state=best_run, **best_params['RandomForest'],
                bootstrap=True, class_weight='balanced', n_jobs=-1
            )
        elif model_name == 'CatBoost_Default':
            model = CatBoostClassifier(
                random_state=best_run,
                **best_params.get('CatBoost_Default', {}),
                verbose=0,
                thread_count=-1,
                eval_metric='F1',
                class_weights=class_weights,
                early_stopping_rounds=15,
                allow_writing_files=False,
                task_type='CPU',
            )
        elif model_name == 'LogisticRegression':
            model = LogisticRegression(
                random_state=best_run,
                class_weight='balanced',
                max_iter=5000,
                solver='liblinear',
            )
        elif model_name == 'TabPFN':
            model = TabPFNClassifier(
                device='cuda' if (args.use_gpu and torch.cuda.is_available()) else 'cpu',
                balance_probabilities=True,
                random_state=best_run,
            )
        elif model_name == 'TabFM':
            from tabfm_model import create_tabfm_classifier
            model = create_tabfm_classifier(
                use_gpu=args.use_gpu,
                use_ensemble_preset=getattr(args, 'tabfm_ensemble', False),
                random_state=best_run,
            )
        elif model_name == 'Baseline':
            model = BaselineClassifier(
                moca_col=MOCA_COL,
                mci_thr=MCI_THR,
                scaled_threshold=moca_threshold_scaled,
            )
        else:
            continue
        
        X_train_model = features_for_model(model_name, X_train_norm, X_train)
        X_test_model = features_for_model(model_name, X_test_norm, X_test)
        model.fit(X_train_model, np.array(y_train).flatten())
        y_train_pred = model.predict(X_train_model)
        y_test_pred = model.predict(X_test_model)
        
        # Plot raw confusion matrices
        fig = plot_model_comparison(
            [(model_name, model, y_train_pred, y_test_pred, None, None)],
            X_train_norm, X_test_norm, y_train, y_test
        )
        fig_path = os.path.join(output_dir, f'confusion_matrix_{model_name}.pdf')
        fig.savefig(fig_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Confusion matrix for {model_name} saved to {fig_path}")
        
        # Plot normalized confusion matrices
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        cm_train = confusion_matrix(y_train, y_train_pred)
        cm_test = confusion_matrix(y_test, y_test_pred)
        
        cm_train_norm = cm_train.astype('float') / cm_train.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_train_norm, annot=True, fmt='.2f', cmap='Blues', ax=axes[0], rasterized=False)
        axes[0].set_title(f'{model_name} - Training (Normalized)')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('True')
        
        cm_test_norm = cm_test.astype('float') / cm_test.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_test_norm, annot=True, fmt='.2f', cmap='Blues', ax=axes[1], rasterized=False)
        axes[1].set_title(f'{model_name} - Test (Normalized)')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('True')
        
        plt.tight_layout()
        fig_path_norm = os.path.join(
            output_dir, f'confusion_matrix_normalized_{model_name}.pdf'
        )
        fig.savefig(fig_path_norm, format='pdf', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Normalized confusion matrix for {model_name} saved to {fig_path_norm}")

