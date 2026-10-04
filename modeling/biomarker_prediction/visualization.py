"""
Visualization functions for tau-217 classification.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from scipy.stats import spearmanr
from config import HIGH_TAU_217_THRESHOLD

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'


def plot_confusion_matrix(model, X, y, output_dir=None, model_name=None, 
                         test=True, y_pred=None, normalized=False):
    """Plot confusion matrix.
    
    Parameters:
    -----------
    model : sklearn model
        Trained model
    X : array-like
        Features
    y : array-like
        True labels
    output_dir : str, optional
        Directory to save the plot
    model_name : str, optional
        Name of the model for title and filename
    test : bool, default=True
        Whether this is test data (affects title)
    y_pred : array-like, optional
        If provided, use these predictions instead of calling model.predict(X)
    normalized : bool or str, default=False
        If True or 'row', plot row-normalized confusion matrix (recall).
        If 'column', plot column-normalized confusion matrix (precision).
        
    Returns:
    --------
    str or None
        Filename if saved, None otherwise
    """
    # Use provided predictions if available, otherwise compute them
    predictions = y_pred if y_pred is not None else model.predict(X)
    cm = metrics.confusion_matrix(y, predictions)
    
    if normalized in (True, 'row'):
        cm_to_plot = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        annot_kws = {'fontsize': 10}
        metric_label = 'Recall'
        suffix = 'normalized'
    elif normalized == 'column':
        cm_to_plot = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
        fmt = '.2f'
        annot_kws = {'fontsize': 10}
        metric_label = 'Precision'
        suffix = 'normalized_column'
    else:
        cm_to_plot = cm
        fmt = 'd'
        annot_kws = {}
        metric_label = None
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_to_plot, annot=True, fmt=fmt, cmap='Blues',
                annot_kws=annot_kws, rasterized=False)
    if test:
        if metric_label:
            title_suffix = f'Test Normalized Confusion Matrix ({metric_label})'
        else:
            title_suffix = 'Test Confusion Matrix'
        plt.title(f'{model_name} - {title_suffix}' if model_name else title_suffix)
    else:
        if metric_label:
            title_suffix = f'Train Normalized Confusion Matrix ({metric_label})'
        else:
            title_suffix = 'Train Confusion Matrix'
        plt.title(f'{model_name} - {title_suffix}' if model_name else title_suffix)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    
    if output_dir and model_name:
        plt.tight_layout()
        if metric_label is None:
            suffix = 'raw'
        filename = f'{model_name}_confusion_matrix_{"test" if test else "train"}_{suffix}.pdf'
        plt.savefig(os.path.join(output_dir, filename), format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
        return filename
    else:
        plt.show()
        return None


def plot_roc_curve(y_true, y_pred_proba, model_name, output_dir):
    """Plot ROC curve.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred_proba : array-like
        Predicted probabilities for positive class
    model_name : str
        Name of the model
    output_dir : str
        Directory to save the plot
        
    Returns:
    --------
    float
        ROC AUC score
    """
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred_proba)
    roc_auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{model_name} AUC={roc_auc:.4f}', rasterized=False)
    plt.plot([0, 1], [0, 1], 'k--', rasterized=False)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(output_dir, f'{model_name}_roc_curve.pdf'), 
                format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    return roc_auc


def plot_model_comparison(models_results, X_train_norm, X_test_norm, y_train, y_test):
    """Plot comparison of models' performance with confusion matrices.
    
    Parameters:
    -----------
    models_results : list
        List of tuples (model_name, model, y_train_pred, y_test_pred, train_metrics, test_metrics)
    X_train_norm : pd.DataFrame
        Normalized training features
    X_test_norm : pd.DataFrame
        Normalized test features
    y_train : array-like
        Training labels
    y_test : array-like
        Test labels
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object with all confusion matrices
    """
    n_models = len(models_results)
    # Ensure axes is always 2D
    fig, axes = plt.subplots(2, n_models, figsize=(6*n_models, 10), squeeze=False)

    # Unpack the tuple - only need model_name, y_train_pred, and y_test_pred
    for i, result_tuple in enumerate(models_results):
        model_name, y_train_pred, y_test_pred = result_tuple[0], result_tuple[2], result_tuple[3]
        cm_train = confusion_matrix(y_train, y_train_pred)
        cm_test = confusion_matrix(y_test, y_test_pred)
        sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues',
                    ax=axes[0, i], rasterized=False)
        axes[0, i].set_title(f'{model_name} Training Confusion Matrix')
        axes[0, i].set_xlabel('Predicted')
        axes[0, i].set_ylabel('True')

        sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues',
                    ax=axes[1, i], rasterized=False)
        axes[1, i].set_title(f'{model_name} Test Confusion Matrix')
        axes[1, i].set_xlabel('Predicted')
        axes[1, i].set_ylabel('True')

    plt.tight_layout()
    return fig


def plot_pred_proba_vs_ptau217(model, X_test_norm, y_test_ptau217, model_name, output_dir):
    """Plot predicted probability vs. continuous p-tau217 values.
    
    Parameters:
    -----------
    model : sklearn model
        Trained model with predict_proba method
    X_test_norm : pd.DataFrame
        Normalized test features
    y_test_ptau217 : array-like
        Continuous p-tau217 values
    model_name : str
        Name of the model
    output_dir : str
        Directory to save the plot
    """
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test_norm)[:, 1]
        # Calculate Spearman correlation
        corr, pval = spearmanr(y_test_ptau217, y_pred_proba)
        # Linear regression for regression line and R^2
        x = np.asarray(y_test_ptau217, dtype=float)
        y = np.asarray(y_pred_proba, dtype=float)
        a, b = np.polyfit(x, y, 1)
        y_fit = a * x + b
        ss_res = np.sum((y - y_fit) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test_ptau217, y_pred_proba, alpha=0.7, rasterized=False)
        plt.xlabel('p-tau217 level (continuous)')
        plt.ylabel('Predicted probability (class 1)')
        plt.title(f'{model_name}: Predicted Probability vs. p-tau217')
        # Regression line
        order = np.argsort(x)
        plt.plot(x[order], y_fit[order], color='black', linewidth=2,
                rasterized=False,
                label=f'Regression line (R^2={r_squared:.2f})')
        plt.axvline(x=HIGH_TAU_217_THRESHOLD, color='red', linestyle='dotted',
                   rasterized=False,
                   linewidth=2, label=f'Cutoff {HIGH_TAU_217_THRESHOLD}')
        # Annotate Spearman correlation and R^2
        plt.annotate(f"Spearman r = {corr:.2f}\np = {pval:.2g}\nR^2 = {r_squared:.2f}",
                     xy=(0.05, 0.95), xycoords='axes fraction',
                     ha='left', va='top', fontsize=12,
                     bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.7))
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{model_name}_pred_proba_vs_ptau217.pdf'), 
                   format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        print(f"Model {model_name} does not support predict_proba, skipping proba vs. p-tau217 plot.")


def plot_combined_confusion_matrices(y_train, y_test, y_train_pred, y_test_pred, 
                                    model_name, output_dir, mci_flag, nci_flag):
    """Plot combined train and test confusion matrices (raw and normalized).
    
    Parameters:
    -----------
    y_train : array-like
        Training labels
    y_test : array-like
        Test labels
    y_train_pred : array-like
        Training predictions
    y_test_pred : array-like
        Test predictions
    model_name : str
        Name of the model
    output_dir : str
        Directory to save plots
    mci_flag : bool
        MCI flag for filename
    nci_flag : bool
        NCI flag for filename
    """
    # Raw confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    cm_train = confusion_matrix(y_train, y_train_pred)
    sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', ax=axes[0], rasterized=False)
    axes[0].set_title(f'{model_name} - Training')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    
    cm_test = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', ax=axes[1], rasterized=False)
    axes[1].set_title(f'{model_name} - Test')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 
                f'{model_name}_confusion_matrix_combined_MCI={mci_flag}_NCI={nci_flag}.pdf'), 
                format='pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Normalized confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
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
    fig.savefig(os.path.join(output_dir, 
                f'{model_name}_confusion_matrix_combined_normalized_MCI={mci_flag}_NCI={nci_flag}.pdf'), 
                format='pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_bootstrap_boxplots(results_df, output_dir):
    """Plot boxplots for bootstrap evaluation metrics.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame with bootstrap results
    output_dir : str
        Directory to save plots
    """
    metrics_list = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'prauc']
    for metric in metrics_list:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='model', y=metric, hue='set', data=results_df)
        plt.title(f'Bootstrap Distribution of {metric.capitalize()}')
        plt.ylabel(metric.capitalize())
        plt.xlabel('Model')
        plt.legend(title='Set')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'bootstrap_{metric}_boxplot.pdf'), 
                   format='pdf', dpi=300, bbox_inches='tight')
        plt.close()

