"""
Model training and evaluation functions.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from tabpfn import TabPFNClassifier
from sklearn.utils.class_weight import compute_class_weight
from utils import evaluate_metrics


def compute_class_weights(y):
    """Compute balanced class weights.
    
    Parameters:
    -----------
    y : array-like
        Target labels
        
    Returns:
    --------
    dict
        Class weights dictionary
    """
    if isinstance(y, pd.DataFrame):
        y_flat = y.iloc[:, 0]
    else:
        y_flat = y
    
    classes = np.unique(y_flat)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_flat)
    class_weights = {int(k): v for k, v in zip(classes, weights)}
    return class_weights


def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, eval_set=None):
    """Train and evaluate a model with class balancing and early stopping.
    
    Parameters:
    -----------
    model : sklearn model
        Model to train
    X_train : array-like
        Training features
    X_test : array-like
        Test features
    y_train : array-like
        Training labels
    y_test : array-like
        Test labels
    eval_set : tuple, optional
        (X_val, y_val) for early stopping
    use_gpu : bool, optional
        Whether to use GPU when re-initializing TabFM/TabPFN after CUDA errors
    use_tabfm_ensemble : bool, optional
        Whether to use TabFM ensemble preset when re-initializing TabFM
        
    Returns:
    --------
    tuple
        (trained_model, y_train_pred, y_test_pred, train_metrics, test_metrics)
    """
    model_name = type(model).__name__
    print(f"\nTraining {model_name}...")
    
    # Flatten y_train to 1D array/Series
    if isinstance(y_train, pd.DataFrame):
        y_train_flat = y_train.iloc[:, 0]
    else:
        y_train_flat = y_train
    
    classes = np.unique(y_train_flat)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train_flat)
    class_weights = {int(k): v for k, v in zip(classes, weights)}

    # Training with early stopping for appropriate models
    if isinstance(model, RandomForestClassifier):
        model.set_params(class_weight=class_weights)
        model.fit(X_train, np.array(y_train).flatten())
    elif isinstance(model, CatBoostClassifier):
        # CatBoost supports early stopping with eval_set
        if eval_set:
            model.fit(
                X_train, np.array(y_train).flatten(),
                eval_set=eval_set,
                verbose=False
            )
        else:
            model.fit(
                X_train, np.array(y_train).flatten(),
                verbose=False
            )
    else:
        # For other models (TabPFN, etc.)
        model.fit(X_train, np.array(y_train).flatten())

    # Predictions with error handling for GPU-backed foundation models
    try:
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
    except RuntimeError as e:
        if 'CUDA' in str(e) or 'cuda' in str(e).lower():
            print(f"CUDA error during prediction: {e}")
            print(f"Re-initializing {model_name} with CPU device...")
            model = TabPFNClassifier(
                device='cpu',
                balance_probabilities=True,
                random_state=getattr(model, 'random_state', None),
            )
            model.fit(X_train, np.array(y_train).flatten())
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
        else:
            raise e
    
    # Training metrics
    print(f"\n{model_name} Training Metrics:")
    train_metrics = evaluate_metrics(model, X_train, y_train, test=False)
    print(f"Train AUC: {train_metrics[4]:.4f}" if train_metrics[4] is not None else "Train AUC: N/A")
    
    # Test metrics
    print(f"\n{model_name} Test Metrics:")
    test_metrics = evaluate_metrics(model, X_test, y_test, test=True)
    print(f"Test AUC: {test_metrics[4]:.4f}" if test_metrics[4] is not None else "Test AUC: N/A")
    
    return model, y_train_pred, y_test_pred, train_metrics, test_metrics

