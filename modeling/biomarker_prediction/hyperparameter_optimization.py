"""
Hyperparameter optimization functions using Optuna.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn import metrics
import optuna

from config import RANDOM_STATE


def objective_rf(trial, X_train, y_train, X_val, y_val):
    """Objective function for RandomForest hyperparameter optimization.
    
    Parameters:
    -----------
    trial : optuna.Trial
        Optuna trial object
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    X_val : array-like
        Validation features
    y_val : array-like
        Validation labels
        
    Returns:
    --------
    float
        F1 score on validation set
    """
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
        'max_depth': trial.suggest_int('max_depth', 1, 10),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
    }
    
    rf = RandomForestClassifier(
        random_state=42,
        bootstrap=True,
        class_weight='balanced',
        n_jobs=-1,
        **params
    )
    
    rf.fit(X_train, np.array(y_train).flatten())
    y_pred = rf.predict(X_val)
    f1 = metrics.f1_score(y_val, y_pred)
    return f1


def objective_catboost(trial, X_train, y_train, X_val, y_val, class_weights, use_gpu):
    """Objective function for CatBoost hyperparameter optimization.
    
    Parameters:
    -----------
    trial : optuna.Trial
        Optuna trial object
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    X_val : array-like
        Validation features
    y_val : array-like
        Validation labels
    class_weights : dict
        Class weights for balancing
    use_gpu : bool
        Whether to use GPU (not used in HPO, always CPU)
        
    Returns:
    --------
    float
        F1 score on validation set
    """
    params = {
        'depth': trial.suggest_int('depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'iterations': trial.suggest_int('iterations', 100, 500),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.01, 0.2),
    }
    
    cb = CatBoostClassifier(
        random_state=42,
        verbose=0,
        thread_count=-1,
        eval_metric='F1',
        class_weights=class_weights,
        allow_writing_files=False,
        early_stopping_rounds=10,
        task_type='CPU',  # Use CPU for HPO to avoid CUDA errors
        **params
    )
    
    cb.fit(
        X_train, np.array(y_train).flatten(),
        eval_set=(X_val, np.array(y_val).flatten()),
        verbose=False
    )
    
    # Store the best iteration
    if cb.best_iteration_ is not None:
        trial.set_user_attr("best_iteration", cb.best_iteration_)

    y_pred = cb.predict(X_val)
    f1 = metrics.f1_score(y_val, y_pred)
    return f1


def optimize_random_forest(X_train, y_train, X_val, y_val, n_trials=10):
    """Optimize RandomForest hyperparameters.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    X_val : array-like
        Validation features
    y_val : array-like
        Validation labels
    n_trials : int, default=10
        Number of Optuna trials
        
    Returns:
    --------
    dict
        Best hyperparameters
    """
    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(
        lambda trial: objective_rf(trial, X_train, y_train, X_val, y_val),
        n_trials=n_trials
    )
    return study.best_params, study.best_value


def optimize_catboost(X_train, y_train, X_val, y_val, class_weights, use_gpu, n_trials=10):
    """Optimize CatBoost hyperparameters.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    X_val : array-like
        Validation features
    y_val : array-like
        Validation labels
    class_weights : dict
        Class weights for balancing
    use_gpu : bool
        Whether to use GPU (not used in HPO)
    n_trials : int, default=10
        Number of Optuna trials
        
    Returns:
    --------
    dict
        Best hyperparameters
    """
    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(
        lambda trial: objective_catboost(trial, X_train, y_train, X_val, y_val, class_weights, use_gpu),
        n_trials=n_trials
    )
    return study.best_params, study.best_value

