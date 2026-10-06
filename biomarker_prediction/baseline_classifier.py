import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin


class BaselineClassifier(BaseEstimator, ClassifierMixin):
    """
    A baseline classifier that predicts based on MOCA score threshold.
    
    This classifier doesn't require training and simply uses the MOCA score
    to determine if a subject has MCI (score <= MCI_THR) or not.
    
    Parameters:
    -----------
    moca_col : str, default='MOCA_score'
        The column name containing MOCA scores in the input data.
    mci_thr : int, default=26
        The threshold below which a subject is classified as having MCI.
    
    Attributes:
    -----------
    moca_col_ : str
        The actual column name used for MOCA scores.
    mci_thr_ : int
        The actual threshold used for MCI classification.
    classes_ : array-like
        The classes that the classifier can predict.
    """
    
    def __init__(self, moca_col='MOCA_score', mci_thr=26, scaled_threshold=None):
        self.moca_col = moca_col
        self.mci_thr = mci_thr
        self.scaled_threshold = scaled_threshold
    
    def fit(self, X, y=None):
        """
        Fit the baseline classifier.
        
        This method does nothing as the classifier doesn't require training.
        It just validates the input and sets internal attributes.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data. Must contain the MCI score column.
        y : array-like of shape (n_samples,), optional
            Target values. Ignored for this classifier.
            
        Returns:
        --------
        self : object
            Returns self.
        """
        # Validate that MOCA column exists in the data and find its index
        if isinstance(X, pd.DataFrame):
            if self.moca_col not in X.columns:
                raise ValueError(f"MOCA column '{self.moca_col}' not found in input data")
            self.moca_col_idx_ = X.columns.get_loc(self.moca_col)
        else:
            # If X is not a DataFrame, we can't check column names
            # This is a limitation but allows the classifier to work with numpy arrays
            # Assume first column is MOCA score as fallback
            self.moca_col_idx_ = 0
        
        # Use the pre-calculated scaled threshold
        if self.scaled_threshold is not None:
            self.mci_thr_scaled_ = self.scaled_threshold
            print(f"Using pre-calculated scaled threshold: {self.mci_thr_scaled_:.4f}")
        else:
            # Fallback: if no scaled threshold provided, use the original threshold
            # This won't work correctly with scaled data but prevents errors
            self.mci_thr_scaled_ = self.mci_thr
            print(f"Warning: No scaled threshold provided, using original threshold: {self.mci_thr_scaled_}")
            print("This may not work correctly with scaled data!")
        
        # Set internal attributes
        self.moca_col_ = self.moca_col
        self.mci_thr_ = self.mci_thr
        
        # Set classes (binary classification: 0 for no MCI, 1 for MCI)
        self.classes_ = np.array([0, 1])
        
        return self
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input samples. Must contain the MCI score column.
            
        Returns:
        --------
        y : array-like of shape (n_samples,)
            The predicted class labels.
        """
        # Check if fitted
        if not hasattr(self, 'classes_'):
            raise ValueError("Classifier must be fitted before making predictions")
        
        # Extract MOCA scores using the stored column index from scaled data
        if isinstance(X, pd.DataFrame):
            moca_scores = X.iloc[:, self.moca_col_idx_].values
        else:
            # If X is not a DataFrame, use the stored column index
            moca_scores = X[:, self.moca_col_idx_]
        
        # Convert to numeric if needed
        moca_scores = pd.to_numeric(moca_scores, errors='coerce')
        
        # Predict: 1 if MOCA score <= scaled threshold (MCI), 0 otherwise
        predictions = (moca_scores <= self.mci_thr_scaled_).astype(int)
        
        return predictions
    
    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.
        
        Since this is a deterministic classifier, it returns probabilities
        of 1.0 for the predicted class and 0.0 for the other class.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
            
        Returns:
        --------
        y_proba : array-like of shape (n_samples, n_classes)
            The predicted class probabilities.
        """
        predictions = self.predict(X)
        
        # Create probability matrix
        n_samples = len(predictions)
        proba = np.zeros((n_samples, 2))
        
        # Set probability to 1.0 for predicted class, 0.0 for other class
        proba[np.arange(n_samples), predictions] = 1.0
        
        return proba
    
    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        
        Parameters:
        -----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
            
        Returns:
        --------
        params : dict
            Parameter names mapped to their values.
        """
        return {
            'moca_col': self.moca_col,
            'mci_thr': self.mci_thr
        }
    
    def set_params(self, **params):
        """
        Set the parameters of this estimator.
        
        Parameters:
        -----------
        **params : dict
            Estimator parameters.
            
        Returns:
        --------
        self : object
            Estimator instance.
        """
        for key, value in params.items():
            if key == 'moca_col':
                self.moca_col = value
            elif key == 'mci_thr':
                self.mci_thr = value
            else:
                raise ValueError(f"Invalid parameter '{key}' for BaselineClassifier")
        
        return self
