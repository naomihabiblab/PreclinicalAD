import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from utils import evaluate_metrics
from config import RAW_FEATURE_MODELS

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'


def features_for_model(model_name, X_scaled, X_raw):
    """Return the feature matrix appropriate for a given model."""
    if model_name in RAW_FEATURE_MODELS:
        return X_raw
    return X_scaled


def build_features_by_model(model_names, X_scaled, X_raw):
    """Build per-model feature dict for ensemble prediction."""
    return {
        model_name: features_for_model(model_name, X_scaled, X_raw)
        for model_name in model_names
    }


class EnsembleModel:
    """Base class for ensemble models"""
    
    def __init__(self, models, model_names):
        """
        Initialize ensemble with trained models
        
        Args:
            models: List of trained model objects
            model_names: List of model names corresponding to the models
        """
        self.models = models
        self.model_names = model_names
        self.n_models = len(models)
        
    def predict(self, X):
        """Base predict method to be implemented by subclasses"""
        raise NotImplementedError
        
    def get_model_predictions(self, X, X_by_model=None):
        """Get predictions from all individual models."""
        predictions = []
        for i, model in enumerate(self.models):
            X_model = X
            if X_by_model is not None:
                X_model = X_by_model.get(self.model_names[i], X)
            pred = model.predict(X_model)
            predictions.append(pred)
        return np.array(predictions)


class MajorityVoteEnsemble(EnsembleModel):
    """Ensemble model using simple majority voting"""
    
    def predict(self, X, X_by_model=None):
        """
        Predict using majority voting
        
        Args:
            X: Default input features (scaled)
            X_by_model: Optional dict mapping model names to feature matrices
        """
        predictions = self.get_model_predictions(X, X_by_model=X_by_model)
        # Take majority vote for each sample
        ensemble_predictions = []
        for i in range(predictions.shape[1]):  # For each sample
            sample_predictions = predictions[:, i]
            # Count votes for each class
            unique, counts = np.unique(sample_predictions, return_counts=True)
            # Select class with most votes
            majority_class = unique[np.argmax(counts)]
            ensemble_predictions.append(majority_class)
        
        return np.array(ensemble_predictions)


class WeightedMajorityVoteEnsemble(EnsembleModel):
    """Ensemble model using weighted majority voting based on F1 scores"""
    
    def __init__(self, models, model_names, f1_scores):
        """
        Initialize weighted ensemble with trained models and their F1 scores
        
        Args:
            models: List of trained model objects
            model_names: List of model names corresponding to the models
            f1_scores: List of F1 scores for each model
        """
        super().__init__(models, model_names)
        self.f1_scores = np.array(f1_scores)
        # Normalize F1 scores to sum to 1
        self.weights = self.f1_scores / np.sum(self.f1_scores)
        
    def predict(self, X, X_by_model=None):
        """
        Predict using weighted majority voting
        
        Args:
            X: Default input features (scaled)
            X_by_model: Optional dict mapping model names to feature matrices
        """
        predictions = self.get_model_predictions(X, X_by_model=X_by_model)
        ensemble_predictions = []
        
        for i in range(predictions.shape[1]):  # For each sample
            sample_predictions = predictions[:, i]
            # Calculate weighted votes for each class
            class_votes = {}
            for j, pred in enumerate(sample_predictions):
                class_label = int(pred)
                if class_label not in class_votes:
                    class_votes[class_label] = 0
                class_votes[class_label] += self.weights[j]
            
            # Select class with highest weighted votes
            best_class = max(class_votes.items(), key=lambda x: x[1])[0]
            ensemble_predictions.append(best_class)
        
        return np.array(ensemble_predictions)


def _evaluate_ensemble_metrics(ensemble, X, y, X_by_model):
    """Evaluate ensemble models that only expose predict()."""
    from sklearn import metrics as sk_metrics

    y_pred = ensemble.predict(X, X_by_model=X_by_model)
    accuracy = sk_metrics.accuracy_score(y, y_pred)
    precision = sk_metrics.precision_score(y, y_pred)
    recall = sk_metrics.recall_score(y, y_pred)
    f1 = sk_metrics.f1_score(y, y_pred)
    return accuracy, precision, recall, f1, None, None


def create_ensemble_models(best_results, X_test_norm, y_test, X_test_raw=None):
    """
    Create ensemble models from individual model results
    
    Args:
        best_results: List of tuples containing (model_name, model, y_train_pred, y_test_pred, test_metrics)
        X_test_norm: Scaled test features
        y_test: Test labels
        X_test_raw: Unscaled test features (required for models in RAW_FEATURE_MODELS)
        
    Returns:
        Dictionary containing ensemble models and their results
    """
    # Extract models and their F1 scores
    models = []
    model_names = []
    f1_scores = []
    
    for result_tuple in best_results:
        if len(result_tuple) == 6:
            model_name, model, _, _, _, test_metrics = result_tuple
        else:
            # Backward compatibility with old format
            model_name, model, _, _, test_metrics = result_tuple[:5]
        models.append(model)
        model_names.append(model_name)
        # Extract F1 score from test_metrics (index 3)
        f1_score = test_metrics[3] if len(test_metrics) > 3 else 0.0
        f1_scores.append(f1_score)
    
    # Create ensemble models
    majority_ensemble = MajorityVoteEnsemble(models, model_names)
    weighted_ensemble = WeightedMajorityVoteEnsemble(models, model_names, f1_scores)
    X_by_model = build_features_by_model(model_names, X_test_norm, X_test_raw or X_test_norm)
    
    # Evaluate ensemble models
    print("\n=== Ensemble Model Evaluation ===")
    
    # Majority Vote Ensemble
    y_pred_majority = majority_ensemble.predict(X_test_norm, X_by_model=X_by_model)
    majority_metrics = _evaluate_ensemble_metrics(
        majority_ensemble, X_test_norm, y_test, X_by_model
    )
    print(f"Majority Vote Ensemble - F1: {majority_metrics[3]:.4f}")
    
    # Weighted Majority Vote Ensemble
    y_pred_weighted = weighted_ensemble.predict(X_test_norm, X_by_model=X_by_model)
    weighted_metrics = _evaluate_ensemble_metrics(
        weighted_ensemble, X_test_norm, y_test, X_by_model
    )
    print(f"Weighted Majority Vote Ensemble - F1: {weighted_metrics[3]:.4f}")
    
    # Print individual model F1 scores and weights
    print("\nIndividual Model F1 Scores and Weights:")
    for i, (name, f1, weight) in enumerate(zip(model_names, f1_scores, weighted_ensemble.weights)):
        print(f"{name}: F1={f1:.4f}, Weight={weight:.4f}")
    
    return {
        'majority_ensemble': majority_ensemble,
        'weighted_ensemble': weighted_ensemble,
        'majority_predictions': y_pred_majority,
        'weighted_predictions': y_pred_weighted,
        'majority_metrics': majority_metrics,
        'weighted_metrics': weighted_metrics,
        'individual_f1_scores': f1_scores,
        'individual_weights': weighted_ensemble.weights
    }


def plot_ensemble_comparison(ensemble_results, best_results, X_test_norm, y_test, output_dir):
    """
    Plot comparison between ensemble models and individual models
    
    Args:
        ensemble_results: Dictionary containing ensemble model results
        best_results: List of individual model results
        X_test_norm: Test features
        y_test: Test labels
        output_dir: Output directory for saving plots
    """
    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: F1 Score Comparison
    model_names = []
    individual_f1 = []
    for result_tuple in best_results:
        if len(result_tuple) == 6:
            model_name, _, _, _, _, test_metrics = result_tuple
        else:
            model_name, _, _, _, test_metrics = result_tuple[:5]
        model_names.append(model_name)
        individual_f1.append(test_metrics[3])
    
    # Add ensemble models
    all_names = model_names + ['Majority Vote', 'Weighted Vote']
    all_f1 = individual_f1 + [ensemble_results['majority_metrics'][3], ensemble_results['weighted_metrics'][3]]
    
    bars = axes[0].bar(
        range(len(all_names)),
        all_f1,
        color=['skyblue']*len(model_names) + ['lightgreen', 'orange'],
        rasterized=False
    )
    axes[0].set_xlabel('Models')
    axes[0].set_ylabel('F1 Score')
    axes[0].set_title('F1 Score Comparison: Individual vs Ensemble Models')
    axes[0].set_xticks(range(len(all_names)))
    axes[0].set_xticklabels(all_names, rotation=45, ha='right')
    axes[0].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, all_f1):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{value:.3f}', ha='center', va='bottom')
    
    # Plot 2: Model Weights (for weighted ensemble)
    weights = ensemble_results['individual_weights']
    bars2 = axes[1].bar(range(len(model_names)), weights, color='lightcoral', rasterized=False)
    axes[1].set_xlabel('Models')
    axes[1].set_ylabel('Weight')
    axes[1].set_title('Model Weights in Weighted Ensemble')
    axes[1].set_xticks(range(len(model_names)))
    axes[1].set_xticklabels(model_names, rotation=45, ha='right')
    axes[1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars2, weights):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ensemble_model_comparison.pdf'), format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    return fig


def plot_ensemble_confusion_matrices(ensemble_results, y_test, output_dir, args):
    """
    Plot confusion matrices for ensemble models on the test set.
    """
    y_preds = {
        'Majority Vote Ensemble': ensemble_results['majority_predictions'],
        'Weighted Vote Ensemble': ensemble_results['weighted_predictions']
    }

    # Plot raw confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for i, (model_name, y_pred) in enumerate(y_preds.items()):
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], rasterized=False)
        axes[i].set_title(f'{model_name} - Test Confusion Matrix')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('True')

    plt.tight_layout()
    fig_path = os.path.join(output_dir, f'ensemble_confusion_matrices_MCI={args.mci}_NCI={args.nci}.pdf')
    fig.savefig(fig_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Ensemble confusion matrices saved to {fig_path}")
    
    # Plot normalized confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for i, (model_name, y_pred) in enumerate(y_preds.items()):
        cm = confusion_matrix(y_test, y_pred)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=axes[i], rasterized=False)
        axes[i].set_title(f'{model_name} - Test Normalized Confusion Matrix')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('True')

    plt.tight_layout()
    fig_path_norm = os.path.join(output_dir, f'ensemble_confusion_matrices_normalized_MCI={args.mci}_NCI={args.nci}.pdf')
    fig.savefig(fig_path_norm, format='pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Ensemble normalized confusion matrices saved to {fig_path_norm}")


def save_ensemble_results(ensemble_results, best_results, output_dir, args):
    """
    Save ensemble model results and comparison
    
    Args:
        ensemble_results: Dictionary containing ensemble model results
        best_results: List of individual model results
        output_dir: Output directory for saving results
        args: Command line arguments
    """
    # Save ensemble results summary
    results_path = os.path.join(output_dir, f'ensemble_results_MCI={args.mci}_NCI={args.nci}.txt')
    with open(results_path, 'w') as f:
        f.write("=== ENSEMBLE MODEL RESULTS ===\n\n")
        
        # Individual model results
        f.write("Individual Model Results:\n")
        f.write("-" * 50 + "\n")
        for i, result_tuple in enumerate(best_results):
            if len(result_tuple) == 6:
                name, _, _, _, _, test_metrics = result_tuple
            else:
                name, _, _, _, test_metrics = result_tuple[:5]
            f.write(f"{name}:\n")
            f.write(f"  F1 Score: {test_metrics[3]:.4f}\n")
            f.write(f"  Weight in Ensemble: {ensemble_results['individual_weights'][i]:.4f}\n")
            f.write(f"  Accuracy: {test_metrics[0]:.4f}\n")
            f.write(f"  Precision: {test_metrics[1]:.4f}\n")
            f.write(f"  Recall: {test_metrics[2]:.4f}\n")
            auc_str = f"{test_metrics[4]:.4f}" if test_metrics[4] is not None else "N/A"
            prauc_str = f"{test_metrics[5]:.4f}" if test_metrics[5] is not None else "N/A"
            f.write(f"  AUC: {auc_str}\n")
            f.write(f"  PRAUC: {prauc_str}\n\n")
        
        # Ensemble results
        f.write("Ensemble Model Results:\n")
        f.write("-" * 50 + "\n")
        
        f.write("Majority Vote Ensemble:\n")
        majority_metrics = ensemble_results['majority_metrics']
        f.write(f"  F1 Score: {majority_metrics[3]:.4f}\n")
        f.write(f"  Accuracy: {majority_metrics[0]:.4f}\n")
        f.write(f"  Precision: {majority_metrics[1]:.4f}\n")
        f.write(f"  Recall: {majority_metrics[2]:.4f}\n")
        if majority_metrics[4] is not None:
            f.write(f"  AUC: {majority_metrics[4]:.4f}\n")
        if majority_metrics[5] is not None:
            f.write(f"  PRAUC: {majority_metrics[5]:.4f}\n")
        f.write("\n")
        
        f.write("Weighted Majority Vote Ensemble:\n")
        weighted_metrics = ensemble_results['weighted_metrics']
        f.write(f"  F1 Score: {weighted_metrics[3]:.4f}\n")
        f.write(f"  Accuracy: {weighted_metrics[0]:.4f}\n")
        f.write(f"  Precision: {weighted_metrics[1]:.4f}\n")
        f.write(f"  Recall: {weighted_metrics[2]:.4f}\n")
        if weighted_metrics[4] is not None:
            f.write(f"  AUC: {weighted_metrics[4]:.4f}\n")
        if weighted_metrics[5] is not None:
            f.write(f"  PRAUC: {weighted_metrics[5]:.4f}\n")
        f.write("\n")
        
        # Summary statistics
        individual_f1 = []
        for result_tuple in best_results:
            if len(result_tuple) == 6:
                _, _, _, _, _, test_metrics = result_tuple
            else:
                _, _, _, _, test_metrics = result_tuple[:5]
            individual_f1.append(test_metrics[3])
        f.write("Summary Statistics:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Best Individual Model F1: {max(individual_f1):.4f}\n")
        f.write(f"Average Individual Model F1: {np.mean(individual_f1):.4f}\n")
        f.write(f"Majority Vote Ensemble F1: {majority_metrics[3]:.4f}\n")
        f.write(f"Weighted Vote Ensemble F1: {weighted_metrics[3]:.4f}\n")
        f.write(f"Ensemble Improvement (Majority): {majority_metrics[3] - np.mean(individual_f1):.4f}\n")
        f.write(f"Ensemble Improvement (Weighted): {weighted_metrics[3] - np.mean(individual_f1):.4f}\n")


def save_ensemble_models(ensemble_results, output_dir, args):
    """
    Save trained ensemble models using pickle for later use
    
    Args:
        ensemble_results: Dictionary containing ensemble model results
        output_dir: Output directory for saving models
        args: Command line arguments
    """
    import pickle
    
    # Create models directory
    models_dir = os.path.join(output_dir, 'saved_models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Save majority vote ensemble
    majority_path = os.path.join(models_dir, f'majority_vote_ensemble_MCI={args.mci}_NCI={args.nci}.pkl')
    with open(majority_path, 'wb') as f:
        pickle.dump(ensemble_results['majority_ensemble'], f)
    print(f"Majority vote ensemble saved to: {majority_path}")
    
    # Save weighted majority vote ensemble
    weighted_path = os.path.join(models_dir, f'weighted_vote_ensemble_MCI={args.mci}_NCI={args.nci}.pkl')
    with open(weighted_path, 'wb') as f:
        pickle.dump(ensemble_results['weighted_ensemble'], f)
    print(f"Weighted vote ensemble saved to: {weighted_path}")
    
    # Save ensemble clinical_data
    clinical_data_path = os.path.join(models_dir, f'ensemble_clinical_data_MCI={args.mci}_NCI={args.nci}.pkl')
    clinical_data = {
        'model_names': ensemble_results['majority_ensemble'].model_names,
        'individual_f1_scores': ensemble_results['individual_f1_scores'],
        'individual_weights': ensemble_results['individual_weights'],
        'majority_metrics': ensemble_results['majority_metrics'],
        'weighted_metrics': ensemble_results['weighted_metrics']
    }
    with open(clinical_data_path, 'wb') as f:
        pickle.dump(clinical_data, f)
    print(f"Ensemble clinical_data saved to: {clinical_data_path}")


def load_ensemble_models(output_dir, args):
    """
    Load trained ensemble models from pickle files
    
    Args:
        output_dir: Output directory containing saved models
        args: Command line arguments
        
    Returns:
        Dictionary containing loaded ensemble models and clinical_data
    """
    import pickle
    
    models_dir = os.path.join(output_dir, 'saved_models')
    
    # Load majority vote ensemble
    majority_path = os.path.join(models_dir, f'majority_vote_ensemble_MCI={args.mci}_NCI={args.nci}.pkl')
    with open(majority_path, 'rb') as f:
        majority_ensemble = pickle.load(f)
    
    # Load weighted majority vote ensemble
    weighted_path = os.path.join(models_dir, f'weighted_vote_ensemble_MCI={args.mci}_NCI={args.nci}.pkl')
    with open(weighted_path, 'rb') as f:
        weighted_ensemble = pickle.load(f)
    
    # Load clinical_data
    clinical_data_path = os.path.join(models_dir, f'ensemble_clinical_data_MCI={args.mci}_NCI={args.nci}.pkl')
    with open(clinical_data_path, 'rb') as f:
        clinical_data = pickle.load(f)
    
    return {
        'majority_ensemble': majority_ensemble,
        'weighted_ensemble': weighted_ensemble,
        'clinical_data': clinical_data
    }
