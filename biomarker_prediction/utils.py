from sklearn import metrics
from sklearn.metrics import roc_auc_score, average_precision_score


def evaluate_metrics(model, X, y, test=True):
    """Evaluate model metrics including AUC and PRAUC"""
    y_pred = model.predict(X)
    accuracy = metrics.accuracy_score(y, y_pred)
    precision = metrics.precision_score(y, y_pred)
    recall = metrics.recall_score(y, y_pred)
    f1 = metrics.f1_score(y, y_pred)

    # Calculate AUC and PRAUC
    try:
        y_prob = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, y_prob)
        prauc = average_precision_score(y, y_prob)
        return accuracy, precision, recall, f1, auc, prauc
    except AttributeError:
        print("Model does not support predict_proba for AUC/PRAUC calculation.")
        return accuracy, precision, recall, f1, None, None
