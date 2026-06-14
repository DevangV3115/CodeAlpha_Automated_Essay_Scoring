import logging
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score, mean_squared_error, mean_absolute_error, r2_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

def calculate_qwk(y_true, y_pred, min_rating=1, max_rating=10):
    """
    Calculates the Quadratic Weighted Kappa (QWK) score.
    Ratings are rounded to integers.
    """
    try:
        y_true_int = np.clip(np.round(y_true), min_rating, max_rating).astype(int)
        y_pred_int = np.clip(np.round(y_pred), min_rating, max_rating).astype(int)
        return cohen_kappa_score(y_true_int, y_pred_int, weights="quadratic")
    except Exception as e:
        logger.error(f"Error calculating QWK: {e}")
        return 0.0

def evaluate_predictions(y_true, y_pred):
    """
    Computes standard regression metrics and QWK/Pearson metrics.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    qwk = calculate_qwk(y_true, y_pred)
    
    try:
        corr, _ = pearsonr(y_true, y_pred)
    except Exception:
        corr = 0.0
        
    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "QWK": qwk,
        "Pearson Correlation": corr
    }

def print_evaluation_report(results_dict):
    """Prints a neat markdown table of evaluation results."""
    print("\n### Model Performance Metrics")
    print("| Metric | Value |")
    print("|---|---|")
    for metric, val in results_dict.items():
        print(f"| {metric} | {val:.4f} |")

def perform_error_analysis(y_true, y_pred, essays=None):
    """
    Analyzes prediction errors to identify where the model struggles.
    Returns a dataframe of largest errors.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    errors = np.abs(y_true - y_pred)
    
    df_err = pd.DataFrame({
        "True Score": y_true,
        "Predicted Score": y_pred,
        "Absolute Error": errors
    })
    
    if essays is not None:
        df_err["Essay Snippet"] = [str(e)[:100] + "..." for e in essays]
        
    # Get top 5 largest prediction errors
    largest_errors = df_err.sort_values(by="Absolute Error", ascending=False).head(5)
    return df_err, largest_errors

def plot_confusion_matrix_heatmap(y_true, y_pred, save_path="reports/confusion_matrix.png"):
    """Generates and saves a confusion matrix plot based on rounded integer scores."""
    y_true_int = np.clip(np.round(y_true), 1, 10).astype(int)
    y_pred_int = np.clip(np.round(y_pred), 1, 10).astype(int)
    
    labels = sorted(list(set(y_true_int) | set(y_pred_int)))
    cm = confusion_matrix(y_true_int, y_pred_int, labels=labels)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix (Rounded Scores)")
    plt.ylabel("True Rating")
    plt.xlabel("Predicted Rating")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
