import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy import stats

def create_features(data_list):
    """
    Adds engineered features and the target variable to each data dictionary.

    Args:
        data_list (list): A list of dictionaries, each with 'prediction', 'midpoint', 'final'.

    Returns:
        list: The input list with added 'success', 'growthRatio', 'midpointToPredictionRatio'.
              Returns an empty list if input is invalid.
    """
    if not isinstance(data_list, list) or not all(isinstance(item, dict) for item in data_list):
        print("Error: Input must be a list of dictionaries.")
        return []

    processed_data = []
    for row in data_list:
        if not all(key in row for key in ['prediction', 'midpoint', 'final']):
            print(f"Warning: Skipping row due to missing keys: {row}")
            continue
        try:
            # Target variable: 1 if final >= prediction, else 0
            row['success'] = 1 if row['final'] >= row['prediction'] else 0

            # Feature: Ratio of final outcome to midpoint
            row['growthRatio'] = row['final'] / row['midpoint'] if row['midpoint'] != 0 else 0

            # Feature: Ratio of prediction (target) to midpoint
            row['midpointToPredictionRatio'] = row['prediction'] / row['midpoint'] if row['midpoint'] != 0 else 0

            processed_data.append(row)
        except TypeError:
            print(f"Warning: Skipping row due to non-numeric data: {row}")
            continue
    return processed_data

def calculate_metrics(y_true, y_pred):
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        
    Returns:
        dict: Dictionary containing accuracy, precision, recall, and f1 scores
    """
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }

def calculate_metrics_summary(results):
    """
    Calculate summary statistics for a list of metric results.
    
    Args:
        results (list): List of metric dictionaries
        
    Returns:
        dict: Dictionary containing mean and std for each metric
    """
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    return {
        metric: {
            'mean': np.mean([r[metric] for r in results]),
            'std': np.std([r[metric] for r in results])
        }
        for metric in metrics
    }

def bootstrap_confidence_interval(predictions, actuals, n_bootstraps=1000, alpha=0.05):
    """
    Calculate bootstrap confidence intervals for model performance.
    
    Args:
        predictions (array-like): Model predictions
        actuals (array-like): True values
        n_bootstraps (int): Number of bootstrap samples
        alpha (float): Confidence level (e.g., 0.05 for 95% CI)
        
    Returns:
        tuple: Lower and upper bounds of the confidence interval
    """
    if len(predictions) != len(actuals):
        raise ValueError("predictions and actuals must have the same length")

    accuracies = []
    for _ in range(n_bootstraps):
        indices = np.random.randint(0, len(predictions), len(predictions))
        sample_preds = [predictions[i] for i in indices]
        sample_actuals = [actuals[i] for i in indices]
        accuracies.append(accuracy_score(sample_actuals, sample_preds))

    return np.percentile(accuracies, [100 * alpha / 2, 100 * (1 - alpha / 2)])

def analyze_performance_by_range(predictions, actuals, features, ranges=None):
    """
    Analyze model performance across different value ranges.
    
    Args:
        predictions (array-like): Model predictions
        actuals (array-like): True values
        features (array-like): Feature matrix
        ranges (list): List of tuples defining the ranges to analyze
        
    Returns:
        dict: Performance metrics for each range
    """
    if ranges is None:
        ranges = [(0, 20), (20, 25), (25, 35)]
    
    results = {}
    for low, high in ranges:
        subset_indices = [i for i, f in enumerate(features) if low <= f[0] < high]
        if subset_indices:
            subset_preds = [predictions[i] for i in subset_indices]
            subset_actuals = [actuals[i] for i in subset_indices]
            range_metrics = calculate_metrics(subset_actuals, subset_preds)
            range_metrics['sample_size'] = len(subset_indices)
            results[(low, high)] = range_metrics
    
    return results

def calculate_performance_based_weights(model_performances, temperature=1.0):
    """
    Calculate model weights based on their performance metrics.
    
    Args:
        model_performances (list): List of performance scores
        temperature (float): Temperature parameter for softmax scaling
        
    Returns:
        list: Normalized weights for each model
    """
    if not model_performances:
        return []
        
    # Apply softmax with temperature scaling
    scaled_performances = np.array(model_performances) / temperature
    exp_performances = np.exp(scaled_performances - np.max(scaled_performances))
    weights = exp_performances / exp_performances.sum()
    
    return weights.tolist()