import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_curve,
    roc_curve,
    auc
)
from scipy.stats import pearsonr, spearmanr
import logging
import warnings

# Set up logger
logger = logging.getLogger(__name__)

def calculate_metrics(y_true, y_pred, task_type='regression', average='binary'):
    """
    Calculate evaluation metrics based on task type
    
    Args:
        y_true: True labels/values
        y_pred: Predicted labels/values
        task_type: Type of task ('regression', 'binary', 'multiclass')
        average: Averaging strategy for classification metrics ('binary', 'micro', 'macro', 'weighted')
        
    Returns:
        Dictionary of metrics
    """
    # Convert inputs to numpy arrays if they are not already
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred)
    
    # Handle edge cases with NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if not np.all(mask):
        warnings.warn(f"Found {np.sum(~mask)} NaN values in true or predicted values. These will be ignored.")
        y_true = y_true[mask]
        y_pred = y_pred[mask]
    
    # Check if arrays are empty after masking
    if len(y_true) == 0 or len(y_pred) == 0:
        logger.warning("Empty arrays after removing NaN values")
        return {"error": "Empty arrays after removing NaN values"}
    
    # Ensure arrays have the same shape
    if y_true.shape != y_pred.shape:
        logger.warning(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
        return {"error": f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"}
    
    # Calculate metrics based on task type
    metrics = {}
    
    if task_type == 'regression':
        # Core regression metrics
        metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        
        # R² score
        try:
            metrics['r2'] = r2_score(y_true, y_pred)
        except:
            metrics['r2'] = 0.0
            logger.warning("Could not calculate R² score")
        
        # Correlation coefficients
        try:
            metrics['pearson_r'], metrics['pearson_p'] = pearsonr(y_true.flatten(), y_pred.flatten())
        except:
            metrics['pearson_r'] = 0.0
            metrics['pearson_p'] = 1.0
            logger.warning("Could not calculate Pearson correlation")
        
        try:
            metrics['spearman_r'], metrics['spearman_p'] = spearmanr(y_true.flatten(), y_pred.flatten())
        except:
            metrics['spearman_r'] = 0.0
            metrics['spearman_p'] = 1.0
            logger.warning("Could not calculate Spearman correlation")
        
        # Mean and standard deviations of residuals
        residuals = y_true - y_pred
        metrics['residual_mean'] = np.mean(residuals)
        metrics['residual_std'] = np.std(residuals)
        
        # Median absolute error
        metrics['median_ae'] = np.median(np.abs(residuals))
        
        # Coefficient of variation of RMSE
        if np.mean(y_true) != 0:
            metrics['cv_rmse'] = metrics['rmse'] / np.abs(np.mean(y_true))
        else:
            metrics['cv_rmse'] = np.inf
    
    elif task_type == 'binary':
        # For binary classification, we threshold predictions if they're not already binary
        if not np.all(np.isin(y_pred, [0, 1])):
            y_pred_binary = (y_pred > 0.5).astype(int)
        else:
            y_pred_binary = y_pred
        
        # Core classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred_binary)
        metrics['precision'] = precision_score(y_true, y_pred_binary, average=average, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred_binary, average=average, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred_binary, average=average, zero_division=0)
        
        # Matthews correlation coefficient
        metrics['mcc'] = matthews_corrcoef(y_true, y_pred_binary)
        
        # ROC AUC and PR AUC if we have probability predictions
        if not np.all(np.isin(y_pred, [0, 1])):
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred)
                metrics['pr_auc'] = average_precision_score(y_true, y_pred)
                
                # Calculate ROC curve data for plotting
                fpr, tpr, thresholds = roc_curve(y_true, y_pred)
                metrics['roc_curve'] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'thresholds': thresholds.tolist()}
                
                # Calculate PR curve data for plotting
                precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
                metrics['pr_curve'] = {'precision': precision.tolist(), 'recall': recall.tolist(), 'thresholds': thresholds.tolist()}
            except:
                logger.warning("Could not calculate AUC metrics")
        
        # Confusion matrix
        try:
            cm = confusion_matrix(y_true, y_pred_binary)
            metrics['confusion_matrix'] = cm.tolist()
            
            # Calculate additional metrics from confusion matrix
            tn, fp, fn, tp = cm.ravel()
            metrics['tn'] = int(tn)
            metrics['fp'] = int(fp)
            metrics['fn'] = int(fn)
            metrics['tp'] = int(tp)
            
            # Calculate specificity (true negative rate)
            if tn + fp > 0:
                metrics['specificity'] = tn / (tn + fp)
            else:
                metrics['specificity'] = 0.0
            
            # Calculate balanced accuracy
            metrics['balanced_accuracy'] = (metrics['recall'] + metrics['specificity']) / 2
        except:
            logger.warning("Could not calculate confusion matrix")
    
    elif task_type == 'multiclass':
        # For multiclass classification, we take the argmax of predictions if they're probabilities
        if y_pred.ndim > 1 and y_pred.shape[1] > 1:
            y_pred_classes = np.argmax(y_pred, axis=1)
        else:
            y_pred_classes = y_pred
        
        # Core classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred_classes)
        
        # Class-wise and average metrics
        for avg in ['micro', 'macro', 'weighted']:
            metrics[f'precision_{avg}'] = precision_score(y_true, y_pred_classes, average=avg, zero_division=0)
            metrics[f'recall_{avg}'] = recall_score(y_true, y_pred_classes, average=avg, zero_division=0)
            metrics[f'f1_{avg}'] = f1_score(y_true, y_pred_classes, average=avg, zero_division=0)
        
        # Matthews correlation coefficient
        metrics['mcc'] = matthews_corrcoef(y_true, y_pred_classes)
        
        # Multi-class ROC AUC if we have probability predictions
        if y_pred.ndim > 1 and y_pred.shape[1] > 1:
            try:
                metrics['roc_auc_micro'] = roc_auc_score(y_true, y_pred, average='micro', multi_class='ovr')
                metrics['roc_auc_macro'] = roc_auc_score(y_true, y_pred, average='macro', multi_class='ovr')
                metrics['roc_auc_weighted'] = roc_auc_score(y_true, y_pred, average='weighted', multi_class='ovr')
            except:
                logger.warning("Could not calculate AUC metrics")
        
        # Confusion matrix
        try:
            cm = confusion_matrix(y_true, y_pred_classes)
            metrics['confusion_matrix'] = cm.tolist()
        except:
            logger.warning("Could not calculate confusion matrix")
    
    else:
        raise ValueError(f"Unknown task type: {task_type}")
    
    return metrics


def print_metrics(metrics, task_type='regression'):
    """
    Print metrics in a formatted way
    
    Args:
        metrics: Dictionary of metrics
        task_type: Type of task ('regression', 'binary', 'multiclass')
    """
    if 'error' in metrics:
        logger.error(f"Error in metrics calculation: {metrics['error']}")
        return
    
    if task_type == 'regression':
        print("Regression Metrics:")
        print(f"  RMSE: {metrics.get('rmse', 'N/A'):.4f}")
        print(f"  MAE: {metrics.get('mae', 'N/A'):.4f}")
        print(f"  R²: {metrics.get('r2', 'N/A'):.4f}")
        print(f"  Pearson r: {metrics.get('pearson_r', 'N/A'):.4f} (p={metrics.get('pearson_p', 'N/A'):.4f})")
        print(f"  Spearman r: {metrics.get('spearman_r', 'N/A'):.4f} (p={metrics.get('spearman_p', 'N/A'):.4f})")
        print(f"  Residual Mean: {metrics.get('residual_mean', 'N/A'):.4f}")
        print(f"  Residual Std: {metrics.get('residual_std', 'N/A'):.4f}")
    
    elif task_type == 'binary':
        print("Binary Classification Metrics:")
        print(f"  Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
        print(f"  Precision: {metrics.get('precision', 'N/A'):.4f}")
        print(f"  Recall: {metrics.get('recall', 'N/A'):.4f}")
        print(f"  F1 Score: {metrics.get('f1', 'N/A'):.4f}")
        print(f"  MCC: {metrics.get('mcc', 'N/A'):.4f}")
        if 'roc_auc' in metrics:
            print(f"  ROC AUC: {metrics['roc_auc']:.4f}")
        if 'pr_auc' in metrics:
            print(f"  PR AUC: {metrics['pr_auc']:.4f}")
        if 'confusion_matrix' in metrics:
            print(f"  Confusion Matrix: {metrics['confusion_matrix']}")
    
    elif task_type == 'multiclass':
        print("Multiclass Classification Metrics:")
        print(f"  Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
        print(f"  Macro Precision: {metrics.get('precision_macro', 'N/A'):.4f}")
        print(f"  Macro Recall: {metrics.get('recall_macro', 'N/A'):.4f}")
        print(f"  Macro F1: {metrics.get('f1_macro', 'N/A'):.4f}")
        print(f"  MCC: {metrics.get('mcc', 'N/A'):.4f}")
        if 'roc_auc_macro' in metrics:
            print(f"  Macro ROC AUC: {metrics['roc_auc_macro']:.4f}")
    
    else:
        raise ValueError(f"Unknown task type: {task_type}")


def calculate_modal_importance(model_outputs, targets, modality_names=None):
    """
    Calculate the importance of each modality based on its predictive performance
    
    Args:
        model_outputs: Dictionary of outputs from each modality and combined model
        targets: True target values
        modality_names: List of modality names
        
    Returns:
        Dictionary of modality importance metrics
    """
    if modality_names is None:
        modality_names = list(model_outputs.keys())
    
    # Calculate metrics for each modality
    modality_metrics = {}
    for modality in modality_names:
        if modality in model_outputs:
            modality_metrics[modality] = calculate_metrics(targets, model_outputs[modality])
    
    # Calculate relative importance based on R² or accuracy
    importance = {}
    baseline_metric = None
    
    # Check if we have regression or classification metrics
    is_regression = 'r2' in next(iter(modality_metrics.values()))
    
    if is_regression:
        # For regression, use R² as the metric
        metric_name = 'r2'
        # Higher is better for R²
        higher_better = True
    else:
        # For classification, use accuracy
        metric_name = 'accuracy'
        # Higher is better for accuracy
        higher_better = True
    
    # Get baseline metric from 'combined' if available, else use best modality
    if 'combined' in modality_metrics:
        baseline_metric = modality_metrics['combined'][metric_name]
    else:
        if higher_better:
            baseline_metric = max([metrics[metric_name] for metrics in modality_metrics.values()])
        else:
            baseline_metric = min([metrics[metric_name] for metrics in modality_metrics.values()])
    
    # Calculate raw importance
    raw_importance = {}
    for modality, metrics in modality_metrics.items():
        if modality != 'combined':
            if higher_better:
                # For metrics where higher is better (R², accuracy)
                raw_importance[modality] = metrics[metric_name] / baseline_metric if baseline_metric != 0 else 0
            else:
                # For metrics where lower is better (RMSE, MAE)
                raw_importance[modality] = baseline_metric / metrics[metric_name] if metrics[metric_name] != 0 else 0
    
    # Normalize to sum to 1
    total_importance = sum(raw_importance.values())
    normalized_importance = {
        modality: value / total_importance for modality, value in raw_importance.items()
    } if total_importance > 0 else raw_importance
    
    return {
        'raw_importance': raw_importance,
        'normalized_importance': normalized_importance,
        'metrics': modality_metrics
    }


def bootstrap_confidence_interval(y_true, y_pred, metric_func, n_bootstraps=1000, confidence=0.95):
    """
    Calculate bootstrap confidence intervals for a given metric
    
    Args:
        y_true: True values
        y_pred: Predicted values
        metric_func: Function to calculate the metric
        n_bootstraps: Number of bootstrap samples
        confidence: Confidence level
        
    Returns:
        Dictionary with the metric value, lower and upper bounds
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate the metric on the full dataset
    metric_value = metric_func(y_true, y_pred)
    
    # Generate bootstrap samples
    bootstrap_metrics = []
    indices = np.arange(len(y_true))
    
    for _ in range(n_bootstraps):
        # Sample with replacement
        bootstrap_indices = np.random.choice(indices, size=len(indices), replace=True)
        bootstrap_true = y_true[bootstrap_indices]
        bootstrap_pred = y_pred[bootstrap_indices]
        
        # Calculate metric on bootstrap sample
        bootstrap_metric = metric_func(bootstrap_true, bootstrap_pred)
        bootstrap_metrics.append(bootstrap_metric)
    
    # Calculate confidence interval
    alpha = (1 - confidence) / 2
    lower_bound = np.percentile(bootstrap_metrics, alpha * 100)
    upper_bound = np.percentile(bootstrap_metrics, (1 - alpha) * 100)
    
    return {
        'value': metric_value,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'bootstrap_samples': bootstrap_metrics
    }


def calibration_metrics(y_true, y_prob, n_bins=10):
    """
    Calculate calibration metrics for binary classification
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_bins: Number of bins for calibration curve
        
    Returns:
        Dictionary with calibration metrics
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    # Create bins
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    # Calculate calibration metrics
    bin_counts = np.bincount(bin_indices, minlength=n_bins)
    bin_sums = np.bincount(bin_indices, weights=y_true, minlength=n_bins)
    bin_means = np.zeros(n_bins)
    bin_true_probs = np.zeros(n_bins)
    
    for i in range(n_bins):
        if bin_counts[i] > 0:
            bin_means[i] = np.mean(y_prob[bin_indices == i])
            bin_true_probs[i] = bin_sums[i] / bin_counts[i]
        else:
            bin_means[i] = (bin_edges[i] + bin_edges[i + 1]) / 2
            bin_true_probs[i] = 0
    
    # Calculate expected calibration error (ECE)
    ece = np.sum(bin_counts * np.abs(bin_means - bin_true_probs)) / np.sum(bin_counts)
    
    # Calculate maximum calibration error (MCE)
    mce = np.max(np.abs(bin_means - bin_true_probs))
    
    # Calculate Brier score
    brier_score = np.mean((y_prob - y_true) ** 2)
    
    return {
        'ece': ece,
        'mce': mce,
        'brier_score': brier_score,
        'calibration_curve': {
            'pred_probs': bin_means.tolist(),
            'true_probs': bin_true_probs.tolist(),
            'bin_counts': bin_counts.tolist(),
            'bin_edges': bin_edges.tolist()
        }
    }


def modal_correlation_analysis(modal_embeddings, modality_names=None):
    """
    Analyze correlations between different modality embeddings
    
    Args:
        modal_embeddings: Dictionary of embeddings from different modalities
        modality_names: List of modality names
        
    Returns:
        Dictionary with correlation metrics
    """
    if modality_names is None:
        modality_names = list(modal_embeddings.keys())
    
    # Calculate pairwise correlations
    correlation_matrix = np.zeros((len(modality_names), len(modality_names)))
    cosine_matrix = np.zeros((len(modality_names), len(modality_names)))
    
    for i, mod_i in enumerate(modality_names):
        for j, mod_j in enumerate(modality_names):
            if i == j:
                correlation_matrix[i, j] = 1.0
                cosine_matrix[i, j] = 1.0
            else:
                # Get embeddings
                emb_i = modal_embeddings[mod_i]
                emb_j = modal_embeddings[mod_j]
                
                # Calculate correlation
                correlation_matrix[i, j] = np.mean([
                    pearsonr(emb_i[k], emb_j[k])[0] 
                    for k in range(min(len(emb_i), len(emb_j)))
                    if not (np.isnan(emb_i[k]).any() or np.isnan(emb_j[k]).any())
                ])
                
                # Calculate cosine similarity
                cosine_matrix[i, j] = np.mean([
                    np.dot(emb_i[k], emb_j[k]) / (np.linalg.norm(emb_i[k]) * np.linalg.norm(emb_j[k]))
                    for k in range(min(len(emb_i), len(emb_j)))
                    if not (np.isnan(emb_i[k]).any() or np.isnan(emb_j[k]).any())
                ])
    
    return {
        'correlation_matrix': correlation_matrix.tolist(),
        'cosine_matrix': cosine_matrix.tolist(),
        'modality_names': modality_names
    }