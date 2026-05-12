import numpy as np


def bootstrap_interval(predictions_2d, alpha=0.1):
    """Return lower, mean, upper from shape (n_models, n_samples)."""
    lower_q = alpha / 2.0
    upper_q = 1.0 - lower_q
    lower = np.quantile(predictions_2d, lower_q, axis=0)
    mean = np.mean(predictions_2d, axis=0)
    upper = np.quantile(predictions_2d, upper_q, axis=0)
    return lower, mean, upper


def interval_coverage(y_true, lower, upper):
    y_true = np.asarray(y_true)
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    inside = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(inside))


def mean_interval_width(lower, upper):
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    return float(np.mean(upper - lower))
