import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "rmse": mse ** 0.5,
        "mae": mean_absolute_error(y_true, y_pred),
    }


def summarize_metrics(rows):
    return pd.DataFrame(rows)
