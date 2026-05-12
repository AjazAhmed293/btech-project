import numpy as np
from sklearn.ensemble import IsolationForest

ANOMALY_FEATURES = [
    "throughput_ah",
    "discharge_ah",
    "mean_current_A",
    "std_current_A",
    "mean_voltage_V",
    "std_voltage_V",
    "mean_temp_C",
    "max_temp_C",
    "temp_rise_C",
]


def train_cycle_anomaly_detector(cycle_df, contamination=0.08, seed=42):
    X = cycle_df[ANOMALY_FEATURES].values
    model = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X)
    return model


def score_cycle_anomalies(model, cycle_df):
    X = cycle_df[ANOMALY_FEATURES].values
    # Higher score means more normal, so invert for anomaly severity.
    normality = model.decision_function(X)
    pred = model.predict(X)  # -1 anomaly, 1 normal

    out = cycle_df.copy()
    out["anomaly_label"] = np.where(pred == -1, "Anomaly", "Normal")
    out["anomaly_score"] = -normality
    return out.sort_values("anomaly_score", ascending=False).reset_index(drop=True)
