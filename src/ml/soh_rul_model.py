import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.data.loaders import parse_cycle_number
from src.ml.uncertainty import bootstrap_interval, interval_coverage, mean_interval_width

SOH_RUL_FEATURES = [
    "cycle_number",
    "duration_s",
    "throughput_ah",
    "mean_current_A",
    "std_current_A",
    "mean_voltage_V",
    "std_voltage_V",
    "mean_temp_C",
    "max_temp_C",
    "temp_rise_C",
]


def build_cycle_summary_df(files, load_cycle_df_fn):
    rows = []
    for path in files:
        df = load_cycle_df_fn(path)
        if df.empty:
            continue
        cycle = int(df["cycle_number"].iloc[0]) if "cycle_number" in df.columns else parse_cycle_number(path)
        dt = np.diff(df["time_s"].values, prepend=df["time_s"].values[0])
        if len(dt) > 1:
            dt[0] = np.median(dt[1:])
        current = df["current_A"].values
        discharge_mask = current < 0
        if np.any(discharge_mask):
            discharge_ah = float(np.sum(np.abs(current[discharge_mask]) * dt[discharge_mask]) / 3600.0)
        else:
            discharge_ah = float(np.sum(np.abs(current) * dt) / 3600.0)
        throughput_ah = float(np.sum(np.abs(current) * dt) / 3600.0)
        rows.append(
            {
                "cycle_number": cycle,
                "duration_s": float(df["time_s"].iloc[-1] - df["time_s"].iloc[0]),
                "throughput_ah": throughput_ah,
                "discharge_ah": discharge_ah,
                "mean_current_A": float(df["current_A"].mean()),
                "std_current_A": float(df["current_A"].std(ddof=0)),
                "mean_voltage_V": float(df["voltage_V"].mean()),
                "std_voltage_V": float(df["voltage_V"].std(ddof=0)),
                "mean_temp_C": float(df["temp_C"].mean()),
                "max_temp_C": float(df["temp_C"].max()),
                "temp_rise_C": float(df["temp_C"].max() - df["temp_C"].iloc[0]),
            }
        )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values("cycle_number").reset_index(drop=True)

    # Realistic SOH proxy from discharge capacity relative to early-life nominal capacity.
    ref_window = min(10, len(out))
    nominal_capacity_ah = float(np.median(out["discharge_ah"].head(ref_window)))
    nominal_capacity_ah = max(nominal_capacity_ah, 1e-9)
    out["soh_target"] = (out["discharge_ah"] / nominal_capacity_ah).clip(0.3, 1.1)
    out["nominal_capacity_ah"] = nominal_capacity_ah
    return out


def attach_rul_target(cycle_df, soh_threshold=0.8):
    out = cycle_df.copy()
    failed = out[out["soh_target"] <= soh_threshold]
    fail_cycle = int(failed["cycle_number"].iloc[0]) if not failed.empty else int(out["cycle_number"].max())
    out["rul_target"] = (fail_cycle - out["cycle_number"]).clip(lower=0)
    return out


def time_order_split(cycle_df, val_ratio=0.2, test_ratio=0.2):
    n = len(cycle_df)
    if n < 10:
        return cycle_df.copy(), pd.DataFrame(), pd.DataFrame()
    n_test = max(1, int(n * test_ratio))
    n_val = max(1, int(n * val_ratio))
    n_train = max(1, n - n_val - n_test)
    if n_train + n_val + n_test > n:
        n_val = max(1, n - n_train - n_test)
    train_df = cycle_df.iloc[:n_train].copy()
    val_df = cycle_df.iloc[n_train:n_train + n_val].copy()
    test_df = cycle_df.iloc[n_train + n_val:].copy()
    return train_df, val_df, test_df


def train_soh_rul_ensemble(cycle_df, n_models=15, seed=42):
    X = cycle_df[SOH_RUL_FEATURES].values
    y_soh = cycle_df["soh_target"].values
    y_rul = cycle_df["rul_target"].values

    rng = np.random.RandomState(seed)
    soh_models = []
    rul_models = []
    n = len(cycle_df)
    for i in range(n_models):
        idx = rng.randint(0, n, size=n)
        Xb = X[idx]
        yb_soh = y_soh[idx]
        yb_rul = y_rul[idx]
        soh = RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            random_state=seed + i,
            n_jobs=-1,
        )
        rul = RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            random_state=seed + 100 + i,
            n_jobs=-1,
        )
        soh.fit(Xb, yb_soh)
        rul.fit(Xb, yb_rul)
        soh_models.append(soh)
        rul_models.append(rul)
    return {"soh_models": soh_models, "rul_models": rul_models}


def predict_soh_rul_with_uncertainty(ensemble, cycle_df, alpha=0.1):
    X = cycle_df[SOH_RUL_FEATURES].values
    soh_preds = np.vstack([m.predict(X) for m in ensemble["soh_models"]])
    rul_preds = np.vstack([m.predict(X) for m in ensemble["rul_models"]])

    soh_lo, soh_mean, soh_hi = bootstrap_interval(soh_preds, alpha=alpha)
    rul_lo, rul_mean, rul_hi = bootstrap_interval(rul_preds, alpha=alpha)

    out = cycle_df.copy()
    out["soh_pred"] = soh_mean
    out["soh_lo"] = soh_lo
    out["soh_hi"] = soh_hi
    out["rul_pred"] = rul_mean
    out["rul_lo"] = rul_lo
    out["rul_hi"] = rul_hi
    return out


def evaluate_soh_rul_predictions(pred_df):
    soh_err = pred_df["soh_pred"].values - pred_df["soh_target"].values
    rul_err = pred_df["rul_pred"].values - pred_df["rul_target"].values
    soh_rmse = float(np.sqrt(np.mean(soh_err ** 2)))
    soh_mae = float(np.mean(np.abs(soh_err)))
    rul_rmse = float(np.sqrt(np.mean(rul_err ** 2)))
    rul_mae = float(np.mean(np.abs(rul_err)))
    return {
        "soh_rmse": soh_rmse,
        "soh_mae": soh_mae,
        "rul_rmse": rul_rmse,
        "rul_mae": rul_mae,
        "soh_interval_coverage": interval_coverage(
            pred_df["soh_target"].values, pred_df["soh_lo"].values, pred_df["soh_hi"].values
        ),
        "soh_interval_width": mean_interval_width(pred_df["soh_lo"].values, pred_df["soh_hi"].values),
        "rul_interval_coverage": interval_coverage(
            pred_df["rul_target"].values, pred_df["rul_lo"].values, pred_df["rul_hi"].values
        ),
        "rul_interval_width": mean_interval_width(pred_df["rul_lo"].values, pred_df["rul_hi"].values),
    }
