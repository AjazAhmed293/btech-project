import numpy as np
import pandas as pd

from src.data.loaders import parse_cycle_number, lookup_fit_params
from src.physics.thermal_model import thermal_simulation_profile


def build_cycle_features(df, cycle, R_fit, hA_fit):
    T_amb = float(df["temp_C"].iloc[0])
    T_phys = thermal_simulation_profile(
        df["time_s"].values,
        df["current_A"].values,
        R_internal=R_fit,
        hA=hA_fit,
        T_amb=T_amb,
        T0=T_amb,
    )

    dt = np.diff(df["time_s"].values, prepend=df["time_s"].values[0])
    dI = np.diff(df["current_A"].values, prepend=df["current_A"].values[0])
    dV = np.diff(df["voltage_V"].values, prepend=df["voltage_V"].values[0])
    dT_phys = np.diff(T_phys, prepend=T_phys[0]) / np.maximum(dt, 1e-6)
    I2R = (df["current_A"].values ** 2) * R_fit

    feat = pd.DataFrame(
        {
            "current_A": df["current_A"].values,
            "voltage_V": df["voltage_V"].values,
            "time_s": df["time_s"].values,
            "cycle_number": cycle,
            "temp_phys": T_phys,
            "dI": dI,
            "dV": dV,
            "dT_phys": dT_phys,
            "I2R": I2R,
        }
    )
    feat["residual"] = df["temp_C"].values - T_phys
    return feat


def build_training_data(files, df_params, load_cycle_df_fn, max_rows=20000, seed=42):
    frames = []
    for path in files:
        df = load_cycle_df_fn(path)
        if df.empty:
            continue

        cycle = int(df["cycle_number"].iloc[0]) if "cycle_number" in df else parse_cycle_number(path)
        R_fit, hA_fit = lookup_fit_params(cycle, df_params)
        frames.append(build_cycle_features(df, cycle, R_fit, hA_fit))

    if not frames:
        return pd.DataFrame()

    all_df = pd.concat(frames, ignore_index=True)
    if len(all_df) > max_rows:
        all_df = all_df.sample(n=max_rows, random_state=seed)
    return all_df
