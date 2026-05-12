from src.data.features import build_cycle_features
from src.data.loaders import load_cycle_df, lookup_fit_params
from src.evaluation.metrics import metrics
from src.ml.residual_rf import predict_residual
from src.physics.thermal_model import thermal_simulation_profile


def evaluate_cycle(model, cycle_path, cycle_num, df_params):
    df = load_cycle_df(cycle_path)
    R_fit, hA_fit = lookup_fit_params(cycle_num, df_params)
    T_amb = float(df["temp_C"].iloc[0])

    T_phys = thermal_simulation_profile(
        df["time_s"].values,
        df["current_A"].values,
        R_internal=R_fit,
        hA=hA_fit,
        T_amb=T_amb,
        T0=T_amb,
    )

    feat_eval = build_cycle_features(df, cycle_num, R_fit, hA_fit)
    residual_pred = predict_residual(model, feat_eval)
    T_hybrid = T_phys + residual_pred

    return {
        "cycle": cycle_num,
        "physics": metrics(df["temp_C"].values, T_phys),
        "hybrid": metrics(df["temp_C"].values, T_hybrid),
    }
