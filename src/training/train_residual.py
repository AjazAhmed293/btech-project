from src.data.features import build_training_data
from src.data.loaders import load_cycle_df, load_parameter_evolution
from src.ml.residual_rf import train_residual_rf


def train_residual_pipeline(train_files, max_rows=20000, seed=42):
    df_params = load_parameter_evolution()
    train_df = build_training_data(train_files, df_params, load_cycle_df, max_rows=max_rows, seed=seed)
    if train_df.empty:
        raise RuntimeError("No training rows were created from train_files.")
    model = train_residual_rf(train_df, seed=seed)
    return model, train_df
