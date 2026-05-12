from pathlib import Path

import pandas as pd


def parse_cycle_number(path_str):
    name = Path(path_str).stem
    try:
        return int(name.split("_cycle_")[-1])
    except Exception:
        return None


def list_cycle_files(path="data/processed"):
    files = sorted(Path(path).glob("B0005_cycle_*.csv"), key=lambda p: parse_cycle_number(p))
    return [str(p) for p in files]


def load_cycle_df(path):
    return pd.read_csv(path)


def load_parameter_evolution(path="results/parameter_evolution_B0005.csv"):
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=["cycle", "R_fit", "hA_fit"])


def lookup_fit_params(cycle_num, df_params, fallback_R=0.01, fallback_hA=0.1):
    if df_params.empty or "cycle" not in df_params.columns:
        return fallback_R, fallback_hA
    idx = (df_params["cycle"] - cycle_num).abs().idxmin()
    row = df_params.loc[idx]
    return float(row.get("R_fit", fallback_R)), float(row.get("hA_fit", fallback_hA))
