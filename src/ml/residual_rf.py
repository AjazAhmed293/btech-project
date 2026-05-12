from sklearn.ensemble import RandomForestRegressor

FEATURES = [
    "current_A",
    "voltage_V",
    "time_s",
    "cycle_number",
    "temp_phys",
    "dI",
    "dV",
    "dT_phys",
    "I2R",
]


def train_residual_rf(train_df, n_estimators=120, max_depth=12, seed=42):
    X = train_df[FEATURES]
    y = train_df["residual"]
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model


def predict_residual(model, feat_df):
    return model.predict(feat_df[FEATURES])
