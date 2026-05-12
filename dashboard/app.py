import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path

# Add root to path so our src/ modules are importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.features import build_cycle_features, build_training_data
from src.data.loaders import (
    list_cycle_files,
    load_parameter_evolution,
    lookup_fit_params,
    load_cycle_df,
    parse_cycle_number,
)
from src.data.splits import split_cycles
from src.evaluation.metrics import metrics
from src.ml.anomaly import score_cycle_anomalies, train_cycle_anomaly_detector
from src.ml.residual_rf import train_residual_rf, predict_residual
from src.ml.soh_rul_model import (
    attach_rul_target,
    build_cycle_summary_df,
    evaluate_soh_rul_predictions,
    predict_soh_rul_with_uncertainty,
    time_order_split,
    train_soh_rul_ensemble,
)
from src.physics.degradation_model import (
    degradation_model,
    capacity_fade_model,
    estimate_rul_cycles,
)
from src.physics.thermal_model import thermal_simulation, thermal_simulation_profile


# Helper functions

# Simple rule-based alert for the driver. No ML here,
# just checking thresholds in order of severity.
def get_driver_feedback(max_temp, range_km, speed):
    if max_temp > 60:
        return "DANGER: Battery overheating!", "Reduce speed and enable cooling immediately."
    if max_temp > 45:
        return "Caution: Battery heating up.", "Drive slower or avoid aggressive acceleration."
    if range_km < 50:
        return "Low range: less than 50 km left.", "Find the nearest charging station."
    if speed < 50:
        return "Efficient driving.", "Maintain your current speed."
    return "Safe operation.", "Drive steady."


def calculate_current_draw(driving_style, road_type, speed):
    """
    Returns how much current (Amps) the battery draws.
    Values are rough estimates based on typical EV specs.
    Speed scales the base value up or down.
    """
    # Base current lookup — rows = style, cols = road
    base_map = {
        "Normal":     {"City": 50,  "Highway": 100, "Uphill": 120},
        "Eco":        {"City": 30,  "Highway": 50,  "Uphill": 60},
        "Aggressive": {"City": 80,  "Highway": 150, "Uphill": 180},
    }
    base = base_map[driving_style][road_type]
    return base * (speed / 100)


def estimate_remaining_range(soc, consumed_ah, pack_cap_ah, pack_energy_kwh):
    """
    Figures out how much range is left in km.
    We subtract what was consumed from the starting SOC,
    then convert remaining energy to km.
    """
    pct_used = (consumed_ah / pack_cap_ah) * 100
    soc_left = max(0, soc - pct_used)
    energy_left = (soc_left / 100) * pack_energy_kwh
    range_left_km = energy_left / 0.18  # 0.18 kWh/km is standard EV efficiency
    return soc_left, range_left_km


# Page config

st.set_page_config(page_title="EV Battery Digital Twin", layout="wide")
st.caption("Sem 8 modularized app")
st.title("EV Battery Digital Twin")


# Driving profiles — presets so you don't have to set
# every slider from scratch during a demo

driving_profiles = {
    "City Commute":   {"T_amb": 25, "speed": 40,  "style": "Eco",        "road": "City",    "soc": 80},
    "Mountain Drive": {"T_amb": 20, "speed": 60,  "style": "Aggressive", "road": "Uphill",  "soc": 70},
    "Highway Trip":   {"T_amb": 30, "speed": 100, "style": "Normal",     "road": "Highway", "soc": 90},
}

selected_profile = st.sidebar.selectbox("Driving profile", list(driving_profiles.keys()))
preset = driving_profiles[selected_profile]


# Sidebar inputs
# Driving conditions
amb_temp   = st.sidebar.slider("Ambient temp (C)", -20, 60,  preset["T_amb"],  5)
speed      = st.sidebar.slider("Speed (km/h)",      20, 160, preset["speed"],  10)
road_type  = st.sidebar.selectbox("Road type", ["City", "Highway", "Uphill"],
                 index=["City", "Highway", "Uphill"].index(preset["road"]))
drv_style  = st.sidebar.selectbox("Driving style", ["Eco", "Normal", "Aggressive"],
                 index=["Eco", "Normal", "Aggressive"].index(preset["style"]))
init_soc   = st.sidebar.slider("SOC (%)", 10, 100, preset["soc"], 5)

# Degradation params — using physics notation (kR, R0 etc.) to match our report
cycle_num = st.sidebar.slider("Cycle number", 0, 1000, 50, 10)
kR        = st.sidebar.number_input("Resistance growth kR", min_value=1e-5, max_value=5e-3, value=5e-4, step=1e-5, format="%.5f")
R0        = st.sidebar.number_input("Baseline R0 (Ohm)",    min_value=0.005, max_value=0.2,  value=0.01,  step=0.001, format="%.3f")
Q0        = st.sidebar.number_input("Baseline Q0 (Ah)",     min_value=1.0,   max_value=5.0,  value=2.0,   step=0.1,   format="%.1f")
kQ        = st.sidebar.number_input("Capacity fade kQ",     min_value=1e-5, max_value=5e-3, value=1e-4, step=1e-5, format="%.5f")
R_limit   = st.sidebar.number_input("RUL threshold (Ohm)",  min_value=0.02,  max_value=0.3,  value=0.08,  step=0.005, format="%.3f")

# Simulation duration (slider gives minutes, we convert to seconds right away)
sim_duration = st.sidebar.slider("Simulation duration (min)", 10, 60, 30, 5) * 60
use_hybrid   = st.sidebar.checkbox("Enable hybrid twin", True)

# ML settings
max_train_rows  = st.sidebar.slider("ML training rows",   5000,  40000, 20000, 5000)
max_train_files = st.sidebar.slider("Training cycles",    10,    120,   60,    10)
val_ratio       = st.sidebar.slider("Validation split",   0.1,   0.4,   0.2,   0.05)
soh_test_ratio  = st.sidebar.slider("SOH/RUL test split", 0.1,   0.4,   0.2,   0.05)
soh_val_ratio   = st.sidebar.slider("SOH/RUL val split",  0.1,   0.4,   0.2,   0.05)
soh_threshold   = st.sidebar.slider("SOH failure threshold", 0.70, 0.95, 0.80, 0.01)

# Uncertainty + anomaly
unc_alpha      = st.sidebar.slider("Uncertainty alpha",     0.05, 0.30, 0.10, 0.05)
n_ensemble     = st.sidebar.slider("Ensemble models",       5,    30,   15,   5)
contamination  = st.sidebar.slider("Anomaly contamination", 0.02, 0.20, 0.08, 0.01)
export_results = st.sidebar.checkbox("Export SOH/RUL artifacts", False)


# Physics calculations
# These recalculate every time a slider changes.
# Battery degradation at this cycle number
R_internal  = degradation_model(cycle_num, R0=R0, kR=kR)
cell_cap_ah = capacity_fade_model(cycle_num, Q0=Q0, kQ=kQ)
rul_cycles  = estimate_rul_cycles(cycle_num, R0=R0, kR=kR, R_limit=R_limit)

# hA = heat transfer coefficient (cooling rate).
# It goes up with speed because faster driving = more airflow over the pack.
hA = 0.05 + (speed / 160) * 0.15

curr_draw = calculate_current_draw(drv_style, road_type, speed)

# Run the thermal model — gives us temperature at each timestep
time_arr, temp_arr, heat_loss_kj, _ = thermal_simulation(
    curr_draw,
    sim_duration,
    T_amb=amb_temp,
    hA=hA,
    R_internal=R_internal,
)
PACK_CAP_AH = 20000
V_NOM       = 3.7  # nominal lithium-ion cell voltage
pack_energy = (PACK_CAP_AH * V_NOM) / 1000  # kWh
consumed_ah = curr_draw * sim_duration / 3600
soc_left, range_left_km = estimate_remaining_range(init_soc, consumed_ah, PACK_CAP_AH, pack_energy)

# Driver alert message
status_msg, action_msg = get_driver_feedback(float(np.max(temp_arr)), range_left_km, speed)

# Top metrics strip

c1, c2, c3, c4 = st.columns(4)
c1.metric("Max Temp",   f"{temp_arr.max():.1f} C")
c2.metric("SOC Left",   f"{soc_left:.1f}%")
c3.metric("Internal R", f"{R_internal:.4f} Ohm")
c4.metric("RUL",        f"{rul_cycles} cycles")

st.info(f"{status_msg}  {action_msg}")


# Tabs
tab_thermal, tab_hybrid, tab_soh_rul, tab_safety, tab_data = st.tabs(
    ["Thermal", "Hybrid Validation", "SOH/RUL + Uncertainty", "Safety/Anomaly", "Data"]
)


# TAB 1: Thermal plot
with tab_thermal:
    fig, ax = plt.subplots()
    ax.plot(time_arr / 60, temp_arr, label="Battery Temp")
    ax.axhline(45, color="orange", linestyle="--", label="Warning (45C)")
    ax.axhline(60, color="red",    linestyle="--", label="Critical (60C)")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Temperature (C)")
    ax.legend()
    st.pyplot(fig)


# TAB 2: Hybrid validation
# Compares the pure physics model vs. hybrid (physics + ML residual correction)
with tab_hybrid:
    param_df    = load_parameter_evolution()
    cycle_files = list_cycle_files()[:max_train_files]

    if use_hybrid and cycle_files:
        train_files, val_files = split_cycles(cycle_files, parse_cycle_number, val_ratio=val_ratio)
        train_df = build_training_data(train_files, param_df, load_cycle_df, max_rows=max_train_rows)

        if train_df.empty:
            st.warning("Not enough data to train residual model.")
        else:
            rf_model = train_residual_rf(train_df)

            eval_files  = val_files if val_files else cycle_files
            eval_cycles = [parse_cycle_number(p) for p in eval_files]

            picked_cycle = st.selectbox("Cycle to validate", eval_cycles)
            picked_path  = eval_files[eval_cycles.index(picked_cycle)]

            eval_df       = load_cycle_df(picked_path)
            R_fit, hA_fit = lookup_fit_params(picked_cycle, param_df)

            # First measured temp is a good proxy for ambient at start of the cycle
            # (took us a while to figure this out — first tried using a fixed value)
            T_start = float(eval_df["temp_C"].iloc[0])

            phys_temp = thermal_simulation_profile(
                eval_df["time_s"].values,
                eval_df["current_A"].values,
                R_internal=R_fit,
                hA=hA_fit,
                T_amb=T_start,
                T0=T_start,
            )

            feats       = build_cycle_features(eval_df, picked_cycle, R_fit, hA_fit)
            correction  = predict_residual(rf_model, feats)
            hybrid_temp = phys_temp + correction

            m_phys = metrics(eval_df["temp_C"].values, phys_temp)
            m_hyb  = metrics(eval_df["temp_C"].values, hybrid_temp)

            col_a, col_b = st.columns(2)
            col_a.metric("RMSE (Physics)", f"{m_phys['rmse']:.3f}")
            col_b.metric("RMSE (Hybrid)",  f"{m_hyb['rmse']:.3f}")

            fig2, ax2 = plt.subplots()
            ax2.plot(eval_df["time_s"] / 60, eval_df["temp_C"], label="Measured")
            ax2.plot(eval_df["time_s"] / 60, phys_temp,         label="Physics")
            ax2.plot(eval_df["time_s"] / 60, hybrid_temp,       label="Hybrid")
            ax2.set_xlabel("Time (min)")
            ax2.set_ylabel("Temperature (C)")
            ax2.legend()
            st.pyplot(fig2)
    else:
        st.caption("Enable hybrid twin in the sidebar to run validation.")


# ---- TAB 3: SOH / RUL with uncertainty bands ----
with tab_soh_rul:
    cycle_files = list_cycle_files()[:max_train_files]

    if len(cycle_files) < 10:
        st.warning("Need more cycle files to train the SOH/RUL model.")
    else:
        cycle_summary = build_cycle_summary_df(cycle_files, load_cycle_df)

        if cycle_summary.empty:
            st.warning("Unable to compute cycle summary features.")
        else:
            cycle_summary = attach_rul_target(cycle_summary, soh_threshold=soh_threshold)

            # Time-ordered split — important, can't train on future data
            train_df, val_df, test_df = time_order_split(
                cycle_summary, val_ratio=soh_val_ratio, test_ratio=soh_test_ratio
            )

            # seed=42 for reproducibility
            ensemble = train_soh_rul_ensemble(train_df, n_models=n_ensemble, seed=42)

            pred_all  = predict_soh_rul_with_uncertainty(ensemble, cycle_summary, alpha=unc_alpha)
            pred_test = (
                predict_soh_rul_with_uncertainty(ensemble, test_df, alpha=unc_alpha)
                if not test_df.empty
                else pred_all.iloc[0:0].copy()
            )

            # Find predictions for the cycle closest to what's selected in sidebar
            idx      = (pred_all["cycle_number"] - cycle_num).abs().argmin()
            pred_row = pred_all.iloc[idx]

            c1, c2, c3 = st.columns(3)
            c1.metric(
                "Predicted SOH",
                f"{pred_row['soh_pred']:.3f}",
                f"[{pred_row['soh_lo']:.3f}, {pred_row['soh_hi']:.3f}]",
            )
            c2.metric(
                "Predicted RUL (cycles)",
                f"{pred_row['rul_pred']:.1f}",
                f"[{pred_row['rul_lo']:.1f}, {pred_row['rul_hi']:.1f}]",
            )
            c3.metric("Nominal Capacity (Ah)", f"{float(pred_row['nominal_capacity_ah']):.3f}")

            # TODO: maybe add a dropdown to pick which metric to highlight
            if not pred_test.empty:
                test_met = evaluate_soh_rul_predictions(pred_test)
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Test SOH RMSE",        f"{test_met['soh_rmse']:.4f}")
                m2.metric("Test RUL RMSE",         f"{test_met['rul_rmse']:.2f}")
                m3.metric("SOH Interval Coverage", f"{test_met['soh_interval_coverage']:.3f}")
                m4.metric("RUL Interval Coverage", f"{test_met['rul_interval_coverage']:.3f}")
                st.caption(
                    f"Split — train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)} "
                    f"(time-ordered, no leakage)"
                )

            conf_pct = int((1 - unc_alpha) * 100)

            # SOH plot
            fig3, ax3 = plt.subplots()
            ax3.plot(pred_all["cycle_number"], pred_all["soh_target"], color="black",   label="SOH actual")
            ax3.plot(pred_all["cycle_number"], pred_all["soh_pred"],   color="#2563eb", label="SOH predicted")
            ax3.fill_between(
                pred_all["cycle_number"],
                pred_all["soh_lo"], pred_all["soh_hi"],
                alpha=0.2, color="#2563eb",
                label=f"{conf_pct}% interval",
            )
            ax3.axhline(soh_threshold, linestyle="--", color="red", label="failure threshold")
            if len(train_df) > 0:
                ax3.axvline(float(train_df["cycle_number"].max()), linestyle="--", color="gray", alpha=0.6)
            ax3.set_xlabel("Cycle")
            ax3.set_ylabel("SOH")
            ax3.legend()
            st.pyplot(fig3)

            # RUL plot
            fig4, ax4 = plt.subplots()
            ax4.plot(pred_all["cycle_number"], pred_all["rul_target"], color="black",   label="RUL actual")
            ax4.plot(pred_all["cycle_number"], pred_all["rul_pred"],   color="#16a34a", label="RUL predicted")
            ax4.fill_between(
                pred_all["cycle_number"],
                pred_all["rul_lo"], pred_all["rul_hi"],
                alpha=0.2, color="#16a34a",
                label=f"{conf_pct}% interval",
            )
            if len(train_df) > 0:
                ax4.axvline(float(train_df["cycle_number"].max()), linestyle="--", color="gray", alpha=0.6)
            ax4.set_xlabel("Cycle")
            ax4.set_ylabel("RUL (cycles)")
            ax4.legend()
            st.pyplot(fig4)

            if export_results:
                out_dir = Path("results")
                out_dir.mkdir(parents=True, exist_ok=True)

                pred_all.to_csv(out_dir / "soh_rul_predictions.csv", index=False)
                pred_test.to_csv(out_dir / "soh_rul_test_predictions.csv", index=False)
                fig3.savefig(out_dir / "soh_uncertainty_curve.png",  dpi=180, bbox_inches="tight")
                fig4.savefig(out_dir / "rul_uncertainty_curve.png",  dpi=180, bbox_inches="tight")

                if not pred_test.empty:
                    pd.DataFrame([evaluate_soh_rul_predictions(pred_test)]).to_csv(
                        out_dir / "soh_rul_test_metrics.csv", index=False
                    )
                st.success("Exported to results/ folder.")

            st.caption("SOH uses discharge-capacity degradation. Metrics shown are test-split only.")


# TAB 4: Anomaly Detection
with tab_safety:
    cycle_files   = list_cycle_files()[:max_train_files]
    cycle_summary = build_cycle_summary_df(cycle_files, load_cycle_df)

    if cycle_summary.empty or len(cycle_summary) < 10:
        st.warning("Need more cycle summaries for anomaly analysis.")
    else:
        # Isolation Forest learns what "normal" looks like, then flags outliers
        anom_model = train_cycle_anomaly_detector(cycle_summary, contamination=contamination, seed=42)
        scored     = score_cycle_anomalies(anom_model, cycle_summary)

        # Find the row nearest to the sidebar cycle number
        nearest_idx = (scored["cycle_number"] - cycle_num).abs().argmin()
        nearest_num = int(scored.iloc[nearest_idx]["cycle_number"])
        nearest_row = scored[scored["cycle_number"] == nearest_num].iloc[0]

        st.metric("Anomaly status (selected cycle)", nearest_row["anomaly_label"])
        st.metric("Anomaly score", f"{float(nearest_row['anomaly_score']):.4f}")

        # Top 10 most suspicious cycles
        top10 = scored.head(10)[["cycle_number", "anomaly_label", "anomaly_score", "max_temp_C", "temp_rise_C"]]
        st.write("Top potentially anomalous cycles")
        st.dataframe(top10, use_container_width=True)

        # Scatter — red = anomaly, green = normal
        dot_colors = np.where(scored["anomaly_label"].values == "Anomaly", "#dc2626", "#16a34a")
        fig5, ax5 = plt.subplots()
        ax5.scatter(scored["cycle_number"], scored["anomaly_score"], c=dot_colors, alpha=0.8)
        ax5.set_title("Cycle-level anomaly scores")
        ax5.set_xlabel("Cycle")
        ax5.set_ylabel("Anomaly score (higher = more abnormal)")
        st.pyplot(fig5)


# TAB 5: Raw data summary
with tab_data:
    st.write({
        "profile":        selected_profile,
        "heat_loss_kj":   round(float(heat_loss_kj), 3),
        "cell_cap_ah":    round(float(cell_cap_ah), 3),
        "range_left_km":  round(float(range_left_km), 2),
        "n_ensemble":     n_ensemble,
        "unc_alpha":      unc_alpha,
        "soh_val_ratio":  soh_val_ratio,
        "soh_threshold":  soh_threshold,
        "soh_test_ratio": soh_test_ratio,
        "contamination":  contamination,
    })