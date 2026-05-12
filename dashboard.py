import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


# data loading

@st.cache_data(show_spinner=False)
def load_fitted_params(path="results/parameter_evolution_B0005.csv"):
    # load pre-fitted R and hA from the parameter fitting step
    # return empty frame if file doesn't exist yet — avoids crashing on first run
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=["cycle", "R_fit", "hA_fit"])


@st.cache_data(show_spinner=False)
def get_cycle_files(folder="data/processed"):
    files = sorted(
        Path(folder).glob("B0005_cycle_*.csv"),
        key=lambda f: get_cycle_num(f)
    )
    return [str(f) for f in files]


@st.cache_data(show_spinner=False)
def load_cycle_csv(path):
    return pd.read_csv(path)


def get_cycle_num(filepath):
    # e.g. B0005_cycle_42.csv -> 42
    try:
        return int(Path(filepath).stem.split("_cycle_")[-1])
    except Exception:
        return None


def lookup_cycle_params(cycle_num, df_fitted, fallback_R=0.01, fallback_hA=0.1):
    # find closest matching cycle in the fitted params table
    # fall back to defaults if the table is empty or missing columns
    if df_fitted.empty or "cycle" not in df_fitted.columns:
        return fallback_R, fallback_hA
    idx = (df_fitted["cycle"] - cycle_num).abs().idxmin()
    row = df_fitted.loc[idx]
    R  = float(row.get("R_fit",  fallback_R))
    hA = float(row.get("hA_fit", fallback_hA))
    return R, hA


# physics 

def run_thermal_sim(current, duration=1200, R=0.01, hA=0.1,
                    mass=0.045, Cp=900, T_amb=25):
    """
    Simulates battery temperature for a constant current draw.
    dT = (I^2*R - hA*(T-T_amb)) * dt / (m*Cp)
    Also returns total heat lost and energy consumed.
    """
    t_arr     = np.arange(0, duration, 1)
    I_arr     = np.ones_like(t_arr) * current
    temp      = np.zeros_like(t_arr, dtype=float)
    temp[0]   = T_amb
    dt        = 1.0
    heat_lost = 0
    energy_Ah = 0

    for i in range(1, len(t_arr)):
        Q_gen  = (I_arr[i] ** 2) * R          # joule heating
        Q_loss = hA * (temp[i-1] - T_amb)     # cooling
        dT = (Q_gen - Q_loss) * dt / (mass * Cp)
        temp[i] = temp[i-1] + dT
        heat_lost += Q_loss * dt
        energy_Ah += abs(I_arr[i]) * dt

    return t_arr, temp, heat_lost / 1000, energy_Ah / 3600


def run_thermal_sim_profile(time_s, current_A, R=0.01, hA=0.1,
                            mass=0.045, Cp=900, T_amb=25, T0=None):
    """
    Same model but for a real (variable) current profile from a CSV.
    Used when comparing against actual measured temperature.
    """
    time_s    = np.asarray(time_s)
    current_A = np.asarray(current_A)
    temp      = np.zeros_like(time_s, dtype=float)
    temp[0]   = T_amb if T0 is None else float(T0)

    dt_arr = np.diff(time_s, prepend=time_s[0])
    if len(dt_arr) > 1:
        dt_arr[0] = np.median(dt_arr[1:])  # first diff is a dummy, fix it

    for i in range(1, len(time_s)):
        Q_gen  = (current_A[i] ** 2) * R
        Q_loss = hA * (temp[i-1] - T_amb)
        dT = (Q_gen - Q_loss) * dt_arr[i] / (mass * Cp)
        temp[i] = temp[i-1] + dT

    return temp


def calc_resistance(n, R0=0.01, kR=5e-4):
    # R grows linearly: R(n) = R0 * (1 + kR * n)
    return R0 * (1 + kR * n)


def calc_capacity(n, Q0=2.0, kQ=1e-4):
    # Q fades linearly, clamped at 60% of original
    return max(Q0 * (1 - kQ * n), Q0 * 0.6)


#  driver alerts 

def get_status(max_temp, range_km, speed):
    # priority: safety first, then range, then efficiency
    if max_temp > 60:
        return "DANGER: Battery overheating!", "red", "Reduce speed and enable cooling immediately."
    if max_temp > 45:
        return "Caution: Battery heating up.", "orange", "Drive slower or avoid aggressive acceleration."
    if range_km < 50:
        return "Low Range: Less than 50 km left.", "orange", "Find the nearest charging station."
    if speed < 50:
        return "Efficient driving.", "green", "Maintain your current speed."
    return "Safe operation.", "green", "Good driving, keep it steady."


#  ML pipeline 

FEATURE_COLS = [
    "current_A", "voltage_V", "time_s", "cycle_number",
    "temp_phys", "dI", "dV", "dT_phys", "I2R"
]

@st.cache_data(show_spinner=False)
def build_training_data(files, df_fitted, max_rows=20000, seed=42):
    """
    For each cycle: run physics model, compute residual vs measured,
    build derivative features. Returns a combined DataFrame for ML training.
    """
    frames = []

    for path in files:
        df = pd.read_csv(path)
        if df.empty:
            continue

        if "cycle_number" in df.columns:
            cyc = int(df["cycle_number"].iloc[0])
        else:
            cyc = get_cycle_num(path)

        R_fit, hA_fit = lookup_cycle_params(cyc, df_fitted)
        T0 = float(df["temp_C"].iloc[0])

        T_phys = run_thermal_sim_profile(
            df["time_s"].values, df["current_A"].values,
            R=R_fit, hA=hA_fit, T_amb=T0, T0=T0
        )

        # rate-of-change features help the model understand dynamics
        dt   = np.diff(df["time_s"].values,    prepend=df["time_s"].values[0])
        dI   = np.diff(df["current_A"].values, prepend=df["current_A"].values[0])
        dV   = np.diff(df["voltage_V"].values, prepend=df["voltage_V"].values[0])
        dT   = np.diff(T_phys, prepend=T_phys[0]) / np.maximum(dt, 1e-6)
        I2R  = (df["current_A"].values ** 2) * R_fit

        feat = pd.DataFrame({
            "current_A":    df["current_A"].values,
            "voltage_V":    df["voltage_V"].values,
            "time_s":       df["time_s"].values,
            "cycle_number": cyc,
            "temp_phys":    T_phys,
            "dI":           dI,
            "dV":           dV,
            "dT_phys":      dT,
            "I2R":          I2R,
        })
        feat["residual"] = df["temp_C"].values - T_phys  # what the physics missed
        frames.append(feat)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    if len(combined) > max_rows:
        combined = combined.sample(n=max_rows, random_state=seed)
    return combined


@st.cache_resource(show_spinner=False)
def train_rf_model(train_df):
    # Random Forest trained to predict the residual (physics model error)
    # hybrid output = physics prediction + RF correction
    X = train_df[FEATURE_COLS]
    y = train_df["residual"]
    rf = RandomForestRegressor(n_estimators=120, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    return rf


def split_train_val(files, val_frac=0.2):
    # latest cycles go to validation — mirrors real deployment (train on past, test on future)
    all_nums = sorted([c for c in [get_cycle_num(p) for p in files] if c is not None])
    if not all_nums:
        return files, []
    n_val   = max(1, int(len(all_nums) * val_frac))
    val_set = set(all_nums[-n_val:])
    return (
        [p for p in files if get_cycle_num(p) not in val_set],
        [p for p in files if get_cycle_num(p) in val_set]
    )


def calc_metrics(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    return {"rmse": rmse, "mae": mae}


#  page setup 

st.set_page_config(page_title="EV Battery Digital Twin", layout="wide")

st.markdown("""
    <style>
    :root {
        --bg: #0b0f1a;
        --panel: #121826;
        --accent: #ef4444;
        --text: #e2e8f0;
        --muted: #94a3b8;
        --card: #f3f4f6;
        --card-text: #0f172a;
        --success: #22c55e;
    }
    .main {
        background:
            radial-gradient(circle at 20% 10%, rgba(59,130,246,0.18), transparent 40%),
            radial-gradient(circle at 80% 10%, rgba(148,163,184,0.18), transparent 40%),
            linear-gradient(180deg, #0b0f1a 0%, #0a0f19 100%);
    }
    .block-container { padding-top: 2rem; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1524 0%, #0c111e 100%);
        border-right: 1px solid rgba(148,163,184,0.15);
    }
    .title-box {
        background: linear-gradient(135deg, rgba(17,24,39,0.9), rgba(11,18,32,0.9));
        color: var(--text);
        padding: 1.4rem 1.8rem;
        border-radius: 16px;
        border: 1px solid rgba(148,163,184,0.2);
        box-shadow: 0 16px 44px rgba(2,8,23,0.6);
    }
    .metric-card {
        background: var(--card);
        color: var(--card-text);
        border: 1px solid rgba(148,163,184,0.35);
        border-radius: 16px;
        padding: 1rem 1.2rem;
        box-shadow: 0 10px 28px rgba(2,8,23,0.35);
    }
    .section-title { font-weight: 700; color: var(--card-text); }
    .accent { color: #60a5fa; font-weight: 600; }
    .glass {
        background: rgba(15,23,42,0.6);
        color: var(--text);
        border: 1px solid rgba(148,163,184,0.18);
        border-radius: 14px;
        padding: 1rem 1.2rem;
        box-shadow: 0 10px 28px rgba(2,8,23,0.45);
    }
    .status-pill {
        background: rgba(15,23,42,0.7);
        border: 1px solid rgba(148,163,184,0.2);
        border-radius: 999px;
        padding: 0.75rem 1rem;
        box-shadow: 0 10px 24px rgba(2,8,23,0.45);
    }
    .white-card {
        background: #f8fafc;
        border: 1px solid rgba(148,163,184,0.35);
        border-radius: 12px;
        padding: 0.6rem 0.8rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="title-box">
        <div style="font-size:30px; font-weight:700;">EV Battery Digital Twin</div>
        <div style="font-size:15px; color:#cbd5f0; margin-top:4px;">
            Thermal · Degradation · Range Forecasting Dashboard
        </div>
    </div>
""", unsafe_allow_html=True)
st.write("")


#  sidebar 

driving_profiles = {
    "City Commute":   {"T_amb": 25, "speed": 40,  "style": "Eco",        "road": "City",    "soc": 80},
    "Highway Trip":   {"T_amb": 30, "speed": 100, "style": "Normal",     "road": "Highway", "soc": 90},
    "Mountain Drive": {"T_amb": 20, "speed": 60,  "style": "Aggressive", "road": "Uphill",  "soc": 70},
    "Sports Mode":    {"T_amb": 35, "speed": 140, "style": "Aggressive", "road": "Highway", "soc": 100},
}

selected_profile = st.sidebar.selectbox("Choose Driving Profile", list(driving_profiles.keys()))
preset = driving_profiles[selected_profile]

st.sidebar.markdown("### Environment and Driving")
T_amb  = st.sidebar.selectbox("Ambient Temp (°C)", list(range(-20, 65, 5)),
                               index=preset["T_amb"] // 5 + 4)
speed  = st.sidebar.slider("Speed (km/h)", 20, 160, preset["speed"], 10)
style  = st.sidebar.selectbox("Driving Style", ["Eco", "Normal", "Aggressive"],
                               index=["Eco", "Normal", "Aggressive"].index(preset["style"]))
road   = st.sidebar.selectbox("Road Type", ["City", "Highway", "Uphill"],
                               index=["City", "Highway", "Uphill"].index(preset["road"]))
soc    = st.sidebar.slider("SOC (%)", 10, 100, preset["soc"], 5)

st.sidebar.markdown("### Battery Health")
cycle_num = st.sidebar.slider("Cycle Number (Battery Age)", 0, 1000, 50, 10)
R0        = st.sidebar.number_input("Baseline R0 (Ohm)", 0.005, 0.2,  0.01,  0.001, "%.3f")
kR        = st.sidebar.number_input("Resistance Growth kR", 1e-5, 5e-3, 5e-4, 1e-5, "%.5f")
Q0        = st.sidebar.number_input("Baseline Capacity Q0 (Ah)", 1.0, 5.0, 2.0, 0.1, "%.1f")
kQ        = st.sidebar.number_input("Capacity Fade kQ", 1e-5, 5e-3, 1e-4, 1e-5, "%.5f")
R_limit   = st.sidebar.number_input("RUL Threshold R_limit (Ohm)", 0.02, 0.3, 0.08, 0.005, "%.3f")

st.sidebar.markdown("### Simulation")
duration         = st.sidebar.slider("Simulation Duration (min)", 10, 60, 30, 5) * 60
show_param_evo   = st.sidebar.checkbox("Show fitted parameter evolution", True)
enable_hybrid    = st.sidebar.checkbox("Enable ML-Physics Hybrid Twin", True)
max_train_rows   = st.sidebar.slider("ML training rows", 5000, 40000, 20000, 5000)
max_train_cycles = st.sidebar.slider("Training cycles", 10, 120, 60, 10)
val_frac         = st.sidebar.slider("Validation split", 0.1, 0.4, 0.2, 0.05)


#  compute key values 

# cooling improves with speed — more airflow over the pack
hA = 0.05 + (speed / 160) * 0.15

# current draw based on driving aggressiveness and road type
current_map = {
    "Eco":        {"City": 30,  "Highway": 50,  "Uphill": 60},
    "Normal":     {"City": 50,  "Highway": 100, "Uphill": 120},
    "Aggressive": {"City": 80,  "Highway": 150, "Uphill": 180},
}
current_draw = current_map[style][road] * (speed / 100)

pack_Ah  = 20000
V_nom    = 3.7
pack_kWh = (pack_Ah * V_nom) / 1000

R_now  = calc_resistance(cycle_num, R0=R0, kR=kR)
Q_cell = calc_capacity(cycle_num, Q0=Q0, kQ=kQ)

t_sim, T_sim, heat_kJ, _ = run_thermal_sim(
    current_draw, duration, T_amb=T_amb, hA=hA, R=R_now
)

consumed_Ah = current_draw * duration / 3600
soc_used    = (consumed_Ah / pack_Ah) * 100
soc_left    = max(0, soc - soc_used)
energy_left = (soc_left / 100) * pack_kWh
range_km    = energy_left / 0.18  # 0.18 kWh/km assumed efficiency

status_msg, status_col, action_msg = get_status(T_sim.max(), range_km, speed)

# RUL: from R(n) = R0*(1+kR*n), solve for n when R = R_limit
n_limit    = max(0, int(((R_limit / max(R0, 1e-6)) - 1) / max(kR, 1e-9)))
rul_cycles = max(0, n_limit - cycle_num)


#  metric cards 

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(
        f'<div class="metric-card"><div class="section-title">Max Temp</div>'
        f'<div style="font-size:26px;font-weight:700;">{T_sim.max():.1f}°C</div>'
        f'<div style="opacity:0.7;">Ambient: {T_amb}°C</div></div>',
        unsafe_allow_html=True)
with c2:
    st.markdown(
        f'<div class="metric-card"><div class="section-title">SOC Left</div>'
        f'<div style="font-size:26px;font-weight:700;">{soc_left:.1f}%</div>'
        f'<div style="opacity:0.7;">~{range_km:.1f} km Range</div></div>',
        unsafe_allow_html=True)
with c3:
    st.markdown(
        f'<div class="metric-card"><div class="section-title">R_internal</div>'
        f'<div style="font-size:26px;font-weight:700;">{R_now:.4f} Ω</div>'
        f'<div style="opacity:0.7;">Cycle {cycle_num}</div></div>',
        unsafe_allow_html=True)
with c4:
    st.markdown(
        f'<div class="metric-card"><div class="section-title">RUL</div>'
        f'<div style="font-size:26px;font-weight:700;">{rul_cycles:,} cycles</div>'
        f'<div style="opacity:0.7;">Limit: {R_limit:.3f} Ω</div></div>',
        unsafe_allow_html=True)

st.write("")
st.markdown(
    f'<div class="status-pill">'
    f'<span style="color:#22c55e;font-weight:700;">✔ {status_msg}</span>'
    f'<span style="color:#e2e8f0;margin-left:8px;">{action_msg}</span>'
    f'</div>',
    unsafe_allow_html=True)


#  tabs 

tab_overview, tab_thermal, tab_hybrid, tab_degradation, tab_data = st.tabs(
    ["Overview", "Thermal", "Hybrid Twin", "Degradation", "Data"]
)

with tab_overview:
    st.subheader("Key Outputs")
    left_col, right_col = st.columns(2)

    with left_col:
        st.markdown("**Energy Breakdown**")
        st.markdown(f"- Battery energy left: {energy_left:.1f} kWh (~{range_km:.1f} km)")
        st.markdown(f"- Heat loss during run: {heat_kJ / 3600:.2f} kWh")
        st.markdown(f"- Drive energy used: {consumed_Ah * V_nom / 1000:.2f} kWh")
        st.markdown(f"- Estimated cell capacity: {Q_cell:.2f} Ah")

    with right_col:
        st.markdown('<div class="white-card">', unsafe_allow_html=True)
        fig, ax = plt.subplots()
        ax.bar(
            ["Energy Left", "Heat Loss", "Drive Use"],
            [energy_left, heat_kJ / 3600, consumed_Ah * V_nom / 1000],
            color=["#2563eb", "#f59e0b", "#ef4444"]
        )
        ax.set_ylabel("kWh")
        ax.set_title("Energy Summary")
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("**Digital Twin Stack**")
    st.markdown(
        "- Physics core: lumped thermal model + degradation laws\n"
        "- ML layer: residual correction trained on real cycle data\n"
        "- Hybrid output: physics + ML correction combined\n"
        "- Decision layer: safety + range + remaining life guidance"
    )
    st.markdown(
        '<div class="glass"><span class="accent">Project highlight:</span> '
        "Hybrid digital twin fusing first-principles thermal physics "
        "with a data-driven correction layer.</div>",
        unsafe_allow_html=True)


with tab_thermal:
    st.subheader("Thermal Profile")
    fig, ax = plt.subplots()
    ax.plot(t_sim / 60, T_sim, 'r-', label="Battery Temp (°C)")
    ax.axhline(45, color="orange", linestyle="--", label="Warning 45°C")
    ax.axhline(60, color="red",    linestyle="--", label="Critical 60°C")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Temperature (°C)")
    ax.legend()
    st.pyplot(fig)


with tab_hybrid:
    st.subheader("Hybrid Physics + ML Twin")
    df_fitted = load_fitted_params()
    cyc_files = get_cycle_files()

    if enable_hybrid and cyc_files:
        train_files, val_files = split_train_val(cyc_files[:max_train_cycles], val_frac=val_frac)
        train_df = build_training_data(train_files, df_fitted, max_rows=max_train_rows)

        if train_df.empty:
            st.warning("Not enough data to train the ML model.")
        else:
            rf = train_rf_model(train_df)

            eval_files  = val_files if val_files else get_cycle_files()
            eval_cycles = [get_cycle_num(p) for p in eval_files]

            chosen = st.selectbox("Choose a cycle to validate", eval_cycles, index=0)
            df_val = load_cycle_csv(eval_files[eval_cycles.index(chosen)])

            R_fit, hA_fit = lookup_cycle_params(chosen, df_fitted)
            T0_val = float(df_val["temp_C"].iloc[0])

            T_phys = run_thermal_sim_profile(
                df_val["time_s"].values, df_val["current_A"].values,
                R=R_fit, hA=hA_fit, T_amb=T0_val, T0=T0_val
            )

            dt  = np.diff(df_val["time_s"].values,    prepend=df_val["time_s"].values[0])
            dI  = np.diff(df_val["current_A"].values, prepend=df_val["current_A"].values[0])
            dV  = np.diff(df_val["voltage_V"].values, prepend=df_val["voltage_V"].values[0])
            dT  = np.diff(T_phys, prepend=T_phys[0]) / np.maximum(dt, 1e-6)
            I2R = (df_val["current_A"].values ** 2) * R_fit

            X_val = pd.DataFrame({
                "current_A":    df_val["current_A"].values,
                "voltage_V":    df_val["voltage_V"].values,
                "time_s":       df_val["time_s"].values,
                "cycle_number": chosen,
                "temp_phys":    T_phys,
                "dI":           dI,
                "dV":           dV,
                "dT_phys":      dT,
                "I2R":          I2R,
            })

            T_hybrid = T_phys + rf.predict(X_val)  # physics + ML correction

            m_phys   = calc_metrics(df_val["temp_C"].values, T_phys)
            m_hybrid = calc_metrics(df_val["temp_C"].values, T_hybrid)

            col1, col2, col3 = st.columns(3)
            with col1: st.metric("RMSE Physics", f"{m_phys['rmse']:.3f} °C")
            with col2: st.metric("RMSE Hybrid",  f"{m_hybrid['rmse']:.3f} °C")
            with col3: st.metric("Cycle", str(chosen))

            fig, ax = plt.subplots()
            ax.plot(df_val["time_s"] / 60, df_val["temp_C"], label="Measured", color="#111827")
            ax.plot(df_val["time_s"] / 60, T_phys,          label="Physics",  color="#ef4444")
            ax.plot(df_val["time_s"] / 60, T_hybrid,        label="Hybrid",   color="#2563eb")
            ax.set_xlabel("Time (min)")
            ax.set_ylabel("Temperature (°C)")
            ax.set_title("Measured vs Physics vs Hybrid")
            ax.legend()
            st.pyplot(fig)

            st.markdown("**Validation Summary**")
            st.write({
                "MAE_Physics":  round(m_phys["mae"],    3),
                "MAE_Hybrid":   round(m_hybrid["mae"],  3),
                "RMSE_Physics": round(m_phys["rmse"],   3),
                "RMSE_Hybrid":  round(m_hybrid["rmse"], 3),
                "Train_cycles": len(train_files),
                "Val_cycles":   len(val_files),
            })
    else:
        st.info("Enable the Hybrid Twin in the sidebar to train the ML residual model.")


with tab_degradation:
    st.subheader("Degradation Trends")

    c_range = np.arange(0, 1001, 10)
    R_curve = calc_resistance(c_range, R0=R0, kR=kR)
    Q_curve = [calc_capacity(c, Q0=Q0, kQ=kQ) for c in c_range]

    fig, ax1 = plt.subplots()
    ax1.plot(c_range, R_curve, color="#0ea5e9", label="R_internal (Ω)")
    ax1.set_xlabel("Cycle")
    ax1.set_ylabel("Resistance (Ω)", color="#0ea5e9")
    ax1.tick_params(axis='y', labelcolor="#0ea5e9")
    ax2 = ax1.twinx()
    ax2.plot(c_range, Q_curve, color="#22c55e", label="Capacity (Ah)")
    ax2.set_ylabel("Capacity (Ah)", color="#22c55e")
    ax2.tick_params(axis='y', labelcolor="#22c55e")
    ax1.set_title("Projected Degradation")
    st.pyplot(fig)

    if show_param_evo:
        df_fitted = load_fitted_params()
        if not df_fitted.empty:
            fig, ax = plt.subplots()
            ax.plot(df_fitted["cycle"], df_fitted["R_fit"],  label="Fitted R")
            ax.plot(df_fitted["cycle"], df_fitted["hA_fit"], label="Fitted hA")
            ax.set_xlabel("Cycle")
            ax.set_ylabel("Value")
            ax.set_title("Fitted Parameter Evolution (B0005)")
            ax.legend()
            st.pyplot(fig)
        else:
            st.warning("Fitted parameter file not found.")


with tab_data:
    st.subheader("Scenario Inputs")
    st.write({
        "profile":          selected_profile,
        "ambient_temp_C":   T_amb,
        "speed_kmph":       speed,
        "style":            style,
        "road":             road,
        "soc_pct":          soc,
        "cycle_num":        cycle_num,
        "R_internal":       round(float(R_now), 5),
        "cell_capacity_Ah": round(float(Q_cell), 3),
        "sim_duration_s":   duration,
    })


st.markdown("---")
st.markdown("Try different profiles, ambient temperatures, and speeds to explore how driving affects battery health.")