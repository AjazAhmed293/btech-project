# import streamlit as st
# import numpy as np
# import matplotlib.pyplot as plt

# # --- Core Thermal Model ---
# def thermal_simulation(current, duration=1200, R_internal=0.01, hA=0.1, 
#                        m=0.045, Cp=900, T_amb=25):
#     time_profile = np.arange(0, duration, 1)
#     current_profile = np.ones_like(time_profile) * current
    
#     T = np.zeros_like(time_profile, dtype=float)
#     T[0] = T_amb
#     dt = 1.0
#     heat_loss_total = 0
#     drive_energy_total = 0
    
#     for t in range(1, len(time_profile)):
#         Q_gen = (current_profile[t]**2) * R_internal  # heating
#         Q_loss = hA * (T[t-1] - T_amb)               # cooling
#         dT = (Q_gen - Q_loss) * dt / (m * Cp)
#         T[t] = T[t-1] + dT
#         heat_loss_total += Q_loss * dt
#         drive_energy_total += abs(current_profile[t]) * dt
    
#     return time_profile, T, heat_loss_total/1000, drive_energy_total/3600  # kJ, Ah

# def driver_feedback(max_temp, range_left_km, speed):
#     if max_temp > 60:
#         return "🚨 DANGER! Battery overheating!", "red", "Reduce speed & enable cooling system immediately."
#     elif max_temp > 45:
#         return "⚠️ Caution! Battery is heating up.", "orange", "Drive slower or avoid aggressive acceleration."
#     elif range_left_km < 50:
#         return "⚠️ Low Range! Less than 50 km left.", "orange", "Find the nearest charging station."
#     elif speed < 50:
#         return "👏 Excellent! You are driving efficiently.", "green", "Maintain your current speed."
#     else:
#         return "✅ Safe operation.", "green", "Good driving, keep it steady!"

# # --- Dashboard UI ---
# st.title("🚗 EV Battery Thermal Management System (BTMS)")
# st.markdown("### Interactive EV Digital Twin Dashboard – Advanced Edition")

# # Preset Driving Profiles
# profiles = {
#     "City Commute 🚗": {"T_amb": 25, "speed": 40, "style": "Eco", "road": "City", "soc": 80},
#     "Highway Trip 🛣️": {"T_amb": 30, "speed": 100, "style": "Normal", "road": "Highway", "soc": 90},
#     "Mountain Drive 🏔️": {"T_amb": 20, "speed": 60, "style": "Aggressive", "road": "Uphill", "soc": 70},
#     "Sports Mode ⚡": {"T_amb": 35, "speed": 140, "style": "Aggressive", "road": "Highway", "soc": 100},
# }

# selected_profile = st.sidebar.selectbox("Choose Driving Profile", list(profiles.keys()))
# preset = profiles[selected_profile]

# # Override with user-adjustable sliders
# T_amb = st.sidebar.selectbox("Ambient Temp (°C)", list(range(-20, 65, 5)), index=preset["T_amb"]//5+4)
# speed = st.sidebar.slider("Speed (km/h)", 20, 160, preset["speed"], 10)
# style = st.sidebar.selectbox("Driving Style", ["Eco", "Normal", "Aggressive"], index=["Eco","Normal","Aggressive"].index(preset["style"]))
# road = st.sidebar.selectbox("Road Type", ["City", "Highway", "Uphill"], index=["City","Highway","Uphill"].index(preset["road"]))
# soc = st.sidebar.slider("SOC (%)", 10, 100, preset["soc"], 5)

# # Dynamic cooling (higher at higher speed)
# hA = 0.05 + (speed/160)*0.15

# # Current mapping
# base_map = {
#     "Eco": {"City": 30, "Highway": 50, "Uphill": 60},
#     "Normal": {"City": 50, "Highway": 100, "Uphill": 120},
#     "Aggressive": {"City": 80, "Highway": 150, "Uphill": 180},
# }
# base_current = base_map[style][road]
# current_draw = base_current * (speed / 100)

# # Battery pack
# capacity_Ah = 20000   # ~74 kWh
# V_nominal = 3.7
# pack_energy_kWh = (capacity_Ah * V_nominal) / 1000
# duration = 1800  # 30 min

# # Simulation
# time_profile, T, heat_loss_kJ, drive_energy_Ah = thermal_simulation(current_draw, duration, T_amb=T_amb, hA=hA)
# consumed_Ah = current_draw * duration / 3600
# capacity_used_pct = (consumed_Ah / capacity_Ah) * 100
# range_left_pct = max(0, soc - capacity_used_pct)

# # Range calculation
# efficiency_kWh_per_km = 0.18
# energy_left_kWh = (range_left_pct / 100) * pack_energy_kWh
# range_left_km = energy_left_kWh / efficiency_kWh_per_km

# # Feedback
# status, color, action = driver_feedback(T.max(), range_left_km, speed)

# # --- Outputs ---
# st.subheader("🔋 Simulation Results")
# st.markdown(f"**Ambient Temp:** {T_amb} °C")
# st.markdown(f"**Max Battery Temp:** {T.max():.1f} °C")
# st.markdown(f"**SOC Left:** {range_left_pct:.1f}%")
# st.markdown(f"**Driving Range Left:** {range_left_km:.1f} km")
# st.markdown(f"<span style='color:{color}; font-size:20px; font-weight:bold'>{status}</span>", unsafe_allow_html=True)

# st.subheader("👉 Suggested Action")
# st.info(action)

# # Energy breakdown
# st.subheader("⚡ Energy Breakdown")
# st.markdown(f"- 🔋 **Battery Energy Left:** {energy_left_kWh:.1f} kWh (~{range_left_km:.1f} km)")
# st.markdown(f"- 🔥 **Heat Loss:** {heat_loss_kJ/3600:.2f} kWh")
# st.markdown(f"- ⚡ **Driving Energy Used (30 min):** {consumed_Ah*V_nominal/1000:.2f} kWh")

# # Plot
# fig, ax = plt.subplots()
# ax.plot(time_profile/60, T, 'r-', label="Battery Temp (°C)")
# ax.axhline(45, color="orange", linestyle="--", label="Warning (45°C)")
# ax.axhline(60, color="red", linestyle="--", label="Critical (60°C)")
# ax.set_xlabel("Time (min)")
# ax.set_ylabel("Temperature (°C)")
# ax.legend()
# st.pyplot(fig)

# st.markdown("---")
# st.markdown("🎮 Try different **profiles, ambient temps, and speeds** to see how driving behavior impacts EV battery health.")
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- Core Thermal Model ---
def thermal_simulation(current, duration=1200, R_internal=0.01, hA=0.1, 
                       m=0.045, Cp=900, T_amb=25):
    time_profile = np.arange(0, duration, 1)
    current_profile = np.ones_like(time_profile) * current
    
    T = np.zeros_like(time_profile, dtype=float)
    T[0] = T_amb
    dt = 1.0
    heat_loss_total = 0
    drive_energy_total = 0
    
    for t in range(1, len(time_profile)):
        Q_gen = (current_profile[t]**2) * R_internal  # heating
        Q_loss = hA * (T[t-1] - T_amb)               # cooling
        dT = (Q_gen - Q_loss) * dt / (m * Cp)
        T[t] = T[t-1] + dT
        heat_loss_total += Q_loss * dt
        drive_energy_total += abs(current_profile[t]) * dt
    
    return time_profile, T, heat_loss_total/1000, drive_energy_total/3600  # kJ, Ah

### NEW ###
# --- Degradation Model ---
def degradation_model(cycle_num, R0=0.01, kR=5e-4):
    """
    Returns degraded resistance (Ω) at a given cycle.
    R increases linearly with cycles.
    """
    R = R0 * (1 + kR * cycle_num)
    return R

def driver_feedback(max_temp, range_left_km, speed):
    if max_temp > 60:
        return "🚨 DANGER! Battery overheating!", "red", "Reduce speed & enable cooling system immediately."
    elif max_temp > 45:
        return "⚠️ Caution! Battery is heating up.", "orange", "Drive slower or avoid aggressive acceleration."
    elif range_left_km < 50:
        return "⚠️ Low Range! Less than 50 km left.", "orange", "Find the nearest charging station."
    elif speed < 50:
        return "👏 Excellent! You are driving efficiently.", "green", "Maintain your current speed."
    else:
        return "✅ Safe operation.", "green", "Good driving, keep it steady!"

# --- Dashboard UI ---
st.title("🚗 EV Battery Thermal Management System (BTMS)")
st.markdown("### Interactive EV Digital Twin Dashboard – Advanced Edition")

# Preset Driving Profiles
profiles = {
    "City Commute 🚗": {"T_amb": 25, "speed": 40, "style": "Eco", "road": "City", "soc": 80},
    "Highway Trip 🛣️": {"T_amb": 30, "speed": 100, "style": "Normal", "road": "Highway", "soc": 90},
    "Mountain Drive 🏔️": {"T_amb": 20, "speed": 60, "style": "Aggressive", "road": "Uphill", "soc": 70},
    "Sports Mode ⚡": {"T_amb": 35, "speed": 140, "style": "Aggressive", "road": "Highway", "soc": 100},
}

selected_profile = st.sidebar.selectbox("Choose Driving Profile", list(profiles.keys()))
preset = profiles[selected_profile]

# Override with user-adjustable sliders
T_amb = st.sidebar.selectbox("Ambient Temp (°C)", list(range(-20, 65, 5)), index=preset["T_amb"]//5+4)
speed = st.sidebar.slider("Speed (km/h)", 20, 160, preset["speed"], 10)
style = st.sidebar.selectbox("Driving Style", ["Eco", "Normal", "Aggressive"], index=["Eco","Normal","Aggressive"].index(preset["style"]))
road = st.sidebar.selectbox("Road Type", ["City", "Highway", "Uphill"], index=["City","Highway","Uphill"].index(preset["road"]))
soc = st.sidebar.slider("SOC (%)", 10, 100, preset["soc"], 5)

### NEW ###
# Add this slider to the sidebar controls
cycle_num = st.sidebar.slider("Cycle Number (Battery Age)", 0, 1000, 50, 10)

# Dynamic cooling (higher at higher speed)
hA = 0.05 + (speed/160)*0.15

# Current mapping
base_map = {
    "Eco": {"City": 30, "Highway": 50, "Uphill": 60},
    "Normal": {"City": 50, "Highway": 100, "Uphill": 120},
    "Aggressive": {"City": 80, "Highway": 150, "Uphill": 180},
}
base_current = base_map[style][road]
current_draw = base_current * (speed / 100)

# Battery pack
capacity_Ah = 20000   # ~74 kWh
V_nominal = 3.7
pack_energy_kWh = (capacity_Ah * V_nominal) / 1000
duration = 1800  # 30 min

### MODIFIED ###
# --- Simulation ---
# Calculate degraded internal resistance based on battery age
R_internal_degraded = degradation_model(cycle_num)

# Run the simulation with the new, dynamic resistance
time_profile, T, heat_loss_kJ, drive_energy_Ah = thermal_simulation(
    current_draw, duration, T_amb=T_amb, hA=hA, R_internal=R_internal_degraded
)

consumed_Ah = current_draw * duration / 3600
capacity_used_pct = (consumed_Ah / capacity_Ah) * 100
range_left_pct = max(0, soc - capacity_used_pct)

# Range calculation
efficiency_kWh_per_km = 0.18
energy_left_kWh = (range_left_pct / 100) * pack_energy_kWh
range_left_km = energy_left_kWh / efficiency_kWh_per_km

# Feedback
status, color, action = driver_feedback(T.max(), range_left_km, speed)

### MODIFIED ###
# --- Outputs ---
st.subheader("🔋 Simulation Results")
st.markdown(f"**Cycle Number:** {cycle_num}")
st.markdown(f"**Internal Resistance:** {R_internal_degraded:.4f} Ω")
st.markdown(f"**Ambient Temp:** {T_amb} °C")
st.markdown(f"**Max Battery Temp:** {T.max():.1f} °C")
st.markdown(f"**SOC Left:** {range_left_pct:.1f}%")
st.markdown(f"**Driving Range Left:** {range_left_km:.1f} km")
st.markdown(f"<span style='color:{color}; font-size:20px; font-weight:bold'>{status}</span>", unsafe_allow_html=True)

st.subheader("👉 Suggested Action")
st.info(action)

# Energy breakdown
st.subheader("⚡ Energy Breakdown")
st.markdown(f"- 🔋 **Battery Energy Left:** {energy_left_kWh:.1f} kWh (~{range_left_km:.1f} km)")
st.markdown(f"- 🔥 **Heat Loss:** {heat_loss_kJ/3600:.2f} kWh")
st.markdown(f"- ⚡ **Driving Energy Used (30 min):** {consumed_Ah*V_nominal/1000:.2f} kWh")

# Plot
fig, ax = plt.subplots()
ax.plot(time_profile/60, T, 'r-', label="Battery Temp (°C)")
ax.axhline(45, color="orange", linestyle="--", label="Warning (45°C)")
ax.axhline(60, color="red", linestyle="--", label="Critical (60°C)")
ax.set_xlabel("Time (min)")
ax.set_ylabel("Temperature (°C)")
ax.legend()
st.pyplot(fig)

st.markdown("---")
st.markdown("🎮 Try different **profiles, ambient temps, and speeds** to see how driving behavior impacts EV battery health.")