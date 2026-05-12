import numpy as np


def thermal_simulation(
    current,
    duration=1200,
    R_internal=0.01,
    hA=0.1,
    m=0.045,
    Cp=900,
    T_amb=25,
):
    time_profile = np.arange(0, duration, 1)
    current_profile = np.ones_like(time_profile) * current

    T = np.zeros_like(time_profile, dtype=float)
    T[0] = T_amb
    dt = 1.0
    heat_loss_total = 0.0
    drive_energy_total = 0.0

    for t in range(1, len(time_profile)):
        Q_gen = (current_profile[t] ** 2) * R_internal
        Q_loss = hA * (T[t - 1] - T_amb)
        dT = (Q_gen - Q_loss) * dt / (m * Cp)
        T[t] = T[t - 1] + dT
        heat_loss_total += Q_loss * dt
        drive_energy_total += abs(current_profile[t]) * dt

    return time_profile, T, heat_loss_total / 1000.0, drive_energy_total / 3600.0


def thermal_simulation_profile(
    time_s,
    current_profile,
    R_internal=0.01,
    hA=0.1,
    m=0.045,
    Cp=900,
    T_amb=25,
    T0=None,
):
    time_s = np.asarray(time_s)
    current_profile = np.asarray(current_profile)
    T = np.zeros_like(time_s, dtype=float)
    T[0] = T_amb if T0 is None else float(T0)

    dt = np.diff(time_s, prepend=time_s[0])
    if len(dt) > 1:
        dt[0] = np.median(dt[1:])

    for t in range(1, len(time_s)):
        Q_gen = (current_profile[t] ** 2) * R_internal
        Q_loss = hA * (T[t - 1] - T_amb)
        dT = (Q_gen - Q_loss) * dt[t] / (m * Cp)
        T[t] = T[t - 1] + dT

    return T
