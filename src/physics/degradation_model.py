def degradation_model(cycle_num, R0=0.01, kR=5e-4):
    return R0 * (1 + kR * cycle_num)


def capacity_fade_model(cycle_num, Q0=2.0, kQ=1e-4):
    Q = Q0 * (1 - kQ * cycle_num)
    return max(Q, Q0 * 0.6)


def estimate_rul_cycles(cycle_num, R0=0.01, kR=5e-4, R_limit=0.08):
    cycles_to_limit = max(0, int(((R_limit / max(R0, 1e-6)) - 1) / max(kR, 1e-9)))
    return max(0, cycles_to_limit - cycle_num)
