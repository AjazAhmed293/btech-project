import numpy as np

from src.physics.thermal_model import thermal_simulation


def test_thermal_simulation_shapes():
    time_s, temp, heat_loss_kj, drive_ah = thermal_simulation(current=50, duration=100)
    assert len(time_s) == 100
    assert len(temp) == 100
    assert isinstance(heat_loss_kj, float)
    assert isinstance(drive_ah, float)


def test_thermal_simulation_heats_up():
    _, temp, _, _ = thermal_simulation(current=80, duration=300)
    assert np.max(temp) > temp[0]
