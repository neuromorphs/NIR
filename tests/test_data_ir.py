import numpy as np
import nir


def test_generate_time_gridded_data():
    spikes = np.random.randint(0, 2, size=(10, 10, 10)).astype(bool)
    dt = 0.1
    gridded = nir.TimeGriddedData(spikes, dt)
    node = nir.NIRNodeData({"spikes": gridded})
    graph = nir.NIRGraphData({"node": node})  # noqa: F841


def test_generate_event_data():
    idx = np.random.randint(0, 10, size=(10, 20))
    time = np.sort(np.random.rand(10, 20), axis=1)
    n_neurons = 10
    t_max = 1.0
    event = nir.EventData(idx, time, n_neurons, t_max)
    node = nir.NIRNodeData({"spikes": event})
    graph = nir.NIRGraphData({"node": node})  # noqa: F841


def test_conversion():
    # time_shift = 0.0 * dt
    spikes = np.random.randint(0, 2, size=(10, 10, 10)).astype(bool)
    dt = 0.1
    gridded_1 = nir.TimeGriddedData(spikes, dt)
    event = gridded_1.to_event(n_spikes=100)
    gridded_2 = event.to_time_gridded(dt=dt)
    assert np.array_equal(gridded_1.data, gridded_2.data)
    assert gridded_1.dt == gridded_2.dt

    # time_shift = 0.5 * dt
    spikes = np.random.randint(0, 2, size=(10, 10, 10)).astype(bool)
    dt = 0.1
    gridded_1 = nir.TimeGriddedData(spikes, dt)
    event = gridded_1.to_event(n_spikes=100, time_shift=0.5 * dt)
    gridded_2 = event.to_time_gridded(dt=dt)
    assert np.array_equal(gridded_1.data, gridded_2.data)
    assert gridded_1.dt == gridded_2.dt
