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


def test_generate_valued_event_data():
    idx = np.random.randint(0, 10, size=(10, 20))
    time = np.sort(np.random.rand(10, 20), axis=1)
    values = np.random.rand(10, 20)
    n_neurons = 10
    t_max = 1.0
    valued_event = nir.ValuedEventData(idx, time, n_neurons, t_max, values)
    node = nir.NIRNodeData({"current": valued_event})
    graph = nir.NIRGraphData({"node": node})  # noqa: F841


def test_binary_conversion():
    # time_shift = 0.0 * dt
    spikes = np.random.randint(0, 2, size=(10, 10, 10)).astype(bool)
    dt = 0.1
    gridded_1 = nir.TimeGriddedData(spikes, dt)
    event = gridded_1.to_event(n_events=100)
    gridded_2 = event.to_time_gridded(dt=dt)
    assert np.array_equal(gridded_1.data, gridded_2.data)
    assert gridded_1.dt == gridded_2.dt

    # time_shift = 0.5 * dt
    spikes = np.random.randint(0, 2, size=(10, 10, 10)).astype(bool)
    dt = 0.1
    gridded_1 = nir.TimeGriddedData(spikes, dt)
    event = gridded_1.to_event(n_events=100, time_shift=0.5 * dt)
    gridded_2 = event.to_time_gridded(dt=dt)
    assert np.array_equal(gridded_1.data, gridded_2.data)
    assert gridded_1.dt == gridded_2.dt


def test_valued_conversion():
    idx = np.array([[0, 0, 1, 1]])
    time = np.array([[0.05, 0.15, 0.1, 0.2]])
    values = np.array([[1.0, 4.0, 2.0, 3.0]])
    n_neurons = 2
    dt = 0.01
    t_max = 0.3
    valued_event = nir.ValuedEventData(idx, time, n_neurons, t_max, values)
    gridded = valued_event.to_time_gridded(dt=dt)  # noqa: F841
