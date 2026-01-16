import numpy as np
import nir


def test_generate_time_gridded_data():
    spikes = np.random.randint(0, 2, size=(10, 10, 10)).astype(bool)
    dt = 0.1
    gridded = nir.TimeGriddedData(spikes, dt)
    node = nir.NIRNodeData({"spikes": gridded})
    graph = nir.NIRGraphData({"node": node})  # noqa: F841
    assert np.allclose(graph.nodes["node"].observables["spikes"].data, spikes)
    assert graph.nodes["node"].observables["spikes"].dt == dt


def test_generate_event_data():
    idx = np.random.randint(0, 10, size=(10, 20))
    time = np.sort(np.random.rand(10, 20), axis=1)
    n_neurons = 10
    t_max = 1.0
    event = nir.EventData(idx, time, n_neurons, t_max)
    node = nir.NIRNodeData({"spikes": event})
    graph = nir.NIRGraphData({"node": node})  # noqa: F841
    assert np.allclose(graph.nodes["node"].observables["spikes"].idx, idx)
    assert np.allclose(graph.nodes["node"].observables["spikes"].time, time)
    assert graph.nodes["node"].observables["spikes"].n_neurons == n_neurons
    assert graph.nodes["node"].observables["spikes"].t_max == t_max


def test_generate_valued_event_data():
    idx = np.random.randint(0, 10, size=(10, 20))
    time = np.sort(np.random.rand(10, 20), axis=1)
    value = np.random.rand(10, 20)
    n_neurons = 10
    t_max = 1.0
    valued_event = nir.ValuedEventData(idx, time, n_neurons, t_max, value)
    node = nir.NIRNodeData({"current": valued_event})
    graph = nir.NIRGraphData({"node": node})  # noqa: F841
    assert np.allclose(graph.nodes["node"].observables["current"].idx, idx)
    assert np.allclose(graph.nodes["node"].observables["current"].time, time)
    assert np.allclose(graph.nodes["node"].observables["current"].value, value)
    assert graph.nodes["node"].observables["current"].n_neurons == n_neurons
    assert graph.nodes["node"].observables["current"].t_max == t_max


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
    value = np.array([[1.0, 4.0, 2.0, 3.0]])
    n_neurons = 2
    dt = 0.01
    t_max = 0.3
    valued_event = nir.ValuedEventData(idx, time, n_neurons, t_max, value)
    gridded = valued_event.to_time_gridded(dt=dt)  # noqa: F841
    expected = np.zeros((1, 30, 2))
    expected[0, 5, 0] = 1
    expected[0, 10, 1] = 2
    expected[0, 15, 0] = 4
    expected[0, 20, 1] = 3
    assert np.array_equal(gridded.data, expected)


def test_check_nodes():
    graph = nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type=np.array([5])),
            "linear1": nir.Linear(weight=np.random.rand(10, 5)),
            "lif1": nir.CubaLIF(
                tau_mem=np.array([0.02] * 10),
                tau_syn=np.array([0.005] * 10),
                r=np.array([1.0] * 10),
                v_leak=np.array([0.1] * 10),
                v_reset=np.array([0.0] * 10),
                v_threshold=np.array([1.0] * 10),
            ),
            "linear2": nir.Linear(weight=np.random.rand(5, 10)),
            "lif2": nir.CubaLIF(
                tau_mem=np.array([0.02] * 5),
                tau_syn=np.array([0.005] * 5),
                r=np.array([1.0] * 5),
                v_leak=np.array([0.1] * 5),
                v_reset=np.array([0.0] * 5),
                v_threshold=np.array([1.0] * 5),
            ),
            "output": nir.Output(output_type=np.array([5])),
        },
        edges=[
            ("input", "linear1"),
            ("linear1", "lif1"),
            ("lif1", "linear2"),
            ("linear2", "lif2"),
            ("lif2", "output"),
        ],
    )
    graph_data = nir.NIRGraphData(
        nodes={
            "lif1": nir.NIRNodeData(
                observables={
                    "spikes": nir.TimeGriddedData(
                        data=np.random.randint(0, 2, (4, 20, 10)).astype(bool), dt=0.001
                    )
                }
            ),
            "lif2": nir.NIRNodeData(
                observables={
                    "spikes": nir.EventData(
                        idx=np.random.randint(0, 5, (3, 5)),
                        time=np.random.rand(3, 5) * 0.1,
                        n_neurons=5,
                        t_max=0.1,
                    )
                }
            ),
        }
    )
    graph_data.check_nodes(graph)
