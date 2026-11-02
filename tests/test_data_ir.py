import numpy as np

import nir


def test_generate_time_gridded_data():
    spikes = np.random.randint(0, 2, size=(10, 10, 10)).astype(bool)
    dt = 0.1
    gridded = nir.TimeGriddedData(spikes, dt)
    node = nir.NIRNodeData({"spikes": gridded})
    graph = nir.NIRGraphData({"node": node})
    assert np.allclose(graph.nodes["node"].observables["spikes"].data, spikes)
    assert graph.nodes["node"].observables["spikes"].dt == dt


def test_generate_event_data():
    idx = np.random.randint(0, 10, size=(10, 20))
    time = np.sort(np.random.rand(10, 20), axis=1)
    n_neurons = 10
    t_max = 1.0
    event = nir.EventData(idx, time, n_neurons, t_max)
    node = nir.NIRNodeData({"spikes": event})
    graph = nir.NIRGraphData({"node": node})
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
    graph = nir.NIRGraphData({"node": node})
    assert np.allclose(graph.nodes["node"].observables["current"].idx, idx)
    assert np.allclose(graph.nodes["node"].observables["current"].time, time)
    assert np.allclose(graph.nodes["node"].observables["current"].value, value)
    assert graph.nodes["node"].observables["current"].n_neurons == n_neurons
    assert graph.nodes["node"].observables["current"].t_max == t_max


def test_binary_conversion():
    for dynamic_before_transition in [True, False]:
        spikes = np.random.randint(0, 2, size=(5, 10, 10)).astype(bool)
        dt = 0.1
        gridded_1 = nir.TimeGriddedData(
            spikes, dt, dynamic_before_transition=dynamic_before_transition
        )
        event = gridded_1.get_event(n_events=100)
        gridded_2 = event.get_time_gridded(
            dt, dynamic_before_transition=dynamic_before_transition
        )
        assert np.array_equal(gridded_1.data, gridded_2.data)
        assert gridded_1.dt == gridded_2.dt
        assert (
            gridded_1.dynamic_before_transition == gridded_2.dynamic_before_transition
        )
        assert gridded_1.dimension_order == gridded_2.dimension_order


def test_valued_conversion():
    idx = np.array([[0, 0, 1, 1]])
    time = np.array([[0.05, 0.15, 0.1, 0.2]])
    value = np.array([[1.0, 4.0, 2.0, 3.0]])
    n_neurons = 2
    dt = 0.01
    t_max = 0.3
    valued_event = nir.ValuedEventData(idx, time, n_neurons, t_max, value)
    gridded = valued_event.get_time_gridded(dt=dt)
    expected = np.zeros((30, 1, 2))
    expected[4, 0, 0] = 1
    expected[9, 0, 1] = 2
    expected[14, 0, 0] = 4
    expected[19, 0, 1] = 3
    assert np.array_equal(gridded.data, expected)
    assert gridded.dt == dt
    assert gridded.dimension_order == ("time", "batch", "neuron")  # default value
    assert gridded.dynamic_before_transition == True  # default value


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


def test_setter_getter():
    spikes = np.random.randint(0, 2, size=(10, 10, 10)).astype(bool)
    dt = 0.1
    time_gridded_data = nir.TimeGriddedData(spikes, dt)
    node_data = nir.NIRNodeData({"spikes": time_gridded_data})
    graph_data = nir.NIRGraphData({"node": node_data})
    assert np.allclose(graph_data["node"]["spikes"][:], spikes)
    new_spikes = np.random.randint(0, 2, size=(10, 10, 10)).astype(bool)
    graph_data["node"]["spikes"] = nir.TimeGriddedData(new_spikes, dt)
    assert np.array_equal(graph_data["node"]["spikes"][:], new_spikes)


def test_events_outside_time_grid_are_dropped():
    event = nir.EventData(
        idx=np.array([[0, 1, 0]]),
        time=np.array([[0.0, 0.1, 0.5]]),
        n_neurons=2,
        t_max=0.3,
    )
    gridded = event.get_time_gridded(dt=0.1)
    expected = np.zeros((3, 1, 2), dtype=bool)
    expected[0, 0, 1] = True
    assert np.array_equal(gridded.data, expected)

    valued = nir.ValuedEventData(
        idx=event.idx,
        time=event.time,
        n_neurons=2,
        t_max=0.3,
        value=np.array([[1.0, 2.0, 3.0]]),
    )
    gridded = valued.get_time_gridded(dt=0.1)
    expected = np.zeros((3, 1, 2))
    expected[0, 0, 1] = 2.0
    assert np.array_equal(gridded.data, expected)


def test_change_dynamic_before_transition():
    # with dynamic_before_transition, events occur one time step earlier
    spikes = np.zeros((10, 1, 1), dtype=bool)
    spikes[5, 0, 0] = True
    for dynamic_before_transition, expected_step in [(True, 6), (False, 4)]:
        gridded = nir.TimeGriddedData(
            spikes, dt=1.0, dynamic_before_transition=dynamic_before_transition
        )
        target = not dynamic_before_transition
        toggled = gridded.get_time_gridded(1.0, dynamic_before_transition=target)
        via_events = gridded.get_event().get_time_gridded(
            1.0, dynamic_before_transition=target
        )
        assert toggled.dynamic_before_transition == target
        assert np.argmax(toggled.data[:, 0, 0]) == expected_step
        assert np.array_equal(toggled.data, via_events.data)

    # random data with a non-default dimension order, both directions
    spikes = np.random.rand(20, 3, 5) < 0.2
    spikes[0] = spikes[-1] = False  # keep all events on the grid after shifting
    for dynamic_before_transition in [True, False]:
        gridded = nir.TimeGriddedData(
            spikes.transpose(1, 2, 0),
            dt=0.1,
            dimension_order=("batch", "neuron", "time"),
            dynamic_before_transition=dynamic_before_transition,
        )
        target = not dynamic_before_transition
        toggled = gridded.get_time_gridded(
            0.1, ("batch", "neuron", "time"), dynamic_before_transition=target
        )
        via_events = gridded.get_event().get_time_gridded(
            0.1, ("batch", "neuron", "time"), dynamic_before_transition=target
        )
        assert np.array_equal(toggled.data, via_events.data)
        assert toggled.data.sum() == spikes.sum()
