from typing import Dict, Union
import numpy as np
from nir import NIRGraph


class TimeGriddedData:
    """
    Time gridded data of shape (n_samples, n_time_steps, n_neurons)
    with binary entries (0 or 1) indicating whether a neuron spiked at a
    particular time step dt.

    :param data: Array of shape (n_samples, n_time_steps, n_neurons) with binary entries
    :param dt: Time step size
    """

    @property
    def shape(self):
        return self.data.shape  # pylint: disable=no-member

    def __init__(self, data: np.ndarray, dt: float):  # pylint: disable=invalid-name
        if not isinstance(data, np.ndarray) or data.ndim != 3:
            raise ValueError(
                "Data must be of shape (n_samples, n_time_steps, n_neurons)"
            )
        self.data = data
        self.dt = dt
        self.n_neurons = data.shape[2]
        self.t_max = dt * data.shape[1]

    def to_event(self, n_spikes: int, time_shift: float = 0.0):
        """
        :params n_spikes: Maximum number of spikes stored for each neuron.
        :params time_shift: Shift the spike times by this value.
                           Must be in interval [0, dt].
        """
        idx = np.full((self.data.shape[0], n_spikes), -1)
        time = np.full((self.data.shape[0], n_spikes), np.inf)

        if time_shift < 0 or time_shift > self.dt:
            raise ValueError("time_shift must be in interval [0, dt]")

        for sample in range(self.data.shape[0]):  # n_samples
            step, neuron = np.where(self.data[sample] == 1)

            sorted_indices = np.argsort(step)
            step = step[sorted_indices]
            neuron = neuron[sorted_indices]

            num_spikes = min(len(step), n_spikes)
            idx[sample, :num_spikes] = neuron[:num_spikes]
            time[sample, :num_spikes] = (step[:num_spikes] + time_shift) * self.dt

        return EventData(idx, time, self.data.shape[2], self.t_max)


class EventData:
    """
    Event-based data represented as a list of spike indices and spike times.

    :param idx: Array of shape (n_samples, n_events) with neuron indices. If
        there is no event, the index is `-1`.
    :param time: Array of shape (n_samples, n_events) with event times. If
        there is no event, the time is `np.inf`.
    :param n_neurons: Total number of neurons in the layer.
    :param t_max: Maximum time of the recording.
    """

    @property
    def shape(self):
        return self.idx.shape  # pylint: disable=no-member

    def __init__(self, idx: np.ndarray, time: np.ndarray, n_neurons: int, t_max: float):
        if idx.shape != time.shape:
            raise ValueError("idx and time must have the same shape")
        self.idx = np.array(idx, dtype=int)
        self.time = np.array(time, dtype=float)
        self.n_neurons = n_neurons
        self.t_max = t_max

    def to_time_gridded(self, dt):  # pylint: disable=invalid-name
        n_samples = self.idx.shape[0]
        n_time_steps = int(self.t_max / dt)
        discrete_data = np.zeros((n_samples, n_time_steps, self.n_neurons), dtype=int)

        for sample in range(n_samples):
            valid_spikes = self.idx[sample] != -1
            steps = (self.time[sample][valid_spikes] / dt).astype(int)
            neurons = self.idx[sample][valid_spikes]
            discrete_data[sample, steps, neurons] = 1
        return TimeGriddedData(discrete_data, dt)


class NIRNodeData:
    """
    Dictionary of EventData or TimeGriddedData where each entry represents an
    observable (e.g., spikes, voltages) of a corresponding NIRNode
    """

    observables: Dict[str, Union[EventData, TimeGriddedData]]

    def __init__(self, observables):
        if not isinstance(observables, dict):
            raise TypeError(
                "observables must be a dictionary of EventData or TimeGriddedData"
            )

        self.observables = observables

    def check_observables():  # TODO: should be called by check nodes
        return True


class NIRGraphData:
    """
    Dictionary of NIRNodeData where each entry represents a NIRNode of a
    corresponding NIRGraph with its observables.
    """

    nodes: Dict[str, Union["NIRGraphData", NIRNodeData]]

    def __init__(self, nodes: Dict[str, Union["NIRGraphData", NIRNodeData]]):
        """
        :param nodes: Dictionary of NIRNodeData or NIRGraphData
        """
        if not isinstance(nodes, dict):
            raise TypeError(
                "nodes must be a dictionary of NIRNodeData or " "NIRGraphData"
            )

        self.nodes = nodes

    def check_nodes(self, graph: NIRGraph):
        """
        Check if the nodes in NIRData are a subset of the nodes in the NIRGraph
        """

        for key in self.nodes.keys():
            if key not in graph.nodes:
                return False

        return True
