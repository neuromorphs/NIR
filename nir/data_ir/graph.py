from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Union
import numpy as np
from nir import NIRGraph, NIRNode


@dataclass
class TimeGriddedData:
    """
    Either boolean entries indicate whether a binary event is present at a
    particular time step, or a real-valued signal provides the measurement
    of a quantity (e.g. the membrane potential).

    Arguments
    ---------
    data : np.ndarray, shape (n_samples, n_time_steps, n_neurons)
        Input data. For binary data the dtype should be bool.
    dt: float
        Time step size.
    """

    data: np.ndarray
    dt: float  # pylint: disable=invalid-name

    def __post_init__(self):
        if not isinstance(self.data, np.ndarray) or self.data.ndim != 3:
            raise ValueError(
                "Data must be of shape (n_samples, n_time_steps, n_neurons)"
                "and of type np.ndarray"
            )

    @property
    def shape(self):
        return self.data.shape

    @property
    def n_samples(self):
        return self.data.shape[0]

    @property
    def n_time_steps(self):
        return self.data.shape[1]

    @property
    def n_neurons(self):
        return self.data.shape[2]

    @property
    def t_max(self):
        return self.n_time_steps * self.dt

    def to_event(self, n_events: int, time_shift: float = 0.0) -> EventData:
        """
        Arguments
        ---------
        n_spikes : int
            Maximum number of events stored for each neuron.
        time_shift : float, optional
            Shift the event times by this value from the beginning of each time
            step. Must be in interval [0, dt). Default is 0.0.
        """

        if not self.data.dtype == bool:
            raise ValueError("Data must be boolean to convert to EventData.")
        idx = np.full((self.n_samples, n_events), -1)
        time = np.full((self.n_samples, n_events), np.inf)

        if time_shift < 0 or time_shift >= self.dt:
            raise ValueError("time_shift must be in interval [0, dt)")
        for sample in range(self.n_samples):
            time_step, neuron = np.where(self.data[sample])

            order = np.argsort(time_step)  # sort events by time
            time_step = time_step[order]
            neuron = neuron[order]

            num_events = min(len(time_step), n_events)
            idx[sample, :num_events] = neuron[:num_events]
            time[sample, :num_events] = time_step[:num_events] * self.dt + time_shift

        return EventData(idx, time, self.n_neurons, self.t_max)


@dataclass
class EventData:
    """
    Event-based data represented as a list of event indices and their
    corresponding timestamps. Each event is discrete and carries no magnitude;
    it is defined solely by its occurrence at a certain time.

    Arguments
    ---------
    idx : np.ndarray[int], shape (n_samples, n_events)
        Event indices. If there is no event, the index is `-1`.
    time : np.ndarray[float], shape (n_samples, n_events)
        Event times. If there is no event, the time is `np.inf`.
    n_neurons : int
        Total number of neurons in the layer.
    t_max : float
        Maximum time of the recording.
    """

    idx: np.ndarray
    time: np.ndarray
    n_neurons: int
    t_max: float

    def __post_init__(self):
        if self.idx.shape != self.time.shape:
            raise ValueError("idx and time must have the same shape")

    @property
    def shape(self):
        return self.idx.shape

    @property
    def n_samples(self):
        return self.idx.shape[0]

    def to_time_gridded(
        self, dt: float  # pylint: disable=invalid-name
    ) -> TimeGriddedData:
        """
        Arguments
        ---------
        dt : float
            Time step size.
        """
        n_time_steps = int(self.t_max / dt)
        discrete_data = np.zeros(
            (self.n_samples, n_time_steps, self.n_neurons), dtype=bool
        )

        for sample in range(self.n_samples):
            valid_spikes = self.idx[sample] != -1
            valid_times = self.time[sample][valid_spikes]
            steps = np.floor((valid_times / dt)).astype(int)
            neurons = self.idx[sample][valid_spikes]
            discrete_data[sample, steps, neurons] = True
        return TimeGriddedData(discrete_data, dt)


@dataclass
class ValuedEventData(EventData):
    """
    Valued event-based data as a list of event indices, event times and event
    values.

    Parameters
    ----------
    idx : np.ndarray[int], shape (n_samples, n_events)
        Event indices. If there is no event, the index is `-1`.
    time : np.ndarray[float], shape (n_samples, n_events)
        Event times. If there is no event, the time is `np.inf`.
    value : np.ndarray[float], shape (n_samples, n_events)
        Event values.
    n_neurons : int
        Total number of neurons in the layer.
    t_max : float
        Maximum time of the recording.
    """

    value: np.ndarray

    def __post_init__(self):
        if self.idx.shape != self.time.shape or self.idx.shape != self.value.shape:
            raise ValueError("idx, time and value must have the same shape")

    def to_time_gridded(
        self,
        dt: float,  # pylint: disable=invalid-name
    ) -> TimeGriddedData:
        """
        Currently, the values are assigned directly to the corresponding time
        steps without any interpolation.

        Parameters
        ----------
        dt : float
            Time step size.
        """
        n_samples = self.n_samples
        n_time_steps = int(self.t_max / dt)
        discrete_data = np.zeros((n_samples, n_time_steps, self.n_neurons), dtype=float)

        for sample in range(n_samples):
            valid_spikes = self.idx[sample] != -1
            valid_times = self.time[sample][valid_spikes]
            steps = np.floor((valid_times / dt)).astype(int)
            neurons = self.idx[sample][valid_spikes]
            value = self.value[sample][valid_spikes]
            discrete_data[sample, steps, neurons] = value

        return TimeGriddedData(discrete_data, dt)


@dataclass
class NIRNodeData:
    """
    Dictionary of EventData or TimeGriddedData where each entry represents an
    observable (e.g., spikes, voltages) of a corresponding NIRNode

    Arguments
    ---------
    observables : Dict[str, Union[EventData, TimeGriddedData]]
        Dictionary of observables for a NIRNode.
    """

    observables: Dict[str, Union[EventData, TimeGriddedData]]

    def __post_init__(self):
        if not isinstance(self.observables, dict):
            raise TypeError(
                "observables must be a dictionary of EventData or TimeGriddedData"
            )

    def check_observables(self, node: NIRNode):
        """
        Check that the shapes of the observables match the node's output shapes
        """
        output_shape = node.output_type["output"]
        if not all(
            obs.n_neurons == output_shape for obs in self.observables.values()
        ):
            return False
        return True


@dataclass
class NIRGraphData:
    """
    Dictionary of NIRNodeData where each entry represents a NIRNode of a
    corresponding NIRGraph with its observables.

    Arguments
    ---------
    nodes : Dict[str, Union[NIRGraphData, NIRNodeData]]
        Dictionary of NIRNodeData or NIRGraphData for a NIRGraph.
    """

    nodes: Dict[str, Union["NIRGraphData", NIRNodeData]]

    def __post_init__(self):
        if not isinstance(self.nodes, dict):
            raise TypeError("nodes must be a dictionary of NIRNodeData or NIRGraphData")

    def check_nodes(self, graph: NIRGraph):
        """
        Check if the nodes in NIRData are a subset of the nodes in the NIRGraph
        """

        for key, node in self.nodes.items():
            if key not in graph.nodes:
                raise KeyError(f"Node {key} not found in the NIRGraph")
            graph_node = graph.nodes[key]
            if isinstance(node, NIRGraphData):
                if not isinstance(graph_node, NIRGraph):
                    raise TypeError(f"Node {key} is not a NIRGraph in the NIRGraph")
                node.check_nodes(graph_node)
            elif isinstance(node, NIRNodeData):
                if not isinstance(graph_node, NIRNode):
                    raise TypeError(f"Node {key} is not a NIRNode in the NIRGraph")
                if not node.check_observables(graph_node):
                    raise ValueError(f"Observables for node {key} do not match the NIRNode")