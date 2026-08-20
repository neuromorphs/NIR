from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Union
import warnings
import numpy as np
from nir.ir import NIRGraph, NIRNode


@dataclass
class ObservableData:
    """
    Base class for observable data in SNNs.
    """

    def get_event(self, n_events: int | None) -> EventData:
        pass

    def get_time_gridded(
        self,
        dt: float,
        dimension_order: tuple = ("time", "batch", "neuron"),
        dynamic_before_transition: bool = True,
    ) -> TimeGriddedData:
        pass


@dataclass
class TimeGriddedData(ObservableData):
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
    dimension_order: tuple, optional
        The order of dimensions in the spike tensors. Defaults to ('time',
        'batch', 'neuron') for (time, batch, neurons).
    dynamic_before_transition: bool, optional
        If True, it is assumed that the framework evolves the state (e.g.,
        membrane potential) of the neurons before checking if the threshold has
        been crossed and generating an event (transition). If False, the state
        is updated after the event generation.
    """

    data: np.ndarray
    dt: float  # pylint: disable=invalid-name
    dimension_order: tuple = ("time", "batch", "neuron")
    dynamic_before_transition: bool = True

    def __post_init__(self):
        if not isinstance(self.data, np.ndarray):
            raise TypeError("Data must be a numpy array")
        if self.data.ndim != 3:
            raise ValueError("Data must be a 3D array")
        if (
            "time" not in self.dimension_order
            or "batch" not in self.dimension_order
            or "neuron" not in self.dimension_order
        ):
            raise ValueError("dimension_order must contain 'time', 'batch', and 'neuron'")

    def _view_as(self, order):
        if order == self.dimension_order:
            return self.data
        perm = tuple(self.dimension_order.index(dim) for dim in order)
        return self.data.transpose(perm)

    def __getitem__(self, idx, out_order=("time", "batch", "neuron")):
        """
        Get a slice of the data, with the option to specify the output
        dimension order. The `idx` refers to the reordered data.
        """
        return self._view_as(out_order)[idx]

    def __setitem__(self, idx, value, in_order=("time", "batch", "neuron")):
        """
        Set a slice of the data, given as `value` in `in_order` dimension order.
        """
        self._view_as(in_order)[idx] = value

    @property
    def shape(self):
        return self.data.shape

    @property
    def n_samples(self):
        return self.shape[self.dimension_order.index("batch")]

    @property
    def n_time_steps(self):
        return self.shape[self.dimension_order.index("time")]

    @property
    def n_neurons(self):
        return self.shape[self.dimension_order.index("neuron")]

    @property
    def t_max(self):
        return self.n_time_steps * self.dt

    def toggle_dynamic_before_transition(self):
        """
        Toggle the dynamic_before_transition flag and update the data
        accordingly.
        """
        time_axis = self.dimension_order.index("time")
        # if dynamic_before_transition is True, shift events back by one time step
        if self.dynamic_before_transition:
            self.data = np.roll(self.data, shift=-1, axis=time_axis)
            # build an indexer equivalent to [:, -1, :] but for the correct axis
            idx = [slice(None)] * self.data.ndim
            idx[time_axis] = -1
            self.data[tuple(idx)] = False
        else:
            self.data = np.roll(self.data, shift=1, axis=time_axis)
            # build an indexer equivalent to [:, 0, :] but for the correct axis
            idx = [slice(None)] * self.data.ndim
            idx[time_axis] = 0
            self.data[tuple(idx)] = False

        self.dynamic_before_transition = not self.dynamic_before_transition

    def get_event(self, n_events: int | None = None) -> EventData:
        """
        Convert the time-gridded data to event-based data, where each neuron
        can have at most `n_events` events. If a neuron has more than
        `n_events`, the earliest events are kept and the rest are dropped.

        Arguments
        ---------
        n_spikes : int
            Maximum number of events stored for each neuron.
        """
        if n_events == None:
            n_events = self.n_time_steps
        if not self.data.dtype == bool:
            raise ValueError("Data must be boolean to convert to EventData.")
        idx = np.full((self.n_samples, n_events), -1)
        time = np.full((self.n_samples, n_events), np.inf)

        for sample in range(self.n_samples):
            sample_idx = [slice(None)] * self.data.ndim
            sample_idx[self.dimension_order.index("batch")] = sample
            time_step, neuron = np.where(self.data[tuple(sample_idx)])

            order = np.argsort(time_step)  # sort events by time
            time_step = time_step[order]
            neuron = neuron[order]

            num_events = min(len(time_step), n_events)
            idx[sample, :num_events] = neuron[:num_events]
            time[sample, :num_events] = (
                time_step[:num_events] + self.dynamic_before_transition
            ) * self.dt

        return EventData(idx, time, self.n_neurons, self.t_max)

    def get_time_gridded(
        self,
        dt: float,
        dimension_order: tuple = ("time", "batch", "neuron"),
        dynamic_before_transition: bool = True,
    ) -> TimeGriddedData:
        """
        Return a new TimeGriddedData object with the specified dt and
        dynamic_before_transition flag. If the current object already has the
        desired dt and dynamic_before_transition, return self.
        """

        if self.dt != dt:
            event_data = self.get_event()
            return event_data.get_time_gridded(
                dt=dt,
                dimension_order=dimension_order,
                dynamic_before_transition=dynamic_before_transition
            )

        if self.dimension_order == dimension_order:
            return self
        else:
            new_data = TimeGriddedData(
                data=self._view_as(dimension_order),
                dt=dt,
                dimension_order=self.dimension_order,
                dynamic_before_transition=self.dynamic_before_transition,
            )
            if self.dynamic_before_transition != dynamic_before_transition:
                new_data.toggle_dynamic_before_transition()
            return new_data


@dataclass
class EventData(ObservableData):
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

    def get_event(self, n_events: int | None) -> EventData:
        """
        Return a new EventData object with at most `n_events` events per sample.
        If a sample has more than `n_events`, the earliest events are kept and
        the rest are dropped.

        Arguments
        ---------
        n_events : int
            Maximum number of events stored for each sample. If None, return all events.
        """
        if n_events is None or n_events >= self.idx.shape[1]:
            return self

        new_idx = np.full((self.n_samples, n_events), -1)
        new_time = np.full((self.n_samples, n_events), np.inf)

        for sample in range(self.n_samples):
            valid_events = self.idx[sample] != -1
            valid_times = self.time[sample][valid_events]
            valid_indices = self.idx[sample][valid_events]

            num_events = min(len(valid_times), n_events)
            if num_events > 0:
                order = np.argsort(valid_times)  # sort events by time
                new_idx[sample, :num_events] = valid_indices[order][:num_events]
                new_time[sample, :num_events] = valid_times[order][:num_events]

        return EventData(new_idx, new_time, self.n_neurons, self.t_max)

    def get_time_gridded(
        self,
        dt: float,
        dimension_order: tuple = ("time", "batch", "neuron"),
        dynamic_before_transition: bool = True,  # pylint: disable=invalid-name
    ) -> TimeGriddedData:
        """
        Arguments
        ---------
        dt : float
            Time step size.
        dynamic_before_transition : bool, optional
            If True, the membrane potential is updated before checking if the
            threshold has been crossed and generating an event (transition). If
            False, the state is updated after the event generation. Default is
            True.
        """
        n_time_steps = round(self.t_max / dt)
        discrete_data = np.zeros((n_time_steps, self.n_samples, self.n_neurons), dtype=bool)
        for sample in range(self.n_samples):
            valid_spikes = self.idx[sample] != -1
            valid_times = self.time[sample][valid_spikes]
            eps = dt * 1e-10  # small epsilon to avoid floating point issues
            steps = np.ceil(valid_times / dt - eps).astype(int) - dynamic_before_transition
            neurons = self.idx[sample][valid_spikes]
            mask = steps < n_time_steps
            if np.any(mask):
                steps, neurons = steps[mask], neurons[mask]
                warnings.warn(
                    "Some events got dropped because they occur after the "
                    "maximum time of the recording."
                )

            discrete_data[steps, sample, neurons] = True
        if dimension_order != ("time", "batch", "neuron"):
            perm = tuple(("time", "batch", "neuron").index(dim) for dim in dimension_order)
            discrete_data = discrete_data.transpose(perm)
        return TimeGriddedData(
            data=discrete_data, dt=dt, dynamic_before_transition=dynamic_before_transition
        )


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

    def get_event(self, n_events: int | None) -> ValuedEventData:
        """
        Return a new ValuedEventData object with at most `n_events` events per
        sample. If a sample has more than `n_events`, the earliest events are
        kept and the rest are dropped.

        Arguments
        ---------
        n_events : int
            Maximum number of events stored for each sample. If None, return all events.
        """
        if n_events is None or n_events >= self.idx.shape[1]:
            return self

        new_idx = np.full((self.n_samples, n_events), -1)
        new_time = np.full((self.n_samples, n_events), np.inf)
        new_value = np.zeros((self.n_samples, n_events))

        for sample in range(self.n_samples):
            valid_events = self.idx[sample] != -1
            valid_times = self.time[sample][valid_events]
            valid_indices = self.idx[sample][valid_events]
            valid_values = self.value[sample][valid_events]

            num_events = min(len(valid_times), n_events)
            if num_events > 0:
                order = np.argsort(valid_times)  # sort events by time
                new_idx[sample, :num_events] = valid_indices[order][:num_events]
                new_time[sample, :num_events] = valid_times[order][:num_events]
                new_value[sample, :num_events] = valid_values[order][:num_events]

        return ValuedEventData(new_idx, new_time, self.n_neurons, self.t_max, new_value)

    def get_time_gridded(
        self,
        dt: float,  # pylint: disable=invalid-name
        dimension_order: tuple = ("time", "batch", "neuron"),
        dynamic_before_transition: bool = True,
    ) -> TimeGriddedData:
        """
        Currently, the values are assigned directly to the corresponding time
        steps without any interpolation.

        Parameters
        ----------
        dt : float
            Time step size.
        dynamic_before_transition : bool, optional
            If True, the membrane potential is updated before checking if the
            threshold has been crossed and generating an event (transition). If
            False, the state is updated after the event generation. Default is
            True.
        """
        n_samples = self.n_samples
        n_time_steps = int(self.t_max / dt)
        discrete_data = np.zeros((n_samples, n_time_steps, self.n_neurons), dtype=float)

        for sample in range(n_samples):
            valid_spikes = self.idx[sample] != -1
            valid_times = self.time[sample][valid_spikes]
            if dynamic_before_transition:
                steps = np.floor(valid_times / dt).astype(int)
            else:
                steps = np.ceil(valid_times / dt).astype(int)
            neurons = self.idx[sample][valid_spikes]
            value = self.value[sample][valid_spikes]
            discrete_data[sample, steps, neurons] = value

        return TimeGriddedData(
            data=discrete_data,
            dt=dt,
            dimension_order=dimension_order,
            dynamic_before_transition=dynamic_before_transition,
        )


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
            raise TypeError("observables must be a dictionary of EventData or TimeGriddedData")

    def __getitem__(self, idx):
        return self.observables[idx]

    def __setitem__(self, idx, val):
        self.observables[idx] = val

    def check_observables(self, node: NIRNode):
        """
        Check that the shapes of the observables match the node's output shapes
        """
        output_shape = node.output_type["output"]
        if not all(obs.n_neurons == output_shape for obs in self.observables.values()):
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

    def __getitem__(self, idx):
        return self.nodes[idx]

    def __setitem__(self, idx, val):
        self.nodes[idx] = val

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
