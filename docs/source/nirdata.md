# NIRData

NIRData is an extension of the existing Neuromorphic Intermediate Representation (NIR) to also represent the observable data transmitted in spiking neural networks (SNNs). Similar to NIR, NIRData works as an interface for the conversion of observables between different neuromorphic software/hardware platforms. There are two main points which motivate such a data format:

1. **Neuromorphic hardware**, that is not numerically deterministic needs special training routines, like in-the-loop training. This approach needs repeated data exchange of IO data during training.

2. **Benchmark frameworks** such as Neurobench want to supply data in a general format which can easily be converted to other frameworks' native data formats.

## General Structure
NIRData mirrors the structure of NIR by mapping conceptual components to data containers:

* `NIRGraph` → `NIRGraphData`: Graph structure maps to corresponding data container
* `NIRNode` → `NIRNodeData`: Individual nodes map to their data representations

The format supports multiple observables per node, currently supporting valueless data (e.g. spikes) as well as valued data (e.g. membrane voltage).
In contrast to the intermediate representation for SNN topologies, here the different frameworks have to identify the type of observable by the key.

### NIRGraphData
NIRGraphData contains one batch of output data for the entire graph, organized as a dictionary with one entry per node as `NIRNodeData` objects (or nested `NIRGraphData` objects for subgraphs). The link to the topology is made only through the dictionary keys, which correspond to the node keys of a `NIRGraph`. `NIRGraphData` holds no reference to the graph itself, so the same data object can be used with several graphs that share node names, and the topology IR stays free of observables.

Consistency between data and graph is checked on demand:

```python
graph_data.check_nodes(graph)
```

This raises an error if a key is not a node of `graph`, if `NIRGraphData` and `NIRNodeData` entries do not match subgraphs and nodes, respectively, or if the observables of a node do not fit it (see below).

```python
graph_data = NIRGraphData(
                nodes = {
                    "lif": lif_data,
                    ..
                }
)
```

### NIRNodeData

NIRNodeData contains all observables for a single node, organized as a dictionary where keys represent observable names and values are their corresponding data objects. Currently, the format supports valueless events (e.g., spikes) as observables as well as valued data (e.g., membrane voltage).

```python
lif_data = NIRNodeData(
                observables = {
                    "spikes": spike_data,
                    "membrane": membrane_data,
                    ..
                }
)
```

An observable does not need to cover the whole node. `check_nodes` therefore only requires that each observable has at most as many neurons as the node has outputs, `n_neurons <= prod(node.output_type["output"])`. Nodes with multidimensional outputs (e.g. `Conv2d`) are recorded with the outputs flattened into the neuron axis.

## Observable data types

NIRData supports a flexible design to accommodate arbitrary observables through two complementary approaches used across the neuromorphic community:

* **Discrete time-gridded format**: Divides the time axis into discrete steps (e.g., Norse, snnTorch)
* **Continuous event-based format**: Represents events as tuples of neuron indices and precise spike times (e.g., jaxsnn)

All observable types derive from `ObservableData` and provide two conversion methods, `get_event(n_events=None)` and `get_time_gridded(dt, dimension_order, dynamic_before_transition)`, which convert between these representations. Conversion to and from NIRData itself intended to be implemented by the respective software/hardware platforms. As an example, these conversions are is implemented for two software frameworks: The event-based conversion is provided in [jaxsnn](https://github.com/electronicvisions/jaxsnn/blob/main/src/pyjaxsnn/jaxsnn/event/utils/to_nir_data.py), while the time-gridded one is implemented in [hxtorch](https://github.com/electronicvisions/hxtorch/blob/master/src/pyhxtorch/hxtorch/spiking/utils/to_nir_data.py). In both cases, the framework supplies complementary `from_nir_data.py` and a `to_nir_data.py` modules.

### TimeGriddedData

The `TimeGriddedData` format represents time as a sequence of discrete steps. For spike-based data, each entry is binary (0 or 1), indicating whether a neuron emitted a spike at a given time step. This encoding is used by frameworks such as Norse and snnTorch.

Beyond binary spikes, the same structure can also accommodate continuous-valued signals—for example, storing membrane potential measurements sampled at fixed temporal intervals.

The parameters include:
* `data`: a 3D `np.ndarray` with a time, a batch and a neuron axis,
* `dt`: the size of the time step, representing the temporal resolution,
* `dimension_order`: the order of the axes of `data`, a permutation of `("time", "batch", "neuron")` (default), so that each framework can store its native layout without copying, and
* `dynamic_before_transition`: the update convention of the framework that produced the data (default `True`). If `True`, the state (e.g. the membrane potential) is evolved within a time step before the threshold is checked and an event is generated. If `False`, events are generated from the state at the beginning of the time step before it is evolved. This decides at which time an event in a time step occurred, see the conversion below.

### EventData

In contrast, the `EventData` format describes events as tuples consisting of the index of the neuron and the time when the event occurred. This format is commonly used in frameworks like jaxsnn. 

The parameters for EventData include:
* `idx`: an array containing the indices of the spiking neurons,
* `time`: the specific time at which the spike occurred,
* `n_neurons`: the number of neurons in the layer and
* `t_max`: the maximum time of recording.

This structured approach allows for efficient representation and conversion between different data formats used in neuromorphic computing.

### ValuedEventData

`ValuedEventData` extends the `EventData` format to support events with associated values. This is achieved by introducing a third array, `value`, which stores the value corresponding to each event.

### Conversion Between Event-Based and Time-Gridded Representations

Placing events on a time grid requires a convention for when within a time step an event happens. This is given by `dynamic_before_transition` ($d \in \{0, 1\}$ below). With $d = 1$, the event of time step $k$ is the result of the dynamics during that step, so it occurs at the end of the step. With $d = 0$, it occurs at the beginning of the step.

**TimeGriddedData → EventData**  
`get_event(n_events=None)` requires boolean data. An event in time step $k$ is placed at

\[
\text{time} = (k + d) \cdot dt .
\]

Events are sorted by time and stored in arrays of shape (n_samples, n_events). By default, `n_events` is the largest number of events in any sample, so no event is lost. If a smaller `n_events` is given, only the earliest events of each sample are kept and a warning reports how many were dropped. Calling `get_event(n_events)` on `EventData` or `ValuedEventData` truncates in the same way, or pads with empty events (index `-1`, time `inf`) if `n_events` is larger than the current number of event slots.

**EventData → TimeGriddedData**  
`get_time_gridded(dt, dimension_order, dynamic_before_transition)` creates a grid with $\text{round}(t_\text{max} / dt)$ time steps and assigns each event to the step

\[
\text{step} = \left\lceil \frac{\text{time}}{dt} \right\rceil - d ,
\]

which is the inverse of the conversion above. Events whose step lies outside the grid (e.g. after $t_\text{max}$, or at time 0 with $d = 1$) are dropped with a warning. For `EventData` the corresponding entries are set to `True`. For `ValuedEventData` the event values are stored instead, without any interpolation between events.

**TimeGriddedData → TimeGriddedData**  
`get_time_gridded` on `TimeGriddedData` returns the data in the requested `dimension_order` and shifts the events by one time step if `dynamic_before_transition` changes. If `dt` changes, binary data is converted to `EventData` and back to the new grid. Changing `dt` of real-valued data is not supported yet.

At present, there is no conversion from real-valued `TimeGriddedData` to `ValuedEventData`.
