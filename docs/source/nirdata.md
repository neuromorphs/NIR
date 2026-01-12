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
NIRGraphData contains one batch of output data for the entire graph, organized as a dictionary with one entry per node as `NIRNodeData` objects. The dictionary keys must correspond to the node keys in the associated `NIRGraph` to ensure data-graph consistency validation.

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

## Observable data types

NIRData supports a flexible design to accommodate arbitrary observables through two complementary approaches used across the neuromorphic community:

* **Discrete time-gridded format**: Divides the time axis into discrete steps (e.g., Norse, snnTorch)
* **Continuous event-based format**: Represents events as tuples of neuron indices and precise spike times (e.g., jaxsnn)

The format provides internal conversion functions between these representations via the `TimeGriddedData` and `EventData` classes. Conversion to and from NIRData itself intended to be implemented by the respective software/hardware platforms. As an example, these conversions are is implemented for two software frameworks: The event-based conversion is provided in [jaxsnn](https://github.com/electronicvisions/jaxsnn/blob/main/src/pyjaxsnn/jaxsnn/event/to_nir_data.py), while the time-gridded one is implemented in [hxtorch](https://github.com/electronicvisions/hxtorch/blob/master/src/pyhxtorch/hxtorch/spiking/utils/to_nir_data.py). In both cases, the framework supplies complementary `from_nir_data.py` and a `to_nir_data.py` modules.

### TimeGriddedData

The `TimeGriddedData` format represents time as a sequence of discrete steps. For spike-based data, each entry is binary (0 or 1), indicating whether a neuron emitted a spike at a given time step. This encoding is used by frameworks such as Norse and snnTorch.

Beyond binary spikes, the same structure can also accommodate continuous-valued signals—for example, storing membrane potential measurements sampled at fixed temporal intervals.

The parameters include:
* `data`: an `np.ndarray` of shape (n_samples, n_time_steps, n_neurons) and
* `dt`: the size of the time step, representing the temporal resolution.

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

### Conversion Between Event-Based and Time-Gridded Representations (Binary Data)

The following conversions apply to *valueless (binary)* event data, where events indicate only the presence or absence of an event.

**EventData → TimeGriddedData**  
Converting from `EventData` to `TimeGriddedData` is largely straightforward. Each event time is assigned to a discrete time step according to

\[
\text{step} = \left\lfloor \frac{\text{time}}{dt} \right\rfloor .
\]

The corresponding entry in the time-gridded tensor is set to 1.

**TimeGriddedData → EventData**  
The inverse conversion is less rigid, as it requires a convention for placing an event within a discrete time bin. To address this, a parameter $\text{time\_shift} \in [0, dt)$ is introduced. This parameter specifies the offset within each time step at which the event time is placed.

### ValuedEventData Conversion Support

At present, only a conversion from `ValuedEventData` to `TimeGriddedData` is supported. In contrast to the conversion of binary data, the resulting tensor stores the event values at the corresponding time steps rather than `'1'`s.
