import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

import nir


@dataclass
class StepCurrent(object):
    times: []
    amplitudes: []

    def __post_init__(self):
        assert len(self.times) == len(self.amplitudes)
        assert all(
            t1 <= t2 for t1, t2 in zip(self.times, self.times[1:])
        ), "times are not sorted"

    def size(self):
        return len(self.times)


class NeuronRecord(object):
    """Class for storing spike times and voltage trace of a neuron."""

    def __init__(self):
        self.times = []
        self.voltages = []
        self.spikes = []

    def add_voltage(self, time: float, voltage: float):
        self.times.append(time)
        self.voltages.append(voltage)

    def add_spike(self, time: float):
        self.spikes.append(time)


@dataclass
class LIFParams(object):
    tau: float  # Time constant
    r: float  # Resistance
    v_leak: float  # Leak voltage
    v_threshold: float  # Firing threshold


@dataclass
class LIFState(object):
    v: float  # voltage


def next_spike_time(params, current):
    pass


class ExactLIFNeuron(object):
    """Exact implementation of NIR LIF neuron.

    This class holds the parameters and the state of LIF neuron. The class itself has no
    idea about the time the current state refers.
    """

    def __init__(self, params: LIFParams):
        self.params = params
        self.state = LIFState(v=0.0)

    def advance_by_delta_t(self, i_input: float, delta_t: float):
        """Advance neuron by delta_t for constant input current.

        This uses the exact solution of the time course of the membrane
        potential for a constant input current. See Section 4.4.2 of the paper.

        The neuron must not cross the threshold within `delta_t`.

        Args:
            i_input: constant input current
            delta_t: delta time to advance
        """
        old_v = self.state.v
        p = self.params
        exp_factor = math.exp(-delta_t / p.tau)
        new_v = p.v_leak + p.r * i_input * (1 - exp_factor) + old_v * exp_factor
        self.state.v = new_v

    def calc_next_spike_time(self, i_input: float):
        """Calculate the next spike time of the neuron.

        returns the time to the next spike if there will be a spike in the
        future, else it will return infinity.

        How it works:
        Let `f(t) be the exact solution of the time course of the membrane
        voltage with constant input current, as in Section 4.4.2 of the paper.
        Then we try to find a positive `t` that solves equation:
            f(t) == v_threshold
        For this,
            1. the `exp(-t/tau) is separated from the rest of the variables (if possible)
            2. `log()` is applied to both sides of the equation (if possible)
            3. solve the resulting equation for `t`.

        If a solution `t=t_spike` exists with a posivite value, then it
        provides the time to the next spike. If no such solution exists, the
        neuron will not spike in the future for this input current.

        Args:
            i_input: constant input current

        Returns:
            `np.inf` if the neuron will not spike in the future,
            else a positive float representing the time to the next spike.
        """
        p = self.params
        if (self.state.v - p.r * i_input) == 0.0:  # avoid DivideByZero in next step
            return math.inf
        log_arg = (p.v_threshold - p.v_leak - p.r * i_input) / (
            self.state.v - p.r * i_input
        )
        if log_arg > 0.0:
            t_spike = -1.0 * p.tau * math.log(log_arg)
            if t_spike >= 0.0:
                return t_spike
            else:
                return math.inf
        else:
            return math.inf

    def apply_reset(self):
        """Apply reset by subtraction."""
        self.state.v = self.state.v - self.params.v_threshold


def run_event_based_simulation(
    neuron: ExactLIFNeuron, inputs: StepCurrent, record_dt: float, duration: float
) -> NeuronRecord:
    """Run event-based exact simulation of LIF neuron.

    Args:
        neuron: exact LIF Neuron implementation
        inputs: step current changing over time
        record_dt: interval for recording voltages in seconds
        duration: simulation time in seconds

    Returns:
        the records of the simulation: voltages and spikes.
    """
    # states for the simulation
    current_time = 0.0
    current_input_index = 0
    current_input_amplitude = 0.0

    # there are 3 event types to be considered:
    #   1. the neuron spikes
    #   2. the voltage shall be recorded
    #   3. the input current changes
    # The following variables keep track for the next occurence of each type.
    # If there is no such event in the future, e.g. no spike, the variable
    # takes the value `np.inf`
    next_spike_time = math.inf
    next_record_time = record_dt
    next_input_change_time = inputs.times[current_input_index]

    # voltage and spike recorder
    record = NeuronRecord()

    # run simulation
    while current_time <= duration:
        # Which type of event comes next?
        # The order of events in the `next_event_list` defines the priority by
        # which the events are handled in case one or several happen at the
        # same time.
        next_event_list = [next_spike_time, next_record_time, next_input_change_time]
        next_event_index = np.argmin(next_event_list)
        next_time = next_event_list[next_event_index]

        delta_t = next_time - current_time

        if next_event_index == 0:  # spike
            # 1. advance_by_delta_t
            neuron.advance_by_delta_t(current_input_amplitude, delta_t)
            # 2. reset voltage
            neuron.apply_reset()
            # 3. record spike time
            record.add_spike(next_time)
            # 4. compute next spike time
            time_to_next_spike = neuron.calc_next_spike_time(current_input_amplitude)
            next_spike_time = next_time + time_to_next_spike

        elif next_event_index == 1:  # record
            # 1. advance_by_delta_t
            neuron.advance_by_delta_t(current_input_amplitude, delta_t)
            # 2. record voltage
            record.add_voltage(next_time, neuron.state.v)
            # 3. compute next record time
            next_record_time = next_record_time + record_dt

        elif next_event_index == 2:  # input change
            # 1. advance_by_delta_t with old input
            neuron.advance_by_delta_t(current_input_amplitude, delta_t)
            # 2. update input
            current_input_amplitude = inputs.amplitudes[current_input_index]
            current_input_index += 1
            # 3. compute next input change time
            try:
                next_input_change_time = inputs.times[current_input_index]
            except IndexError:
                next_input_change_time = math.inf
            # 4. compute next spike time
            time_to_next_spike = neuron.calc_next_spike_time(current_input_amplitude)
            next_spike_time = next_time + time_to_next_spike
        else:
            raise Exception()

        current_time = next_time

    return record


def test():
    """Simple test for exact event-based LIF simulation."""
    # global settings
    record_dt: float = 0.0001  # recording time step
    duration = 0.1

    # "StepCurrent" as input
    times = [0.0, 0.01, 0.02, 0.05]
    amplitudes = [0.5, 0.1, 0.3, 1.05]

    inputs = StepCurrent(times, amplitudes)
    params = LIFParams(tau=0.001, r=1.0, v_leak=0.0, v_threshold=1.0)
    neuron = ExactLIFNeuron(params)

    record = run_event_based_simulation(neuron, inputs, record_dt, duration)
    # print(record.times)
    plt.plot(record.times, record.voltages)
    print(record.spikes)
    plt.show()


def lif_nir():
    """Run LIF experiment from NIR paper."""

    # load NIR model
    nir_model = nir.read("lif_norse.nir")

    # some checks to be sure we can use the input and neglect the Affine
    affine_node = nir_model.nodes["0"]
    assert affine_node.weight[0][0] == 1.0
    assert affine_node.bias[0] == 0.0

    # global settings
    dt = 0.0001
    record_dt: float = dt  # recording time step

    # input data
    # fmt: off
    d0 = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
            0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # fmt: on
    d = []
    for t in d0:
        d.append(t)
        for i in range(9):
            d.append(0)
    duration = len(d) * dt

    # convert to # "StepCurrent" as input for exact simulation
    times = np.arange(0.0, duration, dt)
    amplitudes = np.array(d, dtype=np.float32)

    inputs = StepCurrent(times, amplitudes)

    lif_node = nir_model.nodes["1"]
    params = LIFParams(
        tau=lif_node.tau[0],
        r=lif_node.r[0],
        v_leak=lif_node.v_leak[0],
        v_threshold=lif_node.v_threshold[0],
    )
    neuron = ExactLIFNeuron(params)

    record = run_event_based_simulation(neuron, inputs, record_dt, duration)

    plt.figure(figsize=(20, 4))
    plt.plot(record.times, record.voltages)
    plt.scatter(
        record.spikes, np.ones_like(record.spikes) * 0.1, marker="|", c="red", s=400
    )
    plt.xlabel("time [s]")
    plt.ylabel("voltage [V]")
    plt.show()


if __name__ == "__main__":
    lif_nir()
