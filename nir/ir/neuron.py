from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from .node import NIRNode


@dataclass(eq=False)
class CubaLI(NIRNode):
    r"""Current based leaky integrator model.

    The current based leaky integrator neuron model
    is defined by the following equations:

    .. math::
        \tau_{syn} \dot {I} = - I + w_{in} S

    .. math::
        \tau_{mem} \dot {v} = (v_{leak} - v) + R I

    Where :math:`\tau_{syn}` is the synaptic time constant,
    :math:`\tau_{mem}` is the membrane time constant,
    :math:`R` is the resistance,
    :math:`v_{leak}` is the leak voltage,
    :math:`w_{in}` is the input current weight (elementwise),
    and :math:`S` is the input spike.
    """

    tau_syn: np.ndarray  # Time constant
    tau_mem: np.ndarray  # Time constant
    r: np.ndarray  # Resistance
    v_leak: np.ndarray  # Leak voltage
    w_in: np.ndarray = 1.0  # Input current weight
    input_type: Optional[Dict[str, np.ndarray]] = None
    output_type: Optional[Dict[str, np.ndarray]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        assert (
            self.tau_syn.shape
            == self.tau_mem.shape
            == self.r.shape
            == self.v_leak.shape
        ), "All parameters must have the same shape"
        # If w_in is a scalar, make it an array of same shape as v_leak
        self.w_in = np.ones_like(self.v_leak) * self.w_in
        self.input_type = {"input": np.array(self.v_leak.shape)}
        self.output_type = {"output": np.array(self.v_leak.shape)}


@dataclass(eq=False)
class CubaLIF(NIRNode):
    r"""Current based leaky integrate and-fire-neuron model.

    The current based leaky integrate-and-fire neuron model
    is defined by the following equations:

    .. math::
        \tau_{syn} \dot {I} = - I + w_{in} S

    .. math::
        \tau_{mem} \dot {v} = (v_{leak} - v) + R I

    .. math::
        z = \begin{cases}
            1 & v > v_{threshold} \\
            0 & else
        \end{cases}

    .. math::
        v = \begin{cases}
            v_{reset} & z=1 \\
            v & else
        \end{cases}

    Where :math:`\tau_{syn}` is the synaptic time constant,
    :math:`\tau_{mem}` is the membrane time constant,
    :math:`R` is the resistance,
    :math:`v_{leak}` is the leak voltage,
    :math:`v_{threshold}` is the firing threshold,
    :math:`v_{reset}` is the reset potential,
    :math:`w_{in}` is the input current weight (elementwise)
    and :math:`S` is the input spike.
    """

    tau_syn: np.ndarray  # Time constant
    tau_mem: np.ndarray  # Time constant
    r: np.ndarray  # Resistance
    v_leak: np.ndarray  # Leak voltage
    v_threshold: np.ndarray  # Firing threshold
    v_reset: np.ndarray  # Reset potential
    w_in: np.ndarray = 1.0  # Input current weight
    input_type: Optional[Dict[str, np.ndarray]] = None
    output_type: Optional[Dict[str, np.ndarray]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        assert (
            self.tau_syn.shape
            == self.tau_mem.shape
            == self.r.shape
            == self.v_leak.shape
            == self.v_reset.shape
            == self.v_threshold.shape
        ), "All parameters must have the same shape"
        # If w_in is a scalar, make it an array of same shape as v_threshold
        self.w_in = np.ones_like(self.v_threshold) * self.w_in
        self.input_type = {"input": np.array(self.v_threshold.shape)}
        self.output_type = {"output": np.array(self.v_threshold.shape)}

    @classmethod
    def from_dict(cls, kwargs: Dict[str, Any]) -> "CubaLIF":
        if "v_reset" not in kwargs:
            kwargs["v_reset"] = np.zeros_like(kwargs["v_threshold"])
        return super().from_dict(kwargs)


@dataclass(eq=False)
class I(NIRNode):  # noqa: E742
    r"""Integrator.

    The integrator neuron model is defined by the following equation:

    .. math::
        \dot{v} = R I
    """

    r: np.ndarray
    input_type: Optional[Dict[str, np.ndarray]] = None
    output_type: Optional[Dict[str, np.ndarray]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.input_type = {"input": np.array(self.r.shape)}
        self.output_type = {"output": np.array(self.r.shape)}


@dataclass(eq=False)
class IF(NIRNode):
    r"""Integrate-and-fire neuron model.

    The integrate-and-fire neuron model is defined by the following equations:

    .. math::
        \dot{v} = R I

    .. math::
        z = \begin{cases}
            1 & v > v_{thr} \\
            0 & else
        \end{cases}

    .. math::
        v = \begin{cases}
            v_{reset} & z=1 \\
            v & else
        \end{cases}
    """

    r: np.ndarray  # Resistance
    v_threshold: np.ndarray  # Firing threshold
    v_reset: np.ndarray  # Reset potential
    input_type: Optional[Dict[str, np.ndarray]] = None
    output_type: Optional[Dict[str, np.ndarray]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        assert (
            self.r.shape == self.v_threshold.shape == self.v_reset.shape
        ), "All parameters must have the same shape"
        self.input_type = {"input": np.array(self.r.shape)}
        self.output_type = {"output": np.array(self.r.shape)}

    @classmethod
    def from_dict(cls, kwargs: Dict[str, Any]) -> "IF":
        if "v_reset" not in kwargs:
            kwargs["v_reset"] = np.zeros_like(kwargs["v_threshold"])
        return super().from_dict(kwargs)


@dataclass(eq=False)
class LI(NIRNode):
    r"""Leaky integrator neuron model.

    The leaky integrator neuron model is defined by the following equation:

    .. math::
        \tau \dot{v} = (v_{leak} - v) + R I

    Where :math:`\tau` is the time constant, :math:`v` is the membrane potential,
    :math:`v_{leak}` is the leak voltage, :math:`R` is the resistance, and :math:`I`
    is the input current.
    """

    tau: np.ndarray  # Time constant
    r: np.ndarray  # Resistance
    v_leak: np.ndarray  # Leak voltage
    input_type: Optional[Dict[str, np.ndarray]] = None
    output_type: Optional[Dict[str, np.ndarray]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        assert (
            self.tau.shape == self.r.shape == self.v_leak.shape
        ), "All parameters must have the same shape"
        self.input_type = {"input": np.array(self.r.shape)}
        self.output_type = {"output": np.array(self.r.shape)}


@dataclass(eq=False)
class LIF(NIRNode):
    r"""Leaky integrate and-fire-neuron model.

    The leaky integrate-and-fire neuron model is defined by the following equations:

    .. math::
        \tau \dot{v} = (v_{leak} - v) + R I

    .. math::
        z = \begin{cases}
            1 & v > v_{thr} \\
            0 & else
        \end{cases}

    .. math::
        v = \begin{cases}
            v_{reset} & z=1 \\
            v & else
        \end{cases}

    Where :math:`\tau` is the time constant,
    :math:`v` is the membrane potential,
    :math:`v_{leak}` is the leak voltage,
    :math:`R` is the resistance,
    :math:`v_{threshold}` is the firing threshold,
    :math:`v_{reset}` is the reset potential
    and :math:`I` is the input current.
    """

    tau: np.ndarray  # Time constant
    r: np.ndarray  # Resistance
    v_leak: np.ndarray  # Leak voltage
    v_threshold: np.ndarray  # Firing threshold
    v_reset: np.ndarray  # Reset potential
    input_type: Optional[Dict[str, np.ndarray]] = None
    output_type: Optional[Dict[str, np.ndarray]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        assert (
            self.tau.shape
            == self.r.shape
            == self.v_leak.shape
            == self.v_reset.shape
            == self.v_threshold.shape
        ), "All parameters must have the same shape"
        self.input_type = {"input": np.array(self.r.shape)}
        self.output_type = {"output": np.array(self.r.shape)}

    @classmethod
    def from_dict(cls, kwargs: Dict[str, Any]) -> "LIF":
        if "v_reset" not in kwargs:
            kwargs["v_reset"] = np.zeros_like(kwargs["v_threshold"])
        return super().from_dict(kwargs)
