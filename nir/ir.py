from dataclasses import dataclass
import typing

import numpy as np


Edges = typing.NewType("Edges", list[typing.Tuple[int, int]])


@dataclass
class NIR:
    """Neural Intermediate Representation (NIR)"""

    nodes: typing.List[typing.Any]  # List of computational nodes
    edges: Edges


class NIRNode:
    """Basic Neural Intermediate Representation Unit (NIRUnit)"""

    pass


@dataclass
class LI(NIRNode):
    """Leaky integrator neuron model.

    The leaky integrator neuron model is defined by the following equation:
    $$
    \tau \dot{v} = (v_{leak} - v) + R I
    $$
    Where $\tau$ is the time constant, $v$ is the membrane potential,
    $v_{leak}$ is the leak voltage, $R$ is the resistance, and $I$ is the
    input current.
    """

    tau: np.ndarray  # Time constant
    r: np.ndarray  # Resistance
    v_leak: np.ndarray  # Leak voltage


@dataclass
class LIF(NIRNode):
    """Leaky integrate and-fire-neuron model.

    The leaky integrate-and-fire neuron model is defined by the following equations:
    $$
    \tau \dot{v} = (v_{leak} - v) + R I
    z = \being{cases}
        1 & v > theta \\
        0 & else
    \end{cases}
    v = \begin{cases}
        v-theta & z=1 \\
        v & else
    \end{cases}
    $$
    Where $\tau$ is the time constant, $v$ is the membrane potential,
    $v_{leak}$ is the leak voltage, $R$ is the resistance, $theta$ is
    the firing threshold, and $I$ is the input current.
    """

    tau: np.ndarray  # Time constant
    r: np.ndarray  # Resistance
    v_leak: np.ndarray  # Leak voltage
    theta: np.ndarray  # Firing threshold


@dataclass
class Linear(NIRNode):

    weights: np.ndarray  # Weights M * N
    bias: np.ndarray  # Bias M


@dataclass
class Conv1d(NIRNode):
    """Convolutional layer in 1d"""

    weights: np.ndarray  # Weights C_out * C_in * X
    stride: int  # Stride
    padding: int  # Padding
    dilation: int  # Dilation
    groups: int  # Groups
    bias: np.ndarray  # Bias C_out


@dataclass
class Conv2d(NIRNode):
    """Convolutional layer in 2d"""

    weights: np.ndarray  # Weights C_out * C_in * X * Y
    stride: int  # Stride
    padding: int  # Padding
    dilation: int  # Dilation
    groups: int  # Groups
    bias: np.ndarray  # Bias C_out
