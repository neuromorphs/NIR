from dataclasses import dataclass
import typing

import numpy as np


Edges = typing.NewType("Edges", typing.List[typing.Tuple[int, int]])


class NIRNode:
    """Base superclass of Neural Intermediate Representation Unit (NIR).
    All NIR primitives inherit from this class, but NIRNodes should never be
    instantiated.
    """


@dataclass
class NIRGraph(NIRNode):
    """Neural Intermediate Representation (NIR) Graph containing a number
    of nodes and edges.

    A graph of computational nodes and identity edges."""

    nodes: typing.List[NIRNode]  # List of computational nodes
    edges: Edges


@dataclass
class Affine(NIRNode):
    r"""Affine transform:

    .. math::
        y(t) = W*x(t) + b
    """
    weight: np.ndarray  # Weight term
    bias: np.ndarray  # Bias term


@dataclass
class Conv1d(NIRNode):
    """Convolutional layer in 1d"""

    weight: np.ndarray  # Weight C_out * C_in * X
    stride: int  # Stride
    padding: int  # Padding
    dilation: int  # Dilation
    groups: int  # Groups
    bias: np.ndarray  # Bias C_out


@dataclass
class Conv2d(NIRNode):
    """Convolutional layer in 2d"""

    weight: np.ndarray  # Weight C_out * C_in * X * Y
    stride: int  # Stride
    padding: int  # Padding
    dilation: int  # Dilation
    groups: int  # Groups
    bias: np.ndarray  # Bias C_out


@dataclass
class Delay(NIRNode):
    """Simple delay node.

    This node implements a simple delay:

    .. math::
        y(t) = x(t - \tau)
    """

    delay: np.ndarray  # Delay


@dataclass
class I(NIRNode):  # noqa: E742
    r"""Integrator

    The integrator neuron model is defined by the following equation:

    .. math::
        \dot{v} = R I
    """

    r: np.ndarray


@dataclass
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
            v-v_{thr} & z=1 \\
            v & else
        \end{cases}
    """

    r: np.ndarray  # Resistance
    v_threshold: np.ndarray  # Firing threshold


@dataclass
class Input(NIRNode):
    """Input Node.

    This is a virtual node, which allows feeding in data into the graph.
    """

    shape: np.ndarray  # Shape of input data


@dataclass
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


@dataclass
class Linear(NIRNode):
    r"""Linear transform without bias:

    .. math::
        y(t) = W*x(t)
    """
    weight: np.ndarray  # Weight term


@dataclass
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
            v-v_{thr} & z=1 \\
            v & else
        \end{cases}

    Where :math:`\tau` is the time constant, :math:`v` is the membrane potential,
    :math:`v_{leak}` is the leak voltage, :math:`R` is the resistance, 
    :math:`v_{threshold}` is the firing threshold, and :math:`I` is the input current.
    """

    tau: np.ndarray  # Time constant
    r: np.ndarray  # Resistance
    v_leak: np.ndarray  # Leak voltage
    v_threshold: np.ndarray  # Firing threshold


@dataclass
class CubaLIF(NIRNode):
    r"""Current based leaky integrate and-fire-neuron model.

    The current based leaky integrate-and-fire neuron model
    is defined by the following equations:

    .. math::
        \tau_syn \ dot{I} = - I + S

        \tau_mem \dot{v} = (v_{leak} - v) + R I

    .. math::
        z = \begin{cases}
            1 & v > v_{thr} \\
            0 & else
        \end{cases}

    .. math::
        v = \begin{cases}
            v-v_{thr} & z=1 \\
            v & else
        \end{cases}

    Where :math:`\tau_syn` is the synaptic time constant,
    :math:`\tau_mem` is the membrane time constant,
    :math:`v` is the membrane potential,
    :math:`v_{leak}` is the leak voltage,
    :math:`R` is the resistance,
    :math:`v_{threshold}` is the firing threshold,
    and :math:`S` is the input spike.
    """

    tau_syn: np.ndarray  # Time constant
    tau_mem: np.ndarray  # Time constant
    r: np.ndarray  # Resistance
    v_leak: np.ndarray  # Leak voltage
    v_threshold: np.ndarray  # Firing threshold


@dataclass
class Output(NIRNode):
    """Output Node.

    Defines an output of the graph.
    """

    pass


@dataclass
class Threshold(NIRNode):
    r"""Threshold node.

    This node implements the heaviside step function:

    .. math::
        z = \begin{cases}
            1 & v > v_{thr} \\
            0 & else
        \end{cases}
    """

    threshold: np.ndarray  # Firing threshold
