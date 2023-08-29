import typing
from dataclasses import dataclass

import numpy as np

Edges = typing.NewType("Edges", typing.List[typing.Tuple[str, str]])


@dataclass
class NIRNode:
    """Base superclass of Neural Intermediate Representation Unit (NIR).

    All NIR primitives inherit from this class, but NIRNodes should never be
    instantiated.
    """


@dataclass
class NIRGraph(NIRNode):
    """Neural Intermediate Representation (NIR) Graph containing a number of nodes and
    edges.

    A graph of computational nodes and identity edges.
    """

    nodes: typing.Dict[str, NIRNode]  # List of computational nodes
    edges: Edges

    @staticmethod
    def from_list(*nodes: NIRNode) -> "NIRGraph":
        """Create a sequential graph from a list of nodes by labelling them after
        indices."""
        return NIRGraph(
            nodes={str(i): n for i, n in enumerate(nodes)},
            edges=[(str(i), str(i + 1)) for i in range(len(nodes) - 1)],
        )


@dataclass
class Affine(NIRNode):
    r"""Affine transform that linearly maps and translates the input signal.

    This is equivalent to the `Affine transformation <https://en.wikipedia.org/wiki/Affine_transformation>`_

    .. math::
        y(t) = W*x(t) + b
    """
    weight: np.ndarray  # Weight term
    bias: np.ndarray  # Bias term


@dataclass
class Conv1d(NIRNode):
    """Convolutional layer in 1d."""

    weight: np.ndarray  # Weight C_out * C_in * X
    stride: int  # Stride
    padding: int  # Padding
    dilation: int  # Dilation
    groups: int  # Groups
    bias: np.ndarray  # Bias C_out


@dataclass
class Conv2d(NIRNode):
    """Convolutional layer in 2d."""

    weight: np.ndarray  # Weight C_out * C_in * X * Y
    stride: int  # Stride
    padding: int  # Padding
    dilation: int  # Dilation
    groups: int  # Groups
    bias: np.ndarray  # Bias C_out

    def __post_init__(self):
        if isinstance(self.stride, int):
            self.stride = (self.stride, self.stride)
        if isinstance(self.padding, int):
            self.padding = (self.padding, self.padding)
        if isinstance(self.dilation, int):
            self.dilation = (self.dilation, self.dilation)


@dataclass
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
            v-v_{threshold} & z=1 \\
            v & else
        \end{cases}

    Where :math:`\tau_{syn}` is the synaptic time constant,
    :math:`\tau_{mem}` is the membrane time constant,
    :math:`w_{in}` is the input current weight (elementwise),
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
    w_in: np.ndarray = 1.0  # Input current weight

    def __post_init__(self):
        # If w_in is a scalar, make it an array of same shape as v_threshold
        self.w_in = np.ones_like(self.v_threshold) * self.w_in


@dataclass
class Delay(NIRNode):
    """Simple delay node.

    This node implements a simple delay:

    .. math::
        y(t) = x(t - \tau)
    """

    delay: np.ndarray  # Delay


@dataclass
class Flatten(NIRNode):
    """Flatten node.

    This node flattens its input tensor.
    """

    start_dim: int = 1  # First dimension to flatten
    end_dim: int = -1  # Last dimension to flatten


@dataclass
class I(NIRNode):  # noqa: E742
    r"""Integrator.

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
class Output(NIRNode):
    """Output Node.

    Defines an output of the graph.
    """

    shape: int  # Size of output


@dataclass
class Scale(NIRNode):
    r"""Scales a signal by some values.

    This node is equivalent to the
    `Hadamard product <https://en.wikipedia.org/wiki/Hadamard_product_(matrices)>`_.

    .. math::
        y(t) = x(t) \odot s
    """

    scale: np.ndarray  # Scaling factor


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


@dataclass
class Braid(NIRNode):
    r"""Braids the input signals (wires) by providing a mapping from input indices to output indices.
    The mapping does not have to be 1:1 (bijective): 
    it can ignore signals (terminal objects), or duplicate signals, but it cannot create new signals
    (initial objects).
    Note that we assume the input signals are ordered. That is, an input signal will have a 
    set of inputs that are numbered from 0 to N-1, where N is the total number of inputs.

    For instance:

    >>> Braid(output_indices=np.array([np.nan, 0, np.nan, np.nan, 1]))
    >>> (x, y) -> (0, x, 0, 0, y)

    If the given mapping contains a single number, it is interpreted to take the first value of *any*-size input.
    That is, tuples of size one are considered individual values and not as tuples.

    >>> Braid(output_indices=np.array([0]))
    >>> (x, y, z) -> x

    We can also duplicate the same signal twice:

    >>> Braid(output_indices=np.array([np.nan, 0, np.nan, 0])
    >>> x -> (0, x, 0, x)
    """

    output_indices: np.ndarray
