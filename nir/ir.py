import typing
from collections import Counter
from dataclasses import dataclass

import numpy as np

Edges = typing.NewType("Edges", typing.List[typing.Tuple[str, str]])
# shape is either simple array shape or dictionary mapping port name to array shape
Shape = typing.NewType("Shape", typing.Union[np.ndarray, typing.Dict[str, np.ndarray]])


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

    input_shape: typing.Optional[Shape] = None
    output_shape: typing.Optional[Shape] = None

    @staticmethod
    def from_list(*nodes: NIRNode) -> "NIRGraph":
        """Create a sequential graph from a list of nodes by labelling them after
        indices."""

        def unique_node_name(node, counts):
            basename = node.__class__.__name__.lower()
            id = counts[basename]
            name = f"{basename}{f'_{id}' if id>0 else ''}"
            counts[basename] += 1
            return name

        counts = Counter()
        node_dict = {}
        edges = []

        for node in nodes:
            name = unique_node_name(node, counts)
            node_dict[name] = node

        names = list(node_dict)
        for i in range(len(nodes) - 1):
            edges.append((names[i], names[i + 1]))

        return NIRGraph(
            nodes=node_dict,
            edges=edges,
        )

    def __post_init__(self):
        # if unspecified, set input and output shapes based on the graph's input and output nodes
        if self.input_shape is None:
            input_node_keys = [
                k for k, node in self.nodes.items() if isinstance(node, Input)
            ]
            self.input_shape = (
                {node_key: self.nodes[node_key].shape for node_key in input_node_keys}
                if len(input_node_keys) > 0
                else None
            )
        if self.output_shape is None:
            output_node_keys = [
                k for k, node in self.nodes.items() if isinstance(node, Output)
            ]
            self.output_shape = {
                node_key: self.nodes[node_key].shape for node_key in output_node_keys
            }


@dataclass
class Affine(NIRNode):
    r"""Affine transform that linearly maps and translates the input signal.

    This is equivalent to the
    `Affine transformation <https://en.wikipedia.org/wiki/Affine_transformation>`_

    Assumes a one-dimensional input vector of shape (N,).

    .. math::
        y(t) = W*x(t) + b
    """
    weight: np.ndarray  # Weight term
    bias: np.ndarray  # Bias term

    input_shape: typing.Optional[Shape] = None
    output_shape: typing.Optional[Shape] = None

    def __post_init__(self):
        # set input and output shape, if not set by user
        if self.input_shape is None:
            out_dim = 1 if len(self.weight.shape) == 1 else self.weight.shape[1]
            self.input_shape = np.array([out_dim])
        if self.output_shape is None:
            self.output_shape = np.array([self.weight.shape[0]])


@dataclass
class Conv1d(NIRNode):
    """Convolutional layer in 1d."""

    weight: np.ndarray  # Weight C_out * C_in * X
    stride: int  # Stride
    padding: int  # Padding
    dilation: int  # Dilation
    groups: int  # Groups
    bias: np.ndarray  # Bias C_out

    input_shape: typing.Optional[Shape] = None
    output_shape: typing.Optional[Shape] = None

    def __post_init__(self):
        # set input and output shape, if not set by user
        if self.input_shape is None:
            self.input_shape = np.array(self.weight.shape)[1:]
        if self.output_shape is None:
            self.output_shape = np.array(self.weight.shape)[[0, 2]]


@dataclass
class Conv2d(NIRNode):
    """Convolutional layer in 2d."""

    weight: np.ndarray  # Weight C_out * C_in * X * Y
    stride: int  # Stride
    padding: int  # Padding
    dilation: int  # Dilation
    groups: int  # Groups
    bias: np.ndarray  # Bias C_out

    input_shape: typing.Optional[Shape] = None
    output_shape: typing.Optional[Shape] = None

    def __post_init__(self):
        if isinstance(self.stride, int):
            self.stride = (self.stride, self.stride)
        if isinstance(self.padding, int):
            self.padding = (self.padding, self.padding)
        if isinstance(self.dilation, int):
            self.dilation = (self.dilation, self.dilation)
        # set input and output shape, if not set by user
        if self.input_shape is None:
            self.input_shape = np.array(self.weight.shape)[1:]
        if self.output_shape is None:
            self.output_shape = np.array(self.weight.shape)[[0, 2, 3]]


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

    input_shape: typing.Optional[Shape] = None
    output_shape: typing.Optional[Shape] = None

    def __post_init__(self):
        # If w_in is a scalar, make it an array of same shape as v_threshold
        self.w_in = np.ones_like(self.v_threshold) * self.w_in
        # set input and output shape, if not set by user
        self.input_shape = self.output_shape = np.array(self.v_threshold.shape)


@dataclass
class Delay(NIRNode):
    """Simple delay node.

    This node implements a simple delay:

    .. math::
        y(t) = x(t - \tau)
    """

    delay: np.ndarray  # Delay

    input_shape: typing.Optional[Shape] = None
    output_shape: typing.Optional[Shape] = None

    def __post_init__(self):
        # set input and output shape, if not set by user
        self.input_shape = self.output_shape = np.array(self.delay.shape)


@dataclass
class Flatten(NIRNode):
    """Flatten node.

    This node flattens its input tensor.
    """

    start_dim: int = 1  # First dimension to flatten
    end_dim: int = -1  # Last dimension to flatten

    input_shape: typing.Optional[Shape] = None
    output_shape: typing.Optional[Shape] = None

    def __post_init__(self):
        # force shape annotation
        if self.input_shape is None or self.output_shape is None:
            raise ValueError(
                "must provide input and output shape for flatten (cannot infer)"
            )
        # make sure input and output shape are valid
        if np.prod(self.input_shape) != np.prod(self.output_shape):
            raise ValueError("input and output shape must have same number of elements")


@dataclass
class I(NIRNode):  # noqa: E742
    r"""Integrator.

    The integrator neuron model is defined by the following equation:

    .. math::
        \dot{v} = R I
    """

    r: np.ndarray

    input_shape: typing.Optional[Shape] = None
    output_shape: typing.Optional[Shape] = None

    def __post_init__(self):
        self.input_shape = self.output_shape = np.array(self.r.shape)


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

    input_shape: typing.Optional[Shape] = None
    output_shape: typing.Optional[Shape] = None

    def __post_init__(self):
        self.input_shape = self.output_shape = np.array(self.r.shape)


@dataclass
class Input(NIRNode):
    """Input Node.

    This is a virtual node, which allows feeding in data into the graph.
    """

    shape: np.ndarray  # Shape of input data

    input_shape: typing.Optional[Shape] = None
    output_shape: typing.Optional[Shape] = None

    def __post_init__(self):
        self.input_shape = self.output_shape = self.shape


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

    input_shape: typing.Optional[Shape] = None
    output_shape: typing.Optional[Shape] = None

    def __post_init__(self):
        self.input_shape = self.output_shape = np.array(self.r.shape)


@dataclass
class Linear(NIRNode):
    r"""Linear transform without bias:

    .. math::
        y(t) = W*x(t)
    """
    weight: np.ndarray  # Weight term

    input_shape: typing.Optional[Shape] = None
    output_shape: typing.Optional[Shape] = None

    def __post_init__(self):
        # set input and output shape, if not set by user
        if self.input_shape is None:
            out_dim = 1 if len(self.weight.shape) == 1 else self.weight.shape[1]
            self.input_shape = np.array([out_dim])
        if self.output_shape is None:
            self.output_shape = np.array([self.weight.shape[0]])


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

    input_shape: typing.Optional[Shape] = None
    output_shape: typing.Optional[Shape] = None

    def __post_init__(self):
        self.input_shape = self.output_shape = np.array(self.r.shape)


@dataclass
class Output(NIRNode):
    """Output Node.

    Defines an output of the graph.
    """

    shape: int  # Size of output

    input_shape: typing.Optional[Shape] = None
    output_shape: typing.Optional[Shape] = None

    def __post_init__(self):
        self.input_shape = self.output_shape = self.shape


@dataclass
class Scale(NIRNode):
    r"""Scales a signal by some values.

    This node is equivalent to the
    `Hadamard product <https://en.wikipedia.org/wiki/Hadamard_product_(matrices)>`_.

    .. math::
        y(t) = x(t) \odot s
    """

    scale: np.ndarray  # Scaling factor

    input_shape: typing.Optional[Shape] = None
    output_shape: typing.Optional[Shape] = None

    def __post_init__(self):
        self.input_shape = self.output_shape = np.array(self.scale.shape)


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

    input_shape: typing.Optional[Shape] = None
    output_shape: typing.Optional[Shape] = None

    def __post_init__(self):
        self.input_shape = self.output_shape = np.array(self.threshold.shape)
