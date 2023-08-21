import typing
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

Edges = typing.NewType("Edges", typing.List[typing.Tuple[str, str]])


@dataclass
class NIRNode:
    """Base superclass of Neural Intermediate Representation Unit (NIR).

    All NIR primitives inherit from this class, but NIRNodes should never be
    instantiated.
    """

    shape_input: np.ndarray = field(init=False, kw_only=True)  # Shape of input
    shape_output: np.ndarray = field(init=False, kw_only=True)  # Shape of output


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

        def unique_node_name(node, counts):
            basename = node.__class__.__name__.lower()
            id = counts[basename]
            name = f"{basename}{f'_{id}' if id>0 else ''}"
            counts[basename] += 1
            return name

        counts = Counter()
        node_dict = [
            ("input", Input(shape=nodes[0].shape_input)),
        ]
        edges = []

        for node in nodes:
            name = unique_node_name(node, counts)
            node_dict.append((name, node))

        node_dict.append(("output", Output(shape=nodes[-1].shape_output)))

        for i in range(len(node_dict) - 1):
            edges.append((node_dict[i][0], node_dict[i + 1][0]))

        return NIRGraph(
            nodes={n[0]: n[1] for n in node_dict},
            edges=edges,
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

    def __post_init__(self):
        assert len(self.weight.shape) >= 2, "Weight must be at least 2D"
        self.shape_input = self.weight.shape[:-2] + tuple(
            np.array(self.weight.shape[-2:]).T
        )
        self.shape_output = self.weight.shape[:-2] + (self.weight.shape[-2],)


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
        assert (
            self.tau_syn.shape
            == self.tau_mem.shape
            == self.r.shape
            == self.v_leak.shape
            == self.v_threshold.shape
        ), "All parameters must have the same shape"
        # If w_in is a scalar, make it an array of same shape as v_threshold
        self.w_in = np.ones_like(self.v_threshold) * self.w_in
        self.shape_input = self.shape_output = self.v_threshold.shape


@dataclass
class Delay(NIRNode):
    """Simple delay node.

    This node implements a simple delay:

    .. math::
        y(t) = x(t - \tau)
    """

    delay: np.ndarray  # Delay

    def __post_init__(self):
        self.shape_input = self.shape_output = self.delay.shape


@dataclass
class Flatten(NIRNode):
    """Flatten node.

    This node flattens its input tensor.
    """

    # Shape of input tensor (overrrides shape_input from NIRNode to allow for non-keyword (positional) initialization)
    shape_input: np.ndarray = field(kw_only=False)
    start_dim: int = 1  # First dimension to flatten
    end_dim: int = -1  # Last dimension to flatten

    def __post_init__(self):
        concat = self.shape_input[self.start_dim : self.end_dim].prod()
        self.shape_output = np.array(
            [
                *self.shape_input[: self.start_dim],
                concat,
                *self.shape_input[self.end_dim :],
            ]
        )


@dataclass
class I(NIRNode):  # noqa: E742
    r"""Integrator.

    The integrator neuron model is defined by the following equation:

    .. math::
        \dot{v} = R I
    """

    r: np.ndarray

    def __post_init__(self):
        self.shape_input = self.shape_output = self.r.shape


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

    def __post_init__(self):
        assert (
            self.r.shape == self.v_threshold.shape
        ), "All parameters must have the same shape"
        self.shape_input = self.shape_output = self.v_threshold.shape


@dataclass
class Input(NIRNode):
    """Input Node.

    This is a virtual node, which allows feeding in data into the graph.
    """

    shape: np.ndarray  # Shape of input data

    def __post_init__(self):
        self.shape_input = self.shape_output = self.shape


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

    def __post_init__(self):
        assert (
            self.tau.shape == self.r.shape == self.v_leak.shape
        ), "All parameters must have the same shape"
        self.shape_input = self.shape_output = self.v_leak.shape


@dataclass
class Linear(NIRNode):
    r"""Linear transform without bias:

    .. math::
        y(t) = W*x(t)
    """
    weight: np.ndarray  # Weight term

    def __post_init__(self):
        assert len(self.weight.shape) >= 2, "Weight must be at least 2D"
        self.shape_input = self.weight.shape[:-2] + tuple(
            np.array(self.weight.shape[-2:]).T
        )
        self.shape_output = self.weight.shape[:-2] + (self.weight.shape[-2],)


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

    def __post_init__(self):
        assert (
            self.tau.shape
            == self.r.shape
            == self.v_leak.shape
            == self.v_threshold.shape
        ), "All parameters must have the same shape"
        self.shape_input = self.shape_output = self.v_threshold.shape


@dataclass
class Output(NIRNode):
    """Output Node.

    Defines an output of the graph.
    """

    shape: np.ndarray  # Size of output

    def __post_init__(self):
        self.shape_input = self.shape_output = self.shape


@dataclass
class Scale(NIRNode):
    r"""Scales a signal by some values.

    This node is equivalent to the
    `Hadamard product <https://en.wikipedia.org/wiki/Hadamard_product_(matrices)>`_.

    .. math::
        y(t) = x(t) \odot s
    """

    scale: np.ndarray  # Scaling factor

    def __post_init__(self):
        self.shape_input = self.shape_output = self.scale.shape


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

    def __post_init__(self):
        self.shape_input = self.shape_output = self.threshold.shape
