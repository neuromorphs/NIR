import typing
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

# Nodes are uniquely named computational units
Nodes = typing.Dict[str, "NIRNode"]
# Edges map one node id to another via the identity
Edges = typing.List[typing.Tuple[str, str]]
# Types is a dict mapping strings to tensor shapes
Types = typing.Dict[str, np.ndarray]


def _parse_shape_argument(x: Types, key: str):
    """Parse the shape argument of a NIR node."""
    if isinstance(x, np.ndarray):
        return {key: x}
    elif isinstance(x, Sequence):
        return {key: np.array(x)}
    elif isinstance(x, dict):
        return x
    elif x is None:
        return {key: None}


def _calculate_conv_output(
    input_shape: typing.Union[int, typing.Sequence[int]],
    padding: typing.Union[int, str, typing.Sequence[int]],
    dilation: typing.Union[int, typing.Sequence[int]],
    kernel_size: typing.Union[int, typing.Sequence[int]],
    stride: typing.Union[int, typing.Sequence[int]],
) -> typing.Sequence[int]:
    """Calculates the output for a single dimension of a convolutional layer.
    https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html#torch.nn.Conv1d

    :param input_shape: input shape, either int or (int, int)
    :type input_shape: int | typing.Sequence[int]
    :param padding: padding
    :type padding: int | typing.Sequence[int]
    :param dilation: dilation
    :type dilation: int | typing.Sequence[int]
    :param kernel_size: kernel size
    :type kernel_size: int | typing.Sequence[int]
    :param stride: stride
    :type stride: int | typing.Sequence[int]

    :return: output shape
    :rtype: typing.Sequence[int]
    """
    if isinstance(input_shape, (int, np.integer)):
        ndim = 1
    else:
        ndim = len(input_shape)
    if isinstance(padding, str) and padding == 'valid':
        padding = [0] * ndim
    shapes = []
    for i in range(ndim):
        if isinstance(padding, str) and padding == 'same':
            shape = input_shape[i]
        else:
            shape = np.floor(
                (
                    _index_tuple(input_shape, i)
                    + 2 * _index_tuple(padding, i)
                    - _index_tuple(dilation, i) * (_index_tuple(kernel_size, i) - 1)
                    - 1
                )
                / _index_tuple(stride, i)
                + 1
            )
        shapes.append(int(shape))
    return np.array(shapes)


def _index_tuple(
    tuple: typing.Union[int, typing.Sequence[int]], index: int
) -> typing.Union[int, np.ndarray]:
    """If the input is a tuple/array, index it. Otherwise, return it as-is."""
    if isinstance(tuple, np.ndarray) or isinstance(tuple, Sequence):
        return tuple[index]
    elif isinstance(tuple, (int, np.integer)):
        return np.array([tuple])
    else:
        raise TypeError(f"tuple must be int or np.ndarray, not {type(tuple)}")


@dataclass(eq=False)
class NIRNode:
    """Base superclass of Neural Intermediate Representation Unit (NIR).

    All NIR primitives inherit from this class, but NIRNodes should never be
    instantiated.
    """

    # Note: Adding input/output types as follows is ideal, but requires Python 3.10
    # input_type: Types = field(init=False, kw_only=True)
    # output_type: Types = field(init=False, kw_only=True)

    def __eq__(self, other):
        return self is other


@dataclass(eq=False)
class NIRGraph(NIRNode):
    """Neural Intermediate Representation (NIR) Graph containing a number of nodes and
    edges.

    A graph of computational nodes and identity edges.
    """

    nodes: Nodes  # List of computational nodes
    edges: Edges  # List of edges between nodes

    @property
    def inputs(self):
        return {
            name: node for name, node in self.nodes.items() if isinstance(node, Input)
        }

    @property
    def outputs(self):
        return {
            name: node for name, node in self.nodes.items() if isinstance(node, Output)
        }

    @staticmethod
    def from_list(*nodes: NIRNode) -> "NIRGraph":
        """Create a sequential graph from a list of nodes by labelling them after
        indices."""

        if len(nodes) > 0 and (
            isinstance(nodes[0], list) or isinstance(nodes[0], tuple)
        ):
            nodes = [*nodes[0]]

        def unique_node_name(node, counts):
            basename = node.__class__.__name__.lower()
            id = counts[basename]
            name = f"{basename}{f'_{id}' if id>0 else ''}"
            counts[basename] += 1
            return name

        counts = Counter()
        node_dict = {"input": Input(input_type=nodes[0].input_type)}
        edges = []

        for node in nodes:
            name = unique_node_name(node, counts)
            node_dict[name] = node

        node_dict["output"] = Output(output_type=nodes[-1].output_type)

        names = list(node_dict)
        for i in range(len(names) - 1):
            edges.append((names[i], names[i + 1]))

        return NIRGraph(
            nodes=node_dict,
            edges=edges,
        )

    def __post_init__(self):
        input_node_keys = [
            k for k, node in self.nodes.items() if isinstance(node, Input)
        ]
        self.input_type = (
            {node_key: self.nodes[node_key].input_type for node_key in input_node_keys}
            if len(input_node_keys) > 0
            else None
        )
        output_node_keys = [
            k for k, node in self.nodes.items() if isinstance(node, Output)
        ]
        self.output_type = {
            node_key: self.nodes[node_key].output_type for node_key in output_node_keys
        }
        # check that all nodes have consistent and defined input and output types
        try:
            self._check_types()
        except ValueError as e:
            print(f'[warning] {e}')
            self.infer_types()
            self._check_types()
    
    def _check_types(self):
        """Check that all nodes in the graph have input and output types. Will raise ValueError
        if any node has no input or output type, or if the types are inconsistent."""
        for edge in self.edges:
            pre_node = self.nodes[edge[0]]
            post_node = self.nodes[edge[1]]

            # make sure all types are defined
            undef_out_type = pre_node.output_type is None or any(
                v is None for v in pre_node.output_type.values()
            )
            if undef_out_type:
                raise ValueError(f'pre node {edge[0]} has no output type')
            undef_in_type = post_node.input_type is None or any(
                v is None for v in post_node.input_type.values()
            )
            if undef_in_type:
                raise ValueError(f'post node {edge[1]} has no input type')

            # make sure the length of types is equal
            if len(pre_node.output_type) != len(post_node.input_type):
                pre_repr = f'len({edge[0]}.output)={len(pre_node.output_type)}'
                post_repr = f'len({edge[1]}.input)={len(post_node.input_type)}'
                raise ValueError(f'type length mismatch: {pre_repr} -> {post_repr}')

            # make sure the type values match up
            if len(pre_node.output_type.keys()) == 1:
                post_input_type = list(post_node.input_type.values())[0]
                pre_output_type = list(pre_node.output_type.values())[0]
                if not np.array_equal(post_input_type, pre_output_type):
                    pre_repr = f'{edge[0]}.output: {pre_output_type}'
                    post_repr = f'{edge[1]}.input: {post_input_type}'
                    raise ValueError(f'type mismatch: {pre_repr} -> {post_repr}')
            else:
                raise NotImplementedError('multiple input/output types not supported yet')
        return True
    
    def _forward_type_inference(self, debug=True):
        """Infer the types of all nodes in this graph. Will modify the input_type and output_type
        of nodes in the graph as needed. Assumes that the input_type of the graph is set. Moves
        from the input nodes to the output nodes. Raises ValueError if types are inconsistent.

        Assumes that all input types are of form: {'input': ...} and all output types are of form:
        {'output': ...}.

        Currently only supports the inference of output types for Conv1d and Conv2d nodes.
        Does not support nested NIR graphs.
        """
        ready = [e for e in self.edges if e[0] in self.inputs.keys()]
        seen = set([e[0] for e in ready])
        while len(ready) > 0:
            pre_key, post_key = ready.pop()
            pre_node = self.nodes[pre_key]
            post_node = self.nodes[post_key]
            
            if isinstance(pre_node, NIRGraph) or isinstance(post_node, NIRGraph):
                raise NotImplementedError('type inference on nested NIR graphs not supported yet')
            
            # check if post input_type needs to be defined
            undef_post_input_type = post_node.input_type is None or any(
                v is None for v in post_node.input_type.values()
            )
            type_mismatch = any([
                len(post_node.input_type) != len(pre_node.output_type),
                not np.array_equal(
                    np.array(list(pre_node.output_type.values())), 
                    np.array(list(post_node.input_type.values()))
                )
            ])
            if undef_post_input_type:
                # define post input_type to be the same as pre output_type
                print(f'[warning] {post_key}.input_type undefined, set to {pre_key}.output_type')
                post_node.input_type = {
                    k.replace('output', 'input'): v for k, v in pre_node.output_type.items()
                }
            elif type_mismatch:
                # set post input_type to be the same as pre output_type
                pre_repr = f'{pre_key}.output: {np.array(list(pre_node.output_type.values()))}'
                post_repr = f'{post_key}.input: {np.array(list(post_node.input_type.values()))}'
                print(f'[warning] overwriting {post_repr} with {pre_repr}')
                post_node.input_type = {
                    k.replace('output', 'input'): v for k, v in pre_node.output_type.items()
                }

            # check if post output_type needs to be defined
            undef_post_output_type = post_node.output_type is None or any(
                v is None for v in post_node.output_type.values()
            )
            if undef_post_output_type:
                # define post output_type
                if isinstance(post_node, Conv1d) or isinstance(post_node, Conv2d):
                    if isinstance(post_node, Conv1d):
                        post_node.input_shape = post_node.input_type['input'][1]
                    else:
                        post_node.input_shape = tuple(post_node.input_type['input'][1:])
                    output_shape = _calculate_conv_output(
                        post_node.input_shape,
                        post_node.padding,
                        post_node.dilation,
                        post_node.weight.shape[2],
                        post_node.stride,
                    )
                    output_type = np.array([post_node.weight.shape[0], *output_shape])
                    post_node.output_type = {"output": output_type}
            
            seen.add(post_key)
            ready += [e for e in self.edges if e[0] == post_key and e[1] not in seen]

    def infer_types(self):
        """Infer the shapes of all nodes in this graph. Will modify the input_type and
        output_type of all nodes in the graph.
        
        Assumes that either the input type or the output type of the graph is set.
        Assumes that if A->B, then A.output_type.values() = B.input_type.values()
        """
        undef_input_type = self.input_type is None or any(
            v is None for v in self.input_type.values()
        )
        undef_output_type = self.output_type is None or any(
            v is None for v in self.output_type.values()
        )
        if not undef_input_type:
            # forward-mode type inferring
            self._forward_type_inference()
        elif not undef_output_type:
            # backward-mode type inferring
            raise NotImplementedError('backward-mode type inference not implemented yet')
        else:
            raise ValueError("Either input_type or output_type must be set")


@dataclass(eq=False)
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

    def __post_init__(self):
        assert len(self.weight.shape) >= 2, "Weight must be at least 2D"
        self.input_type = {
            "input": np.array(
                self.weight.shape[:-2] + tuple(np.array(self.weight.shape[-1:]).T)
            )
        }
        self.output_type = {
            "output": np.array(self.weight.shape[:-2] + (self.weight.shape[-2],))
        }


@dataclass(eq=False)
class Conv1d(NIRNode):
    """Convolutional layer in 1d.

    Note that the input_shape argument is required to disambiguate the shape, and is used
    to infer the exact output shape along with the other parameters. If the input_shape
    is None, the output shape will also be None.

    The NIRGraph.infer_all_shapes function may be used to automatically infer the input and
    output types on the graph level.

    :param input_shape: Shape of spatial input (N,)
    :type input_shape: Optional[int]
    :param weight: Weight, shape (C_out, C_in, N)
    :type weight: np.ndarray
    :param stride: Stride
    :type stride: int
    :param padding: Padding, if string must be 'same' or 'valid'
    :type padding: int | str
    :param dilation: Dilation
    :type dilation: int
    :param groups: Groups
    :type groups: int
    :param bias: Bias array of shape (C_out,)
    :type bias: np.ndarray
    """

    input_shape: typing.Optional[int]  # N
    weight: np.ndarray  # Weight C_out * C_in * N
    stride: int  # Stride
    padding: typing.Union[int, str]  # Padding
    dilation: int  # Dilation
    groups: int  # Groups
    bias: np.ndarray  # Bias C_out

    def __post_init__(self):
        if isinstance(self.padding, str) and self.padding not in ["same", "valid"]:
            raise ValueError(f"padding must be 'same', 'valid', or int, not {self.padding}")
        if self.input_shape is None:
            # leave input and output types undefined
            self.input_type = {"input": None}
            self.output_type = {"output": None}
        else:
            # infer input and output types from input_shape
            self.input_type = {"input": np.array([self.weight.shape[1], self.input_shape])}
            output_shape = _calculate_conv_output(
                self.input_shape,
                self.padding,
                self.dilation,
                self.weight.shape[2],
                self.stride,
            )
            self.output_type = {"output": np.array([self.weight.shape[0], *output_shape])}


@dataclass(eq=False)
class Conv2d(NIRNode):
    """Convolutional layer in 2d.

    Note that the input_shape argument is required to disambiguate the shape, and is used
    to infer the exact output shape along with the other parameters. If the input_shape
    is None, the output shape will also be None.

    The NIRGraph.infer_all_shapes function may be used to automatically infer the input and
    output types on the graph level.

    :param input_shape: Shape of spatial input (N_x, N_y)
    :type input_shape: Optional[tuple[int, int]]
    :param weight: Weight, shape (C_out, C_in, N_x, N_y)
    :type weight: np.ndarray
    :param stride: Stride
    :type stride: int | int, int
    :param padding: Padding, if string must be 'same' or 'valid'
    :type padding: int | int, int | str
    :param dilation: Dilation
    :type dilation: int | int, int
    :param groups: Groups
    :type groups: int
    :param bias: Bias array of shape (C_out,)
    :type bias: np.ndarray
    """

    # Shape of input tensor (overrrides input_type from
    input_shape: typing.Optional[typing.Tuple[int, int]]  # N_x, N_y
    weight: np.ndarray  # Weight C_out * C_in * W_x * W_y
    stride: typing.Union[int, typing.Tuple[int, int]]  # Stride
    padding: typing.Union[int, typing.Tuple[int, int], str]  # Padding
    dilation: typing.Union[int, typing.Tuple[int, int]]  # Dilation
    groups: int  # Groups
    bias: np.ndarray  # Bias C_out

    def __post_init__(self):
        if isinstance(self.padding, str) and self.padding not in ["same", "valid"]:
            raise ValueError(f"padding must be 'same', 'valid', or int, not {self.padding}")
        if isinstance(self.padding, int):
            self.padding = (self.padding, self.padding)
        if isinstance(self.stride, int):
            self.stride = (self.stride, self.stride)
        if isinstance(self.dilation, int):
            self.dilation = (self.dilation, self.dilation)
        if self.input_shape is None:
            # leave input and output types undefined
            self.input_type = {"input": None}
            self.output_type = {"output": None}
        else:
            # infer input and output types from input_shape
            self.input_type = {"input": np.array([self.weight.shape[1], *self.input_shape])}
            output_shape = _calculate_conv_output(
                self.input_shape,
                self.padding,
                self.dilation,
                self.weight.shape[2],
                self.stride,
            )
            self.output_type = {"output": np.array([self.weight.shape[0], *output_shape])}


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
        self.input_type = {"input": np.array(self.v_threshold.shape)}
        self.output_type = {"output": np.array(self.v_threshold.shape)}


@dataclass(eq=False)
class Delay(NIRNode):
    """Simple delay node.

    This node implements a simple delay:

    .. math::
        y(t) = x(t - \tau)
    """

    delay: np.ndarray  # Delay

    def __post_init__(self):
        # set input and output shape, if not set by user
        self.input_type = {"input": np.array(self.delay.shape)}
        self.output_type = {"output": np.array(self.delay.shape)}


@dataclass(eq=False)
class Flatten(NIRNode):
    """Flatten node.

    This node flattens its input tensor.
    input_type must be a dict with one key: "input".
    """

    # Shape of input tensor (overrrides input_type from
    # NIRNode to allow for non-keyword (positional) initialization)
    input_type: Types
    start_dim: int = 1  # First dimension to flatten
    end_dim: int = -1  # Last dimension to flatten

    def __post_init__(self):
        self.input_type = _parse_shape_argument(self.input_type, "input")
        concat = self.input_type["input"][self.start_dim : self.end_dim].prod()
        self.output_type = {
            "output": np.array(
                [
                    *self.input_type["input"][: self.start_dim],
                    concat,
                    *self.input_type["input"][self.end_dim :],
                ]
            )
        }
        # make sure input and output shape are valid
        if np.prod(self.input_type["input"]) != np.prod(self.output_type["output"]):
            raise ValueError("input and output shape must have same number of elements")


@dataclass(eq=False)
class I(NIRNode):  # noqa: E742
    r"""Integrator.

    The integrator neuron model is defined by the following equation:

    .. math::
        \dot{v} = R I
    """

    r: np.ndarray

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
        self.input_type = {"input": np.array(self.r.shape)}
        self.output_type = {"output": np.array(self.r.shape)}


@dataclass(eq=False)
class Input(NIRNode):
    """Input Node.

    This is a virtual node, which allows feeding in data into the graph.
    """

    # Shape of incoming data (overrrides input_type from
    # NIRNode to allow for non-keyword (positional) initialization)
    input_type: Types

    def __post_init__(self):
        self.input_type = _parse_shape_argument(self.input_type, "input")
        self.output_type = {"output": self.input_type["input"]}


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

    def __post_init__(self):
        assert (
            self.tau.shape == self.r.shape == self.v_leak.shape
        ), "All parameters must have the same shape"
        self.input_type = {"input": np.array(self.r.shape)}
        self.output_type = {"output": np.array(self.r.shape)}


@dataclass(eq=False)
class Linear(NIRNode):
    r"""Linear transform without bias:

    .. math::
        y(t) = W*x(t)
    """
    weight: np.ndarray  # Weight term

    def __post_init__(self):
        assert len(self.weight.shape) >= 2, "Weight must be at least 2D"
        self.input_type = {
            "input": np.array(
                self.weight.shape[:-2] + tuple(np.array(self.weight.shape[-1:]).T)
            )
        }
        self.output_type = {"output": self.weight.shape[:-2] + (self.weight.shape[-2],)}


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
        self.input_type = {"input": np.array(self.r.shape)}
        self.output_type = {"output": np.array(self.r.shape)}


@dataclass(eq=False)
class Output(NIRNode):
    """Output Node.

    Defines an output of the graph.
    """

    # Type of incoming data (overrrides input_type from
    # NIRNode to allow for non-keyword (positional) initialization)
    output_type: Types

    def __post_init__(self):
        self.output_type = _parse_shape_argument(self.output_type, "output")
        self.input_type = {"input": self.output_type["output"]}


@dataclass(eq=False)
class Scale(NIRNode):
    r"""Scales a signal by some values.

    This node is equivalent to the
    `Hadamard product <https://en.wikipedia.org/wiki/Hadamard_product_(matrices)>`_.

    .. math::
        y(t) = x(t) \odot s
    """

    scale: np.ndarray  # Scaling factor

    def __post_init__(self):
        self.input_type = {"input": np.array(self.scale.shape)}
        self.output_type = {"output": np.array(self.scale.shape)}


@dataclass(eq=False)
class SumPool2d(NIRNode):
    """Sum pooling layer in 2d."""

    kernel_size: np.ndarray  # (Height, Width)
    stride: np.ndarray  # (Height, width)
    padding: np.ndarray  # (Height, width)

    def __post_init__(self):
        self.input_type = {"input": ()}
        self.output_type = {"output": ()}


@dataclass(eq=False)
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
        self.input_type = {"input": np.array(self.threshold.shape)}
        self.output_type = {"output": np.array(self.threshold.shape)}
