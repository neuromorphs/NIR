from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from .conv import Conv1d, Conv2d
from .flatten import Flatten
from .node import NIRNode
from .pooling import AvgPool2d, SumPool2d
from .typing import Edges, Nodes, Types
from .utils import (
    calc_flatten_output,
    calculate_conv_output,
    ensure_str,
    parse_shape_argument,
)


@dataclass(eq=False)
class NIRGraph(NIRNode):
    """Neural Intermediate Representation (NIR) Graph containing a number of nodes and
    edges.

    A graph of computational nodes and identity edges.

    Arguments:
        nodes: Dictionary of nodes in the graph.
        edges: List of edges in the graph.
        metadata: Dictionary of metadata for the graph.
        type_check: Whether to check that input and output types match for all nodes in the graph.
            Will not be stored in the graph as an attribute. Defaults to True.
    """

    nodes: Nodes  # List of computational nodes
    edges: Edges  # List of edges between nodes
    input_type: Optional[Dict[str, np.ndarray]] = None
    output_type: Optional[Dict[str, np.ndarray]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        nodes: Nodes,
        edges: Edges,
        input_type: Optional[Dict[str, np.ndarray]] = None,
        output_type: Optional[Dict[str, np.ndarray]] = None,
        metadata: Dict[str, Any] = dict,
        type_check: bool = True,
    ):
        self.nodes = nodes
        self.edges = edges
        self.metadata = metadata
        self.input_type = input_type
        self.output_type = output_type

        # Check that all nodes have input and output types, if requested (default)
        if type_check:
            self.infer_types()
            self.check_types()

        # Call post init to set input_type and output_type
        self.__post_init__()

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
    def from_list(*nodes: NIRNode, type_check: bool = True) -> "NIRGraph":
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
        if not isinstance(nodes[0], Input):
            node_dict = {"input": Input(input_type=nodes[0].input_type)}
        else:
            node_dict = {}
        edges = []

        for node in nodes:
            name = unique_node_name(node, counts)
            node_dict[name] = node

        if not isinstance(nodes[-1], Output):
            node_dict["output"] = Output(output_type=nodes[-1].output_type)

        names = list(node_dict)
        for i in range(len(names) - 1):
            edges.append((names[i], names[i + 1]))

        return NIRGraph(
            nodes=node_dict,
            edges=edges,
            type_check=type_check,
        )

    def __post_init__(self):
        self._update_input_output_types()

        # Assign the metadata attribute if left unset to avoid issues with serialization
        if not isinstance(self.metadata, dict):
            self.metadata = {}

    def _update_input_output_types(self):
        input_node_keys = [
            k for k, node in self.nodes.items() if isinstance(node, Input)
        ]
        self.input_type = (
            {
                node_key: self.nodes[node_key].input_type["input"]
                for node_key in input_node_keys
            }
            if len(input_node_keys) > 0
            else None
        )
        output_node_keys = [
            k for k, node in self.nodes.items() if isinstance(node, Output)
        ]
        self.output_type = {
            node_key: self.nodes[node_key].output_type["output"]
            for node_key in output_node_keys
        }

    def to_dict(self) -> Dict[str, Any]:
        ret = super().to_dict()
        ret["nodes"] = {k: n.to_dict() for k, n in self.nodes.items()}
        return ret

    @classmethod
    def from_dict(cls, kwargs: Dict[str, Any]) -> "NIRGraph":
        from . import dict2NIRNode

        kwargs_local = kwargs.copy()  # Copy the input to avoid overwriting attributes

        # Assert that we have nodes and edges
        assert "nodes" in kwargs, "The incoming dictionary must hade a 'nodes' entry"
        assert "edges" in kwargs, "The incoming dictionary must hade a 'edges' entry"
        # Assert that the type is well-formed
        if "type" in kwargs:
            assert (
                kwargs["type"] == "NIRGraph"
            ), "You are calling NIRGraph.from_dict with a different type "
            f"{type}. Either remove the entry or use <Specific NIRNode>.from_dict, such as Input.from_dict"
        kwargs_local["type"] = "NIRGraph"

        kwargs_local["nodes"] = {
            k: dict2NIRNode(n) for k, n in kwargs_local["nodes"].items()
        }
        # h5py deserializes edges into a numpy array of type bytes and dtype=object,
        # hence using ensure_str here
        kwargs_local["edges"] = [
            (ensure_str(a), ensure_str(b)) for a, b in kwargs_local["edges"]
        ]
        return super().from_dict(kwargs_local)

    def check_types(self):
        """Check that all nodes in the graph have input and output types.

        Will raise ValueError if any node has no input or output type, or if the types
        are inconsistent.
        """
        for edge in self.edges:
            pre_node = self.nodes[edge[0]]
            post_node = self.nodes[edge[1]]

            if isinstance(post_node, NIRGraph):
                post_node.check_types()

            # make sure all types are defined
            undef_out_type = pre_node.output_type is None or any(
                v is None for v in pre_node.output_type.values()
            )
            if undef_out_type:
                raise ValueError(f"pre node {edge[0]} has no output type")
            undef_in_type = post_node.input_type is None or any(
                v is None for v in post_node.input_type.values()
            )
            if undef_in_type:
                raise ValueError(f"post node {edge[1]} has no input type")

            # make sure the length of types is equal
            if len(pre_node.output_type) != len(post_node.input_type):
                pre_repr = f"len({edge[0]}.output)={len(pre_node.output_type)}"
                post_repr = f"len({edge[1]}.input)={len(post_node.input_type)}"
                raise ValueError(f"type length mismatch: {pre_repr} -> {post_repr}")

            # make sure the type values match up
            if len(pre_node.output_type.keys()) == 1:
                post_input_type = list(post_node.input_type.values())[0]
                pre_output_type = list(pre_node.output_type.values())[0]
                if not np.array_equal(post_input_type, pre_output_type):
                    pre_repr = f"{edge[0]}.output: {pre_output_type}"
                    post_repr = f"{edge[1]}.input: {post_input_type}"
                    raise ValueError(f"type mismatch: {pre_repr} -> {post_repr}")
            else:
                raise NotImplementedError(
                    "multiple input/output types not supported yet"
                )
        return True

    def infer_types(self):
        """Infer the types of all nodes in this graph. Will modify the input_type and
        output_type of nodes in the graph as needed. Assumes that the input_type of the
        graph is set. Moves from the input nodes to the output nodes. Raises ValueError
        if types are inconsistent.

        Assumes that all input types are of form: {'input': ...} and all output types are of form:
        {'output': ...}.

        Currently supports the inference of output types for Conv1d, Conv2d, SumPool2d,
        AvgPool2d, and Flatten nodes.
        """
        if not self.nodes:
            return

        # Ensure all graph inputs flow through an Input node
        all_node_keys = set(self.nodes.keys())
        destination_nodes = {edge[1] for edge in self.edges}
        root_nodes = all_node_keys - destination_nodes

        new_nodes: Dict[str, NIRNode] = {}
        new_edges: Edges = []

        for node_key in root_nodes:
            node = self.nodes[node_key]
            if not isinstance(node, Input):
                # This is a root node that is not an Input node.
                # It must have its input_type defined to create a preceding Input node.
                undef_input_type = node.input_type is None or any(
                    v is None for v in node.input_type.values()
                )
                if undef_input_type:
                    raise ValueError(
                        f"Root node '{node_key}' of type {type(node).__name__} is not an "
                        f"Input node and has no defined input_type. Cannot infer graph input."
                    )

                # Prepend an Input node
                input_node_name = f"input_{node_key}"
                i = 0
                original_name = input_node_name
                while input_node_name in self.nodes or input_node_name in new_nodes:
                    input_node_name = f"{original_name}_{i}"
                    i += 1

                new_input_node = Input(input_type=node.input_type)
                new_nodes[input_node_name] = new_input_node
                new_edges.append((input_node_name, node_key))
            else:
                undef_input_type = node.input_type is None or any(
                    v is None for v in node.input_type.values()
                )
                if undef_input_type:
                    raise ValueError(
                        f"Input node '{node_key}' has no defined input_type. Cannot infer graph without input types."
                    )

        if new_nodes:
            self.nodes.update(new_nodes)
            self.edges.extend(new_edges)

        # Start type inference from input nodes
        ready = [e for e in self.edges if e[0] in self.inputs.keys()]
        if len(ready) == 0:
            raise ValueError(
                "Failed to start type inference: No input nodes found. "
                "This may be due to a cyclic dependency at the graph's input. "
                "Please add an `Input` node manually to define an entry point, "
                "or disable type checking (`type_check=False`)."
            )

        seen = set([e[0] for e in ready])
        while len(ready) > 0:
            pre_key, post_key = ready.pop()
            pre_node = self.nodes[pre_key]
            post_node = self.nodes[post_key]

            if isinstance(post_node, NIRGraph):
                post_node.infer_types()

            # check if post input_type needs to be defined
            undef_post_input_type = post_node.input_type is None or any(
                v is None for v in post_node.input_type.values()
            )
            type_mismatch = any(
                [
                    len(post_node.input_type) != len(pre_node.output_type),
                    not np.array_equal(
                        np.array(list(pre_node.output_type.values())),
                        np.array(list(post_node.input_type.values())),
                    ),
                ]
            )
            if undef_post_input_type:
                # define post input_type to be the same as pre output_type
                post_node.input_type = {
                    k.replace("output", "input"): v
                    for k, v in pre_node.output_type.items()
                }
            elif type_mismatch:
                # set post input_type to be the same as pre output_type
                pre_repr = (
                    f"{pre_key}.output: {np.array(list(pre_node.output_type.values()))}"
                )
                post_repr = (
                    f"{post_key}.input: {np.array(list(post_node.input_type.values()))}"
                )
                raise ValueError(
                    f"Type inference error: type mismatch: {pre_repr} -> {post_repr}"
                )

            # make sure that output nodes have output_type = input_type
            if isinstance(post_node, Output):
                post_node.output_type = {
                    k.replace("input", "output"): v
                    for k, v in post_node.input_type.items()
                }

            # check if post output_type needs to be defined
            undef_post_output_type = post_node.output_type is None or any(
                v is None for v in post_node.output_type.values()
            )
            if undef_post_output_type:
                # define post output_type
                if isinstance(post_node, Conv1d) or isinstance(post_node, Conv2d):
                    if isinstance(post_node, Conv1d):
                        post_node.input_shape = post_node.input_type["input"][1]
                    else:
                        post_node.input_shape = tuple(post_node.input_type["input"][1:])
                    output_shape = calculate_conv_output(
                        post_node.input_shape,
                        post_node.padding,
                        post_node.dilation,
                        post_node.weight.shape[2],
                        post_node.stride,
                    )
                    output_type = np.array([post_node.weight.shape[0], *output_shape])
                    post_node.output_type = {"output": output_type}

                elif isinstance(post_node, SumPool2d):
                    output_shape = calculate_conv_output(
                        pre_node.output_type["output"][1:],
                        post_node.padding,
                        1,
                        post_node.kernel_size,
                        post_node.stride,
                    )
                    output_type = np.array(
                        [post_node.input_type["input"][0], *output_shape]
                    )
                    post_node.output_type = {"output": output_type}

                elif isinstance(post_node, AvgPool2d):
                    output_shape = calculate_conv_output(
                        pre_node.output_type["output"][1:],
                        post_node.padding,
                        1,
                        post_node.kernel_size,
                        post_node.stride,
                    )
                    output_type = np.array(
                        [post_node.input_type["input"][0], *output_shape]
                    )
                    post_node.output_type = {"output": output_type}

                elif isinstance(post_node, Flatten):
                    post_node.output_type = {
                        "output": calc_flatten_output(
                            post_node.input_type["input"],
                            post_node.start_dim,
                            post_node.end_dim,
                        )
                    }
                    n_inputs = np.prod(post_node.input_type["input"])
                    n_outputs = np.prod(post_node.output_type["output"])
                    assert (
                        n_inputs == n_outputs
                    ), "Flatten must not change the number of elements"

            seen.add(post_key)
            ready += [e for e in self.edges if e[0] == post_key and e[1] not in seen]

        # Ensure all graph outputs flow through an Output node
        all_node_keys = set(self.nodes.keys())
        source_nodes = {edge[0] for edge in self.edges}
        leaf_nodes = all_node_keys - source_nodes

        if not leaf_nodes:
            raise ValueError(
                "Type inference failed: No output nodes found. "
                "This may be due to a cyclic dependency at the graph's output. "
                "Please add an `Output` node manually to define an exit point, "
                "or disable type checking (`type_check=False`)."
            )

        new_nodes: Dict[str, NIRNode] = {}
        new_edges: Edges = []

        for node_key in leaf_nodes:
            node = self.nodes[node_key]
            if not isinstance(node, Output):
                # This is a leaf node that is not an Output node.
                # It must have its output_type defined to create a succeeding Output
                # node.
                undef_output_type = node.output_type is None or any(
                    v is None for v in node.output_type.values()
                )
                if undef_output_type:
                    # This should not happen if type inference was successful
                    raise ValueError(
                        f"Leaf node '{node_key}' of type {type(node).__name__} "
                        "is not an Output node and has no defined output_type. "
                        "Cannot infer graph output."
                    )

                # Append an Output node
                output_node_name = f"output_{node_key}"
                i = 0
                original_name = output_node_name
                while output_node_name in self.nodes or output_node_name in new_nodes:
                    output_node_name = f"{original_name}_{i}"
                    i += 1

                new_output_node = Output(output_type=node.output_type)
                new_nodes[output_node_name] = new_output_node
                new_edges.append((node_key, output_node_name))

        if new_nodes:
            self.nodes.update(new_nodes)
            self.edges.extend(new_edges)

        self._update_input_output_types()


@dataclass(eq=False)
class Input(NIRNode):
    """Input Node.

    This is a virtual node, which allows feeding in data into the graph.
    """

    # Shape of incoming data (overrrides input_type from
    # NIRNode to allow for non-keyword (positional) initialization)
    input_type: Types
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.input_type = parse_shape_argument(self.input_type, "input")
        self.output_type = {"output": self.input_type["input"]}

    def to_dict(self) -> Dict[str, Any]:
        ret = super().to_dict()
        ret["shape"] = self.input_type["input"]
        return ret

    @classmethod
    def from_dict(cls, node: Dict[str, Any]) -> "NIRNode":
        node["input_type"] = {"input": node["shape"]}
        del node["shape"]
        return super().from_dict(node)


@dataclass(eq=False)
class Output(NIRNode):
    """Output Node.

    Defines an output of the graph.
    """

    # Type of incoming data (overrrides input_type from
    # NIRNode to allow for non-keyword (positional) initialization)
    output_type: Types
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.output_type = parse_shape_argument(self.output_type, "output")
        self.input_type = {"input": self.output_type["output"]}

    def to_dict(self) -> Dict[str, Any]:
        ret = super().to_dict()
        ret["shape"] = self.output_type["output"]
        return ret

    @classmethod
    def from_dict(cls, node: Dict[str, Any]) -> "NIRNode":
        node["output_type"] = {"output": node["shape"]}
        del node["shape"]
        return super().from_dict(node)


@dataclass(eq=False)
class Identity(NIRNode):
    """Identity Node.

    This is a virtual node, which allows for the identity operation.
    """

    input_type: Types

    def __post_init__(self):
        self.output_type = self.input_type

    def to_dict(self) -> Dict[str, Any]:
        ret = super().to_dict()
        ret["shape"] = self.output_type["output"]
        return ret

    @classmethod
    def from_dict(cls, node: Dict[str, Any]) -> "NIRNode":
        return super().from_dict(node)
