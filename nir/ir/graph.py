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
    """

    nodes: Nodes  # List of computational nodes
    edges: Edges  # List of edges between nodes
    input_type: Optional[Dict[str, np.ndarray]] = None
    output_type: Optional[Dict[str, np.ndarray]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

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

    def to_dict(self) -> Dict[str, Any]:
        ret = super().to_dict()
        ret["nodes"] = {k: n.to_dict() for k, n in self.nodes.items()}
        return ret

    @classmethod
    def from_dict(cls, node: Dict[str, Any]) -> "NIRNode":
        from . import dict2NIRNode

        node["nodes"] = {k: dict2NIRNode(n) for k, n in node["nodes"].items()}
        # h5py deserializes edges into a numpy array of type bytes and dtype=object,
        # hence using ensure_str here
        node["edges"] = [(ensure_str(a), ensure_str(b)) for a, b in node["edges"]]
        return super().from_dict(node)

    def _check_types(self):
        """Check that all nodes in the graph have input and output types.

        Will raise ValueError if any node has no input or output type, or if the types
        are inconsistent.
        """
        for edge in self.edges:
            pre_node = self.nodes[edge[0]]
            post_node = self.nodes[edge[1]]

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

    def _forward_type_inference(self, debug=True):
        """Infer the types of all nodes in this graph. Will modify the input_type and
        output_type of nodes in the graph as needed. Assumes that the input_type of the
        graph is set. Moves from the input nodes to the output nodes. Raises ValueError
        if types are inconsistent.

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
                raise NotImplementedError(
                    "type inference on nested NIR graphs not supported yet"
                )

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
                print(
                    f"[warning] {post_key}.input_type undefined, set to {pre_key}.output_type"
                )
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
                print(f"[warning] overwriting {post_repr} with {pre_repr}")
                post_node.input_type = {
                    k.replace("output", "input"): v
                    for k, v in pre_node.output_type.items()
                }

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

            seen.add(post_key)
            ready += [e for e in self.edges if e[0] == post_key and e[1] not in seen]

            self.nodes[pre_key] = pre_node
            self.nodes[post_key] = post_node

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
            raise NotImplementedError(
                "backward-mode type inference not implemented yet"
            )
        else:
            raise ValueError("Either input_type or output_type must be set")

    def _check_types(self):
        """Check that all nodes in the graph have input and output types.

        Will raise ValueError if any node has no input or output type, or if the types
        are inconsistent.
        """
        for edge in self.edges:
            pre_node = self.nodes[edge[0]]
            post_node = self.nodes[edge[1]]

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

    def _forward_type_inference(self, debug=True):
        """Infer the types of all nodes in this graph. Will modify the input_type and
        output_type of nodes in the graph as needed. Assumes that the input_type of the
        graph is set. Moves from the input nodes to the output nodes. Raises ValueError
        if types are inconsistent.

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
                raise NotImplementedError(
                    "type inference on nested NIR graphs not supported yet"
                )

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
                print(
                    f"[warning] {post_key}.input_type undefined, set to {pre_key}.output_type"
                )
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
                print(f"[warning] overwriting {post_repr} with {pre_repr}")
                post_node.input_type = {
                    k.replace("output", "input"): v
                    for k, v in pre_node.output_type.items()
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
                    print("updateing flatten output")
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


@dataclass(eq=False)
class Input(NIRNode):
    """Input Node.

    This is a virtual node, which allows feeding in data into the graph.
    """

    # Shape of incoming data (overrrides input_type from
    # NIRNode to allow for non-keyword (positional) initialization)
    input_type: Types

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
