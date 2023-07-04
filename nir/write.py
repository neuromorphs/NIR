import typing
import pathlib

import h5py
import numpy as np

import nir


def write(filename: typing.Union[str, pathlib.Path], graph: nir.NIR) -> None:
    def convert_node(node: nir.NIRNode) -> dict:
        if isinstance(node, nir.Input):
            return {
                "type": "Input",
                "shape": node.shape,
            }
        elif isinstance(node, nir.Output):
            return {
                "type": "Output",
            }
        elif isinstance(node, nir.LI):
            return {
                "type": "LI",
                "tau": node.tau,
                "r": node.r,
                "v_leak": node.v_leak,
            }
        elif isinstance(node, nir.LIF):
            return {
                "type": "LIF",
                "tau": node.tau,
                "r": node.r,
                "v_leak": node.v_leak,
                "v_th": node.v_th,
            }
        elif isinstance(node, nir.Linear):
            return {
                "type": "Linear",
                "weights": node.weights,
                "bias": node.bias,
            }
        elif isinstance(node, nir.Conv1d):
            return {
                "type": "Conv1d",
                "weights": node.weights,
                "stride": node.stride,
                "padding": node.padding,
                "dilation": node.dilation,
                "groups": node.groups,
                "bias": node.bias,
            }
        elif isinstance(node, nir.Conv2d):
            return {
                "type": "Conv2d",
                "weights": node.weights,
                "stride": node.stride,
                "padding": node.padding,
                "dilation": node.dilation,
                "groups": node.groups,
                "bias": node.bias,
            }
        else:
            raise ValueError(f"Unknown node type: {node}")

    """Write a NIR to a HDF5 file."""
    with h5py.File(filename, "w") as f:
        nodes_group = f.create_group("nodes")
        for i, node in enumerate(graph.nodes):
            d = convert_node(node)
            node_group = nodes_group.create_group(str(i))
            for k, v in d.items():
                if isinstance(v, str):
                    node_group.create_dataset(k, data=v, dtype=h5py.string_dtype())
                elif isinstance(v, np.ndarray):
                    node_group.create_dataset(k, data=v, dtype=v.dtype)
                else:
                    node_group.create_dataset(k, data=v)
                    # raise ValueError(f"Unknown type: {type(v)}")
        f.create_dataset("edges", data=graph.edges)
