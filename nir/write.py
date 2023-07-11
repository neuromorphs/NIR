import typing
import pathlib

import h5py
import numpy as np

import nir


def _convert_node(node: nir.NIRNode) -> dict:
    if isinstance(node, nir.Affine):
        return {
            "type": "Affine",
            "weight": node.weight,
            "bias": node.bias,
        }
    elif isinstance(node, nir.Conv1d):
        return {
            "type": "Conv1d",
            "weight": node.weight,
            "stride": node.stride,
            "padding": node.padding,
            "dilation": node.dilation,
            "groups": node.groups,
            "bias": node.bias,
        }
    elif isinstance(node, nir.Conv2d):
        return {
            "type": "Conv2d",
            "weight": node.weight,
            "stride": node.stride,
            "padding": node.padding,
            "dilation": node.dilation,
            "groups": node.groups,
            "bias": node.bias,
        }
    elif isinstance(node, nir.Delay):
        return {"type": "Delay", "delay": node.delay}
    elif isinstance(node, nir.I):
        return {"type": "I", "r": node.r}
    elif isinstance(node, nir.IF):
        return {"type": "IF", "r": node.r, "v_threshold": node.v_threshold}
    elif isinstance(node, nir.Input):
        return {
            "type": "Input",
            "shape": node.shape,
        }
    elif isinstance(node, nir.LI):
        return {
            "type": "LI",
            "tau": node.tau,
            "r": node.r,
            "v_leak": node.v_leak,
        }
    elif isinstance(node, nir.Linear):
        return {
            "type": "Linear",
            "weight": node.weight,
        }
    elif isinstance(node, nir.LIF):
        return {
            "type": "LIF",
            "tau": node.tau,
            "r": node.r,
            "v_leak": node.v_leak,
            "v_threshold": node.v_threshold,
        }
    elif isinstance(node, nir.CubaLIF):
        return {
            "type": "CubaLIF",
            "tau_mem": node.tau_mem,
            "tau_syn": node.tau_syn,
            "r": node.r,
            "v_leak": node.v_leak,
            "v_threshold": node.v_threshold,
        }
    elif isinstance(node, nir.NIRGraph):
        return {
            "type": "NIRGraph",
            "nodes": {i: _convert_node(n) for i, n in enumerate(node.nodes)},
            "edges": node.edges,
        }
    elif isinstance(node, nir.Output):
        return {
            "type": "Output",
        }
    elif isinstance(node, nir.Threshold):
        return {"type": "Threshold", "threshold": node.threshold}
    else:
        raise ValueError(f"Unknown node type: {node}")


def write(filename: typing.Union[str, pathlib.Path], graph: nir.NIRNode) -> None:
    """Write a NIR to a HDF5 file."""

    def write_recursive(group: h5py.Group, node: dict) -> None:
        for k, v in node.items():
            if isinstance(v, str):
                group.create_dataset(k, data=v, dtype=h5py.string_dtype())
            elif isinstance(v, np.ndarray):
                group.create_dataset(k, data=v, dtype=v.dtype)
            elif isinstance(v, dict):
                write_recursive(group.create_group(str(k)), v)
            else:
                group.create_dataset(k, data=v)

    with h5py.File(filename, "w") as f:
        node_group = f.create_group("node")
        write_recursive(node_group, _convert_node(graph))
