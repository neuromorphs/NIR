import io
import pathlib
from typing import Any, Dict, Union

import h5py
import numpy as np

import nir


def _read_metadata(node: Any) -> Dict[str, Any]:
    if "metadata" in node.keys():
        return {"metadata": {k: v[()] for k, v in node["metadata"].items()}}
    else:
        return {}


def try_byte_to_str(a: Union[bytes, Any]) -> Union[str, Any]:
    return a.decode("utf8") if isinstance(a, bytes) else a


def read_node(node: Any) -> nir.NIRNode:
    """Read a graph from a HDF5 file."""
    if node["type"][()] == b"Affine":
        return nir.Affine(
            weight=node["weight"][()], bias=node["bias"][()], **_read_metadata(node)
        )
    elif node["type"][()] == b"Conv1d":
        return nir.Conv1d(
            input_shape=(
                node["input_shape"][()] if "input_shape" in node.keys() else None
            ),
            weight=node["weight"][()],
            stride=node["stride"][()],
            padding=node["padding"][()],
            dilation=node["dilation"][()],
            groups=node["groups"][()],
            bias=node["bias"][()],
            **_read_metadata(node),
        )
    elif node["type"][()] == b"Conv2d":
        return nir.Conv2d(
            input_shape=(
                node["input_shape"][()] if "input_shape" in node.keys() else None
            ),
            weight=node["weight"][()],
            stride=node["stride"][()],
            padding=node["padding"][()],
            dilation=node["dilation"][()],
            groups=node["groups"][()],
            bias=node["bias"][()],
            **_read_metadata(node),
        )
    elif node["type"][()] == b"SumPool2d":
        return nir.SumPool2d(
            kernel_size=node["kernel_size"][()],
            stride=node["stride"][()],
            padding=node["padding"][()],
            **_read_metadata(node),
        )
    elif node["type"][()] == b"AvgPool2d":
        return nir.AvgPool2d(
            kernel_size=node["kernel_size"][()],
            stride=node["stride"][()],
            padding=node["padding"][()],
        )
    elif node["type"][()] == b"Delay":
        return nir.Delay(delay=node["delay"][()])
    elif node["type"][()] == b"Flatten":
        return nir.Flatten(
            start_dim=node["start_dim"][()],
            end_dim=node["end_dim"][()],
            input_type={
                "input": node["input_type"][()] if "input_type" in node.keys() else None
            },
            **_read_metadata(node),
        )
    elif node["type"][()] == b"I":
        return nir.I(r=node["r"][()], **_read_metadata(node))
    elif node["type"][()] == b"IF":
        return nir.IF(
            r=node["r"][()],
            v_reset=(
                node["v_reset"][()]
                if "v_reset" in node
                else np.zeros_like(node["v_threshold"][()])
            ),
            v_threshold=node["v_threshold"][()],
            **_read_metadata(node),
        )
    elif node["type"][()] == b"Input":
        return nir.Input(
            input_type={"input": node["shape"][()]}, **_read_metadata(node)
        )
    elif node["type"][()] == b"LI":
        return nir.LI(
            tau=node["tau"][()],
            r=node["r"][()],
            v_leak=node["v_leak"][()],
            **_read_metadata(node),
        )
    elif node["type"][()] == b"Linear":
        return nir.Linear(weight=node["weight"][()], **_read_metadata(node))
    elif node["type"][()] == b"LIF":
        return nir.LIF(
            tau=node["tau"][()],
            r=node["r"][()],
            v_leak=node["v_leak"][()],
            v_reset=(
                node["v_reset"][()]
                if "v_reset" in node
                else np.zeros_like(node["v_threshold"][()])
            ),
            v_threshold=node["v_threshold"][()],
            **_read_metadata(node),
        )
    elif node["type"][()] == b"CubaLI":
        return nir.CubaLI(
            tau_mem=node["tau_mem"][()],
            tau_syn=node["tau_syn"][()],
            r=node["r"][()],
            v_leak=node["v_leak"][()],
            w_in=node["w_in"][()],
            **_read_metadata(node),
        )
    elif node["type"][()] == b"CubaLIF":
        return nir.CubaLIF(
            tau_mem=node["tau_mem"][()],
            tau_syn=node["tau_syn"][()],
            r=node["r"][()],
            v_leak=node["v_leak"][()],
            v_reset=(
                node["v_reset"][()]
                if "v_reset" in node
                else np.zeros_like(node["v_threshold"][()])
            ),
            v_threshold=node["v_threshold"][()],
            w_in=node["w_in"][()],
            **_read_metadata(node),
        )
    elif node["type"][()] == b"NIRGraph":
        return nir.NIRGraph(
            nodes={k: read_node(n) for k, n in node["nodes"].items()},
            edges=[(a.decode("utf8"), b.decode("utf8")) for a, b in node["edges"][()]],
            **_read_metadata(node),
        )
    elif node["type"][()] == b"Output":
        return nir.Output(
            output_type={"output": node["shape"][()]}, **_read_metadata(node)
        )
    elif node["type"][()] == b"Scale":
        return nir.Scale(scale=node["scale"][()], **_read_metadata(node))
    elif node["type"][()] == b"Threshold":
        return nir.Threshold(threshold=node["threshold"][()], **_read_metadata(node))
    else:
        raise ValueError(f"Unknown unit type: {node['type'][()]}")


def hdf2dict(node: Any) -> Dict[str, Any]:
    ret = {}

    def read_hdf_to_dict(node, data_dict):
        for key, item in node.items():
            key = try_byte_to_str(key)
            if isinstance(item, h5py.Group):
                data_dict[key] = {}
                read_hdf_to_dict(item, data_dict[key])
            elif isinstance(item, h5py.Dataset):
                item = try_byte_to_str(item[()])
                data_dict[key] = item

    read_hdf_to_dict(node, ret)
    return ret


def read(filename: Union[str, pathlib.Path]) -> nir.NIRGraph:
    """Load a NIR from a HDF/conn5 file.
    Attempts to read a NIRGraph from a file and pass in the key-value parameters to the
    corresponding NIR nodes.
    If either the reading or creation of nodes fail, the function will raise an exception.

    Arguments:
        filename (Union[str, Path]): The filename as either a string or pathlib Path.

    Returns:
        A NIRGraph read from the file.
    """
    with h5py.File(filename, "r") as f:
        data_dict = hdf2dict(f["node"])
        return nir.dict2NIRNode(data_dict)


def read_version(filename: Union[str, pathlib.Path]) -> str:
    """Reads the filename of a given NIR file, and raises an exception if the version
    does not exist in the file.

    Arguments:
        filename (Union[str, Path]): The filename as either a string or pathlib Path.
    """
    with h5py.File(filename, "r") as f:
        return f["version"][()].decode("utf8")


def write(filename: Union[str, pathlib.Path, io.RawIOBase], graph: nir.NIRNode) -> None:
    """Write a NIR to a HDF5 file.

    Arguments:
        filename (Union[str, Path, io.RawIOBase]): The filename as either a string, pathlib Path,
            or io.RawIOBase. In the case of a string or path, the function will attempt to open
            the file and write the bytes to it. In the case of an IOBase, the bytes will be
            written directly to the IOBase.
        graph (nir.NIRNode): The NIR Graph to serialize.
    """

    def write_recursive(group: h5py.Group, node: dict) -> None:
        for k, v in node.items():
            if k == "metadata":
                if not v == {}:  # Skip metadata if empty
                    write_recursive(group.create_group(k), v)
            elif isinstance(v, str):
                group.create_dataset(k, data=v, dtype=h5py.string_dtype())
            elif isinstance(v, np.ndarray):
                group.create_dataset(k, data=v, dtype=v.dtype)
            elif isinstance(v, dict):
                write_recursive(group.create_group(str(k)), v)
            else:
                group.create_dataset(k, data=v)

    with h5py.File(filename, "w") as f:
        f.create_dataset("version", data=nir.version)
        node_group = f.create_group("node")
        write_recursive(node_group, graph.to_dict())
