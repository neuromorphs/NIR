import pathlib
import typing

import h5py
import numpy as np

import nir


def write(filename: typing.Union[str, pathlib.Path], graph: nir.typing.NIRNode) -> None:
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
        f.create_dataset("version", data=nir.version)
        node_group = f.create_group("node")
        write_recursive(node_group, graph.to_dict())
