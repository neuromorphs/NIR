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


def read(filename: Union[str, pathlib.Path], type_check: bool = True) -> nir.NIRGraph:
    """Load a NIR from a HDF/conn5 file.
    Attempts to read a NIRGraph from a file and pass in the key-value parameters to the
    corresponding NIR nodes.
    If either the reading or creation of nodes fail, the function will raise an exception.

    Arguments:
        filename (Union[str, Path]): The filename as either a string or pathlib Path.
        type_check (bool): Whether to type check the graph and verify that all input types align with output types
                for every node in the graph. May break the loading for older graphs. Defaults to True.

    Returns:
        A NIRGraph read from the file.
    """
    with h5py.File(filename, "r") as f:
        data_dict = hdf2dict(f["node"])
        if hasattr(data_dict, "type_check"):
            raise ValueError(
                "The 'type_check' key was found in the read NIR graph, but is unsupported and clashes with the type checking parameter in the read function"
            )
        else:
            data_dict["type_check"] = type_check
        return nir.dict2NIRNode(data_dict)


def read_version(filename: Union[str, pathlib.Path]) -> str:
    """Reads the filename of a given NIR file, and raises an exception if the version
    does not exist in the file.

    Arguments:
        filename (Union[str, Path]): The filename as either a string or pathlib Path.
    """
    with h5py.File(filename, "r") as f:
        return f["version"][()].decode("utf8")


def write(
    filename: Union[str, pathlib.Path, io.RawIOBase],
    graph: nir.NIRNode,
    compression: str = "gzip",
    compression_opts: Any = None,
) -> None:
    """Write a NIR to a HDF5 file.

    Arguments:
        filename (Union[str, Path, io.RawIOBase]): The filename as either a string, pathlib Path,
            or io.RawIOBase. In the case of a string or path, the function will attempt to open
            the file and write the bytes to it. In the case of an IOBase, the bytes will be
            written directly to the IOBase.
        graph (nir.NIRNode): The NIR Graph to serialize.
        compression (str or int): The compression strategy to use when writing the HDF5 file. Defaults to "gzip".
            Legal values are 'gzip', 'szip', 'lzf'.  If an integer in range(10), this indicates gzip
            compression level.
            See the [h5py documentation](https://docs.h5py.org/en/stable/high/dataset.html#lossless-compression-filters) for more details.
        compression_opts
            Compression settings.  This is an integer for gzip, 2-tuple for
            szip, etc. If specifying a dynamically loaded compression filter
            number, this must be a tuple of values.
            See the [h5py documentation](https://docs.h5py.org/en/stable/high/dataset.html#lossless-compression-filters) for more details.
    """

    def write_recursive(group: h5py.Group, node: dict) -> None:
        for k, v in node.items():
            if k == "metadata":
                if not v == {}:  # Skip metadata if empty
                    write_recursive(group.create_group(k), v)
            elif isinstance(v, str):
                group.create_dataset(k, data=v, dtype=h5py.string_dtype())
            elif isinstance(v, np.ndarray):
                group.create_dataset(
                    k,
                    data=v,
                    dtype=v.dtype,
                    compression=compression,
                    compression_opts=compression_opts,
                )
            elif isinstance(v, dict):
                write_recursive(group.create_group(str(k)), v)
            else:
                group.create_dataset(k, data=v)

    with h5py.File(filename, "w") as f:
        f.create_dataset("version", data=nir.version, dtype=h5py.string_dtype())
        node_group = f.create_group("node")
        write_recursive(node_group, graph.to_dict())


def read_data(path: str) -> nir.NIRGraphData:
    """Load NIRGraphData from a HDF/conn5 file.

    Arguments:
        filename (Union[str, Path]): The filename as either a string or pathlib Path.

    Returns:
        NIRGraphData read from the file.
    """

    def _read_time_gridded(group: h5py.Group) -> nir.TimeGriddedData:
        data = group["data"][()]
        dt = float(group.attrs["dt"])
        return nir.TimeGriddedData(data=data, dt=dt)

    def _read_event_data(group: h5py.Group) -> nir.EventData:
        idx = group["idx"][()]
        time = group["time"][()]
        n_neurons = int(group.attrs["n_neurons"])
        t_max = float(group.attrs["t_max"])

        if "__type__" in group.attrs and group.attrs["__type__"] == "ValuedEventData":
            value = group["value"][()]
            return nir.ValuedEventData(
                idx=idx,
                time=time,
                value=value,
                n_neurons=n_neurons,
                t_max=t_max,
            )

        return nir.EventData(
            idx=idx,
            time=time,
            n_neurons=n_neurons,
            t_max=t_max,
        )

    def _read_node_data(group: h5py.Group) -> nir.NIRNodeData:
        observables = {}
        obs_group = group["observables"]

        for name, g in obs_group.items():
            typ = g.attrs["__type__"]
            if typ == "TimeGriddedData":
                observables[name] = _read_time_gridded(g)
            elif typ in ("EventData", "ValuedEventData"):
                observables[name] = _read_event_data(g)
            else:
                raise ValueError(f"Unknown observable type: {typ}")
        return nir.NIRNodeData(observables=observables)

    def _read_graph_data(group: h5py.Group) -> nir.NIRGraphData:
        nodes = {}
        nodes_group = group["nodes"]

        for name, g in nodes_group.items():
            typ = g.attrs["__type__"]
            if typ == "NIRNodeData":
                nodes[name] = _read_node_data(g)
            elif typ == "NIRGraphData":
                nodes[name] = _read_graph_data(g)
            else:
                raise ValueError(f"Unknown node type: {typ}")

        return nir.NIRGraphData(nodes=nodes)

    with h5py.File(path, "r") as f:
        if f.attrs.get("__type__") != "NIRGraphData":
            raise ValueError("Root object is not NIRGraphData")
        return _read_graph_data(f)


def write_data(
    filename: Union[str, pathlib.Path, io.RawIOBase],
    graph_data: nir.NIRGraphData,
    compression: str = "gzip",
    compression_opts: Any = None,
) -> None:
    """Write a NIRDataGraph to a HDF5 file.

    Arguments:
        filename (Union[str, Path, io.RawIOBase]): The filename as either a string, pathlib Path,
            or io.RawIOBase. In the case of a string or path, the function will attempt to open
            the file and write the bytes to it. In the case of an IOBase, the bytes will be
            written directly to the IOBase.
        graph_data (nir.NIRGraphData): The data of a NIR graph to serialize.
        compression (str or int): The compression strategy to use when writing the HDF5 file. Defaults to "gzip".
            Legal values are 'gzip', 'szip', 'lzf'.  If an integer in range(10), this indicates gzip
            compression level.
            See the [h5py documentation](https://docs.h5py.org/en/stable/high/dataset.html#lossless-compression-filters) for more details.
        compression_opts
            Compression settings.  This is an integer for gzip, 2-tuple for
            szip, etc. If specifying a dynamically loaded compression filter
            number, this must be a tuple of values.
            See the [h5py documentation](https://docs.h5py.org/en/stable/high/dataset.html#lossless-compression-filters) for more details.
    """

    def _write_time_gridded(group: h5py.Group, data: nir.TimeGriddedData):
        group.attrs["__type__"] = "TimeGriddedData"
        group.create_dataset(
            "data",
            data=data.data,
            dtype=data.data.dtype,
            compression=compression,
            compression_opts=compression_opts,
        )
        group.attrs["dt"] = data.dt

    def _write_event_data(group: h5py.Group, data: nir.EventData):
        group.attrs["__type__"] = type(data).__name__
        group.create_dataset(
            "idx",
            data=data.idx,
            dtype=data.idx.dtype,
            compression=compression,
            compression_opts=compression_opts,
        )
        group.create_dataset(
            "time",
            data=data.time,
            dtype=data.time.dtype,
            compression=compression,
            compression_opts=compression_opts,
        )
        group.attrs["n_neurons"] = data.n_neurons
        group.attrs["t_max"] = data.t_max

        if isinstance(data, nir.ValuedEventData):
            group.create_dataset(
                "value",
                data=data.value,
                dtype=data.value.dtype,
                compression=compression,
                compression_opts=compression_opts,
            )

    def _write_node_data(group: h5py.Group, node_data: nir.NIRNodeData):
        group.attrs["__type__"] = "NIRNodeData"
        obs_group = group.create_group("observables")

        for name, obs in node_data.observables.items():
            g = obs_group.create_group(name)
            if isinstance(obs, nir.TimeGriddedData):
                _write_time_gridded(g, obs)
            elif isinstance(obs, nir.EventData):
                _write_event_data(g, obs)
            else:
                raise TypeError(f"Unsupported observable type: {type(obs)}")

    def _write_graph_data(group: h5py.Group, node: dict) -> None:
        group.attrs["__type__"] = "NIRGraphData"
        nodes_group = group.create_group("nodes")

        for name, node in graph_data.nodes.items():
            g = nodes_group.create_group(name)
            if isinstance(node, nir.NIRNodeData):
                _write_node_data(g, node)
            elif isinstance(node, nir.NIRGraphData):
                _write_graph_data(g, node)
            else:
                raise TypeError(f"Unsupported node type: {type(node)}")

    with h5py.File(filename, "w") as f:
        _write_graph_data(f, graph_data)
