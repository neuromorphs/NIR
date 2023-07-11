import typing
import h5py
import pathlib

import nir


def read_node(node: typing.Any) -> nir.NIRNode:
    """Read a graph from a HDF/conn5 file."""
    if node["type"][()] == b"Affine":
        return nir.Affine(
            weight=node["weight"][()],
            bias=node["bias"][()],
        )
    elif node["type"][()] == b"Conv1d":
        return nir.Conv1d(
            weight=node["weight"][()],
            stride=node["stride"][()],
            padding=node["padding"][()],
            dilation=node["dilation"][()],
            groups=node["groups"][()],
            bias=node["bias"][()],
        )
    elif node["type"][()] == b"Conv2d":
        return nir.Conv2d(
            weight=node["weight"][()],
            stride=node["stride"][()],
            padding=node["padding"][()],
            dilation=node["dilation"][()],
            groups=node["groups"][()],
            bias=node["bias"][()],
        )
    elif node["type"][()] == b"Delay":
        return nir.Delay(
            delay=node["delay"][()],
        )
    elif node["type"][()] == b"I":
        return nir.I(
            r=node["r"][()],
        )
    elif node["type"][()] == b"IF":
        return nir.IF(
            r=node["r"][()],
            v_threshold=node["v_threshold"][()],
        )
    elif node["type"][()] == b"Input":
        return nir.Input(
            shape=node["shape"][()],
        )
    elif node["type"][()] == b"LI":
        return nir.LI(
            tau=node["tau"][()],
            r=node["r"][()],
            v_leak=node["v_leak"][()],
        )
    elif node["type"][()] == b"Linear":
        return nir.Linear(
            weight=node["weight"][()],
        )
    elif node["type"][()] == b"LIF":
        return nir.LIF(
            tau=node["tau"][()],
            r=node["r"][()],
            v_leak=node["v_leak"][()],
            v_threshold=node["v_threshold"][()],
        )
    elif node["type"][()] == b"CubaLIF":
        return nir.CubaLIF(
            tau_mem=node["tau_mem"][()],
            tau_syn=node["tau_syn"][()],
            r=node["r"][()],
            v_leak=node["v_leak"][()],
            v_threshold=node["v_threshold"][()],
        )
    elif node["type"][()] == b"NIRGraph":
        return nir.NIRGraph(
            nodes=[read_node(n) for n in node["nodes"].values()],
            edges=node["edges"][()],
        )
    elif node["type"][()] == b"Output":
        return nir.Output()
    elif node["type"][()] == b"Threshold":
        return nir.Threshold(
            threshold=node["threshold"][()],
        )
    else:
        raise ValueError(f"Unknown unit type: {node['type'][()]}")


def read(filename: typing.Union[str, pathlib.Path]) -> nir.NIRGraph:
    """Load a NIR from a HDF/conn5 file."""
    with h5py.File(filename, "r") as f:
        return read_node(f["node"])
