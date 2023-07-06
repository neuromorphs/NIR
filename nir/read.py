import typing
import h5py
import pathlib

import nir


def read(filename: typing.Union[str, pathlib.Path]) -> nir.NIR:
    """Load a NIR from a HDF/conn5 file."""
    with h5py.File(filename, "r") as f:
        nodes = []
        for k, node in f["nodes"].items():
            if node["type"][()] == b"Input":
                nodes.append(
                    nir.Input(
                        shape=node["shape"][()],
                    )
                )
            elif node["type"][()] == b"Output":
                nodes.append(
                    nir.Output(
                    )
                )
            elif node["type"][()] == b"LI":
                nodes.append(
                    nir.LI(
                        tau=node["tau"][()],
                        r=node["r"][()],
                        v_leak=node["v_leak"][()],
                    )
                )
            elif node["type"][()] == b"LIF":
                nodes.append(
                    nir.LIF(
                        tau=node["tau"][()],
                        r=node["r"][()],
                        v_leak=node["v_leak"][()],
                        v_threshold=node["v_threshold"][()],
                    )
                )
            elif node["type"][()] == b"Linear":
                nodes.append(
                    nir.Linear(
                        weights=node["weights"][()],
                        bias=node["bias"][()],
                    )
                )
            elif node["type"][()] == b"Convolution":
                nodes.append(
                    nir.Convolution(
                        weights=node["weights"][()],
                        stride=node["stride"][()],
                        padding=node["padding"][()],
                        dilation=node["dilation"][()],
                        groups=node["groups"][()],
                        bias=node["bias"][()],
                    )
                )
            elif node["type"][()] == b"Delay":
                nodes.append(
                    nir.Delay(
                        delay=node["delay"][()],
                    )
                )
            elif node["type"][()] == b"Threshold":
                nodes.append(
                    nir.Threshold(
                        threshold=node["threshold"][()],
                    )
                )
            else:
                raise ValueError(f"Unknown unit type: {node['type'][()]}")
        edges = f["edges"][()]
        return nir.NIR(nodes=nodes, edges=edges)
