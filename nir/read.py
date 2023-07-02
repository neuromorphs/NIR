import typing
import h5py
import pathlib

import nir


def read(filename: typing.Union[str, pathlib.Path]) -> nir.NIR:
    """Load a NIR from a HDF5 file."""
    with h5py.File(filename, "r") as f:
        units = []
        for k, unit in f["units"].items():
            if unit["type"][()] == b"LeakyIntegrator":
                units.append(
                    nir.LeakyIntegrator(
                        tau=unit["tau"][()],
                        r=unit["r"][()],
                        v_leak=unit["v_leak"][()],
                    )
                )
            elif unit["type"][()] == b"Linear":
                units.append(
                    nir.Linear(
                        weights=unit["weights"][()],
                        bias=unit["bias"][()],
                    )
                )
            elif unit["type"][()] == b"Conv1d":
                units.append(
                    nir.Conv1d(
                        weights=unit["weights"][()],
                        stride=unit["stride"][()],
                        padding=unit["padding"][()],
                        dilation=unit["dilation"][()],
                        groups=unit["groups"][()],
                        bias=unit["bias"][()],
                    )
                )
            elif unit["type"][()] == b"Conv2d":
                units.append(
                    nir.Conv2d(
                        weights=unit["weights"][()],
                        stride=unit["stride"][()],
                        padding=unit["padding"][()],
                        dilation=unit["dilation"][()],
                        groups=unit["groups"][()],
                        bias=unit["bias"][()],
                    )
                )
            else:
                raise ValueError(f"Unknown unit type: {unit['type'][()]}")
        connectivity = f["connectivity"][()]
        return nir.NIR(units=units, connectivity=connectivity)
