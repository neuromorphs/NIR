import typing
import pathlib

import h5py
import numpy as np

import nir


def write(filename: typing.Union[str, pathlib.Path], graph: nir.NIR) -> None:
    def convert_unit(unit: nir.NIRUnit) -> dict:
        if isinstance(unit, nir.LeakyIntegrator):
            return {
                "type": "LeakyIntegrator",
                "tau": unit.tau,
                "alpha": unit.alpha,
                "beta": unit.beta,
                "v_leak": unit.v_leak,
            }
        elif isinstance(unit, nir.Linear):
            return {
                "type": "Linear",
                "weights": unit.weights,
                "bias": unit.bias,
            }
        elif isinstance(unit, nir.Conv1d):
            return {
                "type": "Conv1d",
                "weights": unit.weights,
                "stride": unit.stride,
                "padding": unit.padding,
                "dilation": unit.dilation,
                "groups": unit.groups,
                "bias": unit.bias,
            }
        elif isinstance(unit, nir.Conv2d):
            return {
                "type": "Conv2d",
                "weights": unit.weights,
                "stride": unit.stride,
                "padding": unit.padding,
                "dilation": unit.dilation,
                "groups": unit.groups,
                "bias": unit.bias,
            }
        else:
            raise ValueError(f"Unknown unit type: {unit}")

    """Write a NIR to a HDF5 file."""
    with h5py.File(filename, "w") as f:
        units_group = f.create_group("units")
        for i, unit in enumerate(graph.units):
            d = convert_unit(unit)
            unit_group = units_group.create_group(str(i))
            for k, v in d.items():
                if isinstance(v, str):
                    unit_group.create_dataset(k, data=v, dtype=h5py.string_dtype())
                elif isinstance(v, np.ndarray):
                    unit_group.create_dataset(k, data=v, dtype=v.dtype)
                else:
                    unit_group.create_dataset(k, data=v)
                    # raise ValueError(f"Unknown type: {type(v)}")
        f.create_dataset("connectivity", data=graph.connectivity)
