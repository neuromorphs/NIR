from typing import Any

from .conv import Conv1d, Conv2d
from .delay import Delay
from .flatten import Flatten
from .graph import Input, NIRGraph, Output
from .linear import Affine, Linear, Scale
from .neuron import IF, LI, LIF, CubaLI, CubaLIF, I
from .pooling import AvgPool2d, SumPool2d
from .threshold import Threshold
from .typing import NIRNode

# NIRNodes that can be (de)serialized
__all_ir = [
    # conv
    "Conv1d",
    "Conv2d",
    # delay
    "Delay",
    # flatten
    "Flatten",
    # graph
    "Input",
    "NIRGraph",
    "Output",
    # linear
    "Affine",
    "Linear",
    "Scale",
    # neuron
    "CubaLI",
    "CubaLIF",
    "I",
    "IF",
    "LI",
    "LIF",
    # pooling
    "AvgPool2d",
    "SumPool2d",
    # surrogate_gradient
    "Threshold",
]


def str2NIRNode(type: str) -> NIRNode:
    assert type in __all_ir

    return globals()[type]


def dict2NIRNode(data_dict: dict[str, Any]) -> NIRNode:
    """Assume data_dict["type"] exist and correspond to a subclass of NIRNode.

    Other items should match fields in the corresponding NIRNode subclass, unless
    subclass provides from_dict. Any extra item will be rejected and should be removed
    before calling this function
    """
    return str2NIRNode(data_dict["type"]).from_dict(data_dict)


# we could do this, but ruff complains
# __all__ = __all_ir + ["str2NIRNode", "dict2NIRNode"]
__all__ = [
    "IF",
    "LI",
    "LIF",
    # linear
    "Affine",
    # pooling
    "AvgPool2d",
    # conv
    "Conv1d",
    "Conv2d",
    # neuron
    "CubaLI",
    "CubaLIF",
    # delay
    "Delay",
    # flatten
    "Flatten",
    "I",
    # graph
    "Input",
    "Linear",
    "NIRGraph",
    # node
    "NIRNode",
    "Output",
    "Scale",
    "SumPool2d",
    # surrogate_gradient
    "Threshold",
    "dict2NIRNode",
    "str2NIRNode",
]
