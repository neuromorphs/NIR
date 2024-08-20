from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from .node import NIRNode


@dataclass(eq=False)
class Affine(NIRNode):
    r"""Affine transform that linearly maps and translates the input signal.

    This is equivalent to the
    `Affine transformation <https://en.wikipedia.org/wiki/Affine_transformation>`_

    Assumes a one-dimensional input vector of shape (N,).

    .. math::
        y(t) = W*x(t) + b
    """

    weight: np.ndarray  # Weight term
    bias: np.ndarray  # Bias term
    input_type: Optional[Dict[str, np.ndarray]] = None
    output_type: Optional[Dict[str, np.ndarray]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        assert len(self.weight.shape) >= 2, "Weight must be at least 2D"
        self.input_type = {
            "input": np.array(
                self.weight.shape[:-2] + tuple(np.array(self.weight.shape[-1:]).T)
            )
        }
        self.output_type = {
            "output": np.array(self.weight.shape[:-2] + (self.weight.shape[-2],))
        }


@dataclass(eq=False)
class Linear(NIRNode):
    r"""Linear transform without bias:

    .. math::
        y(t) = W*x(t)
    """

    weight: np.ndarray  # Weight term
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        assert len(self.weight.shape) >= 2, "Weight must be at least 2D"
        self.input_type = {
            "input": np.array(
                self.weight.shape[:-2] + tuple(np.array(self.weight.shape[-1:]).T)
            )
        }
        self.output_type = {"output": self.weight.shape[:-2] + (self.weight.shape[-2],)}


@dataclass(eq=False)
class Scale(NIRNode):
    r"""Scales a signal by some values.

    This node is equivalent to the
    `Hadamard product <https://en.wikipedia.org/wiki/Hadamard_product_(matrices)>`_.

    .. math::
        y(t) = x(t) \odot s
    """

    scale: np.ndarray  # Scaling factor
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.input_type = {"input": np.array(self.scale.shape)}
        self.output_type = {"output": np.array(self.scale.shape)}
