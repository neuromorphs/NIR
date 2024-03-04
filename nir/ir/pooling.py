from dataclasses import dataclass

import numpy as np

from .node import NIRNode


@dataclass(eq=False)
class SumPool2d(NIRNode):
    """Sum pooling layer in 2d."""

    kernel_size: np.ndarray  # (Height, Width)
    stride: np.ndarray  # (Height, width)
    padding: np.ndarray  # (Height, width)

    def __post_init__(self):
        self.input_type = {"input": None}
        self.output_type = {"output": None}
