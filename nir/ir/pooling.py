from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from .node import NIRNode


@dataclass(eq=False)
class SumPool2d(NIRNode):
    """Sum pooling layer in 2d."""

    kernel_size: np.ndarray  # (Height, Width)
    stride: np.ndarray  # (Height, width)
    padding: np.ndarray  # (Height, width)
    input_type: Optional[Dict[str, np.ndarray]] = None
    output_type: Optional[Dict[str, np.ndarray]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.input_type = {"input": None}
        self.output_type = {"output": None}


@dataclass(eq=False)
class AvgPool2d(NIRNode):
    """Average pooling layer in 2d."""

    kernel_size: np.ndarray  # (Height, Width)
    stride: np.ndarray  # (Height, width)
    padding: np.ndarray  # (Height, width)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.input_type = {"input": None}
        self.output_type = {"output": None}
