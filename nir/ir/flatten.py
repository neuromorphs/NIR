from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from .node import NIRNode
from .typing import Types
from .utils import calc_flatten_output, parse_shape_argument


@dataclass(eq=False)
class Flatten(NIRNode):
    """Flatten node.

    This node flattens its input tensor.
    input_type must be a dict with one key: "input".
    """

    # Shape of input tensor (overrrides input_type from
    # NIRNode to allow for non-keyword (positional) initialization)
    input_type: Types
    start_dim: int = 1  # First dimension to flatten
    end_dim: int = -1  # Last dimension to flatten
    input_type: Optional[Dict[str, np.ndarray]] = None
    output_type: Optional[Dict[str, np.ndarray]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.input_type = parse_shape_argument(self.input_type, "input")
        if self.input_type["input"] is None:
            self.input_type = {"input": None}
            self.output_type = {"output": None}
        else:
            self.output_type = {
                "output": calc_flatten_output(
                    self.input_type["input"], self.start_dim, self.end_dim
                )
            }
            # make sure input and output shape are valid
            if np.prod(self.input_type["input"]) != np.prod(self.output_type["output"]):
                raise ValueError(
                    "input and output shape must have same number of elements"
                )

    def to_dict(self) -> Dict[str, Any]:
        ret = super().to_dict()
        ret["input_type"] = self.input_type["input"]
        return ret

    @classmethod
    def from_dict(cls, node: Dict[str, Any]):
        node["input_type"] = {
            "input": node["input_type"] if "input_type" in node else None
        }
        return super().from_dict(node)
