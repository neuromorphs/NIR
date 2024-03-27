from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from .node import NIRNode


@dataclass(eq=False)
class Delay(NIRNode):
    r"""Simple delay node.

    This node implements a simple delay:

    .. math::
        y(t) = x(t - \tau)
    """

    delay: np.ndarray  # Delay
    input_type: Optional[Dict[str, np.ndarray]] = None
    output_type: Optional[Dict[str, np.ndarray]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # set input and output shape, if not set by user
        self.input_type = {"input": np.array(self.delay.shape)}
        self.output_type = {"output": np.array(self.delay.shape)}
