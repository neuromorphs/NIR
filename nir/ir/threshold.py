from dataclasses import dataclass

import numpy as np

from .node import NIRNode


@dataclass(eq=False)
class Threshold(NIRNode):
    r"""Threshold node.

    This node implements the heaviside step function:

    .. math::
        z = \begin{cases}
            1 & v > v_{thr} \\
            0 & else
        \end{cases}
    """

    threshold: np.ndarray  # Firing threshold

    def __post_init__(self):
        self.input_type = {"input": np.array(self.threshold.shape)}
        self.output_type = {"output": np.array(self.threshold.shape)}
