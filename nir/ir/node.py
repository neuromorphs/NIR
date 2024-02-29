from dataclasses import dataclass
import numpy as np


@dataclass(eq=False)
class NIRNode:
    """Base superclass of Neural Intermediate Representation Unit (NIR).

    All NIR primitives inherit from this class, but NIRNodes should never be
    instantiated.
    """

    # Note: Adding input/output types as follows is ideal, but requires Python 3.10
    # input_type: Types = field(init=False, kw_only=True)
    # output_type: Types = field(init=False, kw_only=True)

    def __eq__(self, other):
        return self is other
