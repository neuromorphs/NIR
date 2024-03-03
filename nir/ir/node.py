from dataclasses import asdict, dataclass, field
from typing import Any, Dict
import numpy as np


@dataclass(eq=False)
class NIRNode:
    """Base superclass of Neural Intermediate Representation Unit (NIR).

    All NIR primitives inherit from this class, but NIRNodes should never be
    instantiated.
    """

    # Note: Adding input/output types as follows is ideal, but requires Python 3.10
    input_type: Dict[str, np.ndarray] = field(init=False, kw_only=True)
    output_type: Dict[str, np.ndarray] = field(init=False, kw_only=True)
    metadata: Dict[str, Any] = field(init=True, kw_only=True, default_factory=dict)

    def __eq__(self, other):
        return self is other

    def to_dict(self) -> Dict[str, Any]:
        """Serialize into a dictionary."""
        ret = asdict(self)
        if "input_type" in ret.keys():
            del ret["input_type"]
        if "output_type" in ret.keys():
            del ret["output_type"]
        # Note: The customization below won't be automatically done recursively for nested NIRNode.
        # Therefore, classes with nested NIRNode e.g. NIRGraph must implement its own to_dict
        ret["type"] = type(self).__name__

        return ret

    @classmethod
    def from_dict(cls, node: Dict[str, Any]) -> "NIRNode":
        assert node["type"] == cls.__name__
        del node["type"]

        return cls(**node)
