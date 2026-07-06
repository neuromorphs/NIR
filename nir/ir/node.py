from abc import ABC
from dataclasses import asdict, dataclass, field
from typing import Any, Dict

import numpy as np


@dataclass(eq=False)
class NIRNode(ABC):
    """Base superclass of Neural Intermediate Representation Unit (NIR).

    All NIR primitives inherit from this class, but NIRNodes should never be
    instantiated.
    """

    # input_type and output_type are not initialized in the constructor, they are
    # populated in __post_init__ for each subclass. metadata is an optional
    # keyword argument. All three are keyword-only so that subclasses can add
    # positional fields without running into the "non-default argument follows
    # default argument" ordering error. (Requires Python 3.10+.)
    input_type: Dict[str, np.ndarray] = field(init=False, kw_only=True)
    output_type: Dict[str, np.ndarray] = field(init=False, kw_only=True)
    metadata: Dict[str, Any] = field(default_factory=dict, kw_only=True)

    def __init__(self) -> None:
        raise AttributeError("NIRNode does not have a default constructor.")

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
    def from_dict(cls, kwargs: Dict[str, Any]) -> "NIRNode":
        assert kwargs["type"] == cls.__name__
        kwargs = kwargs.copy()  # Local scope
        del kwargs["type"]

        return cls(**kwargs)
