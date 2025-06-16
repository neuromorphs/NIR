from abc import ABC
from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass(eq=False)
class NIRNode(ABC):
    """Base superclass of Neural Intermediate Representation Unit (NIR).

    All NIR primitives inherit from this class, but NIRNodes should never be
    instantiated.
    """

    # Note: Adding input/output types and metadata as follows is ideal, but requires Python 3.10
    # TODO: implement this in 2025 when 3.9 is EOL
    # input_type: Dict[str, np.ndarray] = field(init=False, kw_only=True)
    # output_type: Dict[str, np.ndarray] = field(init=False, kw_only=True)
    # metadata: Dict[str, Any] = field(init=True, default_factory=dict)

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
