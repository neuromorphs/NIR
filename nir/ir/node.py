from dataclasses import asdict, dataclass
from typing import Any


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

    def to_dict(self) -> dict[str, Any]:
        """Serialize into a dictionary."""
        ret = asdict(self)
        # Note: The customization below won't be automatically done recursively for nested NIRNode.
        # Therefore, classes with nested NIRNode e.g. NIRGraph must implement its own to_dict
        ret["type"] = type(self).__name__

        return ret
