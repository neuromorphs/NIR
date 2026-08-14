
import numpy as np

from .node import NIRNode

# Nodes are uniquely named computational units
Nodes = dict[str, "NIRNode"]
# Edges map one node id to another via the identity
Edges = list[tuple[str, str]]
# Types is a dict mapping strings to tensor shapes
Types = dict[str, np.ndarray]
