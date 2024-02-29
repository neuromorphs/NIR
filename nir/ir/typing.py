from typing import Dict, List, Tuple

import numpy as np

from .node import NIRNode

# Nodes are uniquely named computational units
Nodes = Dict[str, "NIRNode"]
# Edges map one node id to another via the identity
Edges = List[Tuple[str, str]]
# Types is a dict mapping strings to tensor shapes
Types = Dict[str, np.ndarray]
