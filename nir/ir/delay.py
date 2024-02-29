from .common import *


@dataclass(eq=False)
class Delay(NIRNode):
    r"""Simple delay node.

    This node implements a simple delay:

    .. math::
        y(t) = x(t - \tau)
    """

    delay: np.ndarray  # Delay

    def __post_init__(self):
        # set input and output shape, if not set by user
        self.input_type = {"input": np.array(self.delay.shape)}
        self.output_type = {"output": np.array(self.delay.shape)}
