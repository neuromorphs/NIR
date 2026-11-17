from dataclasses import dataclass

import numpy as np

from .node import NIRNode

UPSAMPLE_MODES = ("nearest", "bilinear")


@dataclass(eq=False)
class Upsample2d(NIRNode):
    """Upsampling layer in 2d.

    Scales the spatial dimensions of a (Channels, Height, Width) input by integer factors,
    with the semantics of ``torch.nn.functional.interpolate``:

    .. math::
        y = \\text{interpolate}(x, \\text{scale\\_factor}, \\text{mode}, \\text{align\\_corners})

    The output has shape (Channels, Height * scale_factor[0], Width * scale_factor[1]).

    Arguments:
        scale_factor: Integer upsampling factors (Height, Width).
        mode: Interpolation mode, "nearest" or "bilinear". Defaults to "nearest".
        align_corners: As in PyTorch, only used for "bilinear". Defaults to False.
    """

    scale_factor: np.ndarray  # (Height, Width)
    mode: str = "nearest"
    align_corners: bool = False

    def __post_init__(self):
        self.scale_factor = np.asarray(self.scale_factor, dtype=np.int64).reshape(-1)
        if self.scale_factor.size == 1:
            self.scale_factor = np.repeat(self.scale_factor, 2)
        if self.scale_factor.size != 2:
            raise ValueError(
                f"scale_factor must be one or two integers, got {self.scale_factor}"
            )
        if np.any(self.scale_factor < 1):
            raise ValueError(f"scale_factor must be >= 1, got {self.scale_factor}")
        self.mode = (
            self.mode.decode("utf8") if isinstance(self.mode, bytes) else str(self.mode)
        )
        if self.mode not in UPSAMPLE_MODES:
            raise ValueError(f"mode must be one of {UPSAMPLE_MODES}, got '{self.mode}'")
        self.align_corners = bool(self.align_corners)
        self.input_type = {"input": None}
        self.output_type = {"output": None}
