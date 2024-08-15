from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from .node import NIRNode
from .utils import calculate_conv_output


@dataclass(eq=False)
class Conv1d(NIRNode):
    """Convolutional layer in 1d.

    Note that the input_shape argument is required to disambiguate the shape, and is
    used to infer the exact output shape along with the other parameters. If the
    input_shape is None, the output shape will also be None.

    The NIRGraph.infer_all_shapes function may be used to automatically infer the input
    and output types on the graph level.

    :param input_shape: Shape of spatial input (N,)
    :type input_shape: Optional[int]
    :param weight: Weight, shape (C_out, C_in, N)
    :type weight: np.ndarray
    :param stride: Stride
    :type stride: int
    :param padding: Padding, if string must be 'same' or 'valid'
    :type padding: int | str
    :param dilation: Dilation
    :type dilation: int
    :param groups: Groups
    :type groups: int
    :param bias: Bias array of shape (C_out,)
    :type bias: np.ndarray
    """

    input_shape: Optional[int]  # N
    weight: np.ndarray  # Weight C_out * C_in * N
    stride: int  # Stride
    padding: Union[int, str]  # Padding
    dilation: int  # Dilation
    groups: int  # Groups
    bias: np.ndarray  # Bias C_out
    input_type: Optional[Dict[str, np.ndarray]] = None
    output_type: Optional[Dict[str, np.ndarray]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.padding, str) and self.padding not in ["same", "valid"]:
            raise ValueError(
                f"padding must be 'same', 'valid', or int, not {self.padding}"
            )
        if self.input_shape is None:
            # leave input and output types undefined
            self.input_type = {"input": None}
            self.output_type = {"output": None}
        else:
            # infer input and output types from input_shape
            self.input_type = {
                "input": np.array([self.weight.shape[1], self.input_shape])
            }
            output_shape = calculate_conv_output(
                self.input_shape,
                self.padding,
                self.dilation,
                self.weight.shape[2],
                self.stride,
            )
            self.output_type = {
                "output": np.array([self.weight.shape[0], *output_shape])
            }


@dataclass(eq=False)
class Conv2d(NIRNode):
    """Convolutional layer in 2d.

    Note that the input_shape argument is required to disambiguate the shape, and is
    used to infer the exact output shape along with the other parameters. If the
    input_shape is None, the output shape will also be None.

    The NIRGraph.infer_all_shapes function may be used to automatically infer the input
    and output types on the graph level.

    :param input_shape: Shape of spatial input (N_x, N_y)
    :type input_shape: Optional[tuple[int, int]]
    :param weight: Weight, shape (C_out, C_in, N_x, N_y)
    :type weight: np.ndarray
    :param stride: Stride
    :type stride: int | int, int
    :param padding: Padding, if string must be 'same' or 'valid'
    :type padding: int | int, int | str
    :param dilation: Dilation
    :type dilation: int | int, int
    :param groups: Groups
    :type groups: int
    :param bias: Bias array of shape (C_out,)
    :type bias: np.ndarray
    """

    # Shape of input tensor (overrrides input_type from
    input_shape: Optional[Tuple[int, int]]  # N_x, N_y
    weight: np.ndarray  # Weight C_out * C_in * W_x * W_y
    stride: Union[int, Tuple[int, int]]  # Stride
    padding: Union[int, Tuple[int, int], str]  # Padding
    dilation: Union[int, Tuple[int, int]]  # Dilation
    groups: int  # Groups
    bias: np.ndarray  # Bias C_out
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.padding, str) and self.padding not in ["same", "valid"]:
            raise ValueError(
                f"padding must be 'same', 'valid', or int, not {self.padding}"
            )
        if isinstance(self.padding, int):
            self.padding = (self.padding, self.padding)
        if isinstance(self.stride, int):
            self.stride = (self.stride, self.stride)
        if isinstance(self.dilation, int):
            self.dilation = (self.dilation, self.dilation)
        if self.input_shape is None:
            # leave input and output types undefined
            self.input_type = {"input": None}
            self.output_type = {"output": None}
        else:
            # infer input and output types from input_shape
            self.input_type = {
                "input": np.array([self.weight.shape[1], *self.input_shape])
            }
            output_shape = calculate_conv_output(
                self.input_shape,
                self.padding,
                self.dilation,
                self.weight.shape[2],
                self.stride,
            )
            self.output_type = {
                "output": np.array([self.weight.shape[0], *output_shape])
            }
