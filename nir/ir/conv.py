from dataclasses import dataclass

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

    input_shape: int | None  # N
    weight: np.ndarray  # Weight C_out * C_in * N
    stride: int  # Stride
    padding: int | str  # Padding
    dilation: int  # Dilation
    groups: int  # Groups
    bias: np.ndarray  # Bias C_out

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

    input_shape: tuple[int, int] | None  # N_x, N_y
    weight: np.ndarray  # Weight C_out * C_in * W_x * W_y
    stride: int | tuple[int, int]  # Stride
    padding: int | tuple[int, int] | str  # Padding
    dilation: int | tuple[int, int]  # Dilation
    groups: int  # Groups
    bias: np.ndarray  # Bias C_out

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
                self.weight.shape[2:],
                self.stride,
            )
            self.output_type = {
                "output": np.array([self.weight.shape[0], *output_shape])
            }


@dataclass(eq=False)
class SparseConv2D(NIRNode):
    """Sparse 2D convolution.

    :param rows: Populated rows in sparse martrix.
    :type rows: numpy.ndarray
    :param cols: Populated columns in the sparse martrix.
    :type cols: numpy.ndarray
    :param weights: Values of the populated rows and columns in the sparse martrix.
        Represents the actual weights of the filters encoded in the matrix
    :type weights: numpy.ndarray
    :param shape: Shape of the sparse matrix (cannot be inferred from the rows & cols).
    :type shape: tuple[int, int]
    """

    # Rows, columns and values that define the fan-in.
    # Suitable for use with the COO format,
    # which can be subsequently converted to other formats.
    rows: np.ndarray # 1D, dense, np.uint32
    cols: np.ndarray # 1D, dense, np.uint32
    weights: np.ndarray # 1D, dense, np.float32

    # Shape of the fan-in sparse matrix.
    shape: tuple[int, int]

    def __post_init__(self):

        # Input shape (1D).
        # Corresponds to the number of columns in the matrix.
        self.input_type = {"input": None if self.shape is None else self.shape[1]}

        # Output shape (1D).
        # Corresponds to the number of rows in the matrix.
        self.output_type = {
            "output": None if self.shape is None else self.shape[0],
        }

        # Ensure that the rows and columns have integer types
        self.rows = self.rows.astype(np.uint32)
        self.cols = self.cols.astype(np.uint32)

        assert self.rows.size == self.cols.size == self.weights.size
        assert len(self.rows.shape) == len(self.cols.shape) == len(self.weights.shape) == 1
    