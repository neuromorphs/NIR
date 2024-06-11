from typing import Sequence, Union

import numpy as np

from .typing import Types


def parse_shape_argument(x: Types, key: str):
    """Parse the shape argument of a NIR node."""
    if isinstance(x, np.ndarray):
        return {key: x}
    elif isinstance(x, Sequence):
        return {key: np.array(x)}
    elif isinstance(x, dict):
        return x
    elif x is None:
        return {key: None}


def calculate_conv_output(
    input_shape: Union[int, Sequence[int]],
    padding: Union[int, str, Sequence[int]],
    dilation: Union[int, Sequence[int]],
    kernel_size: Union[int, Sequence[int]],
    stride: Union[int, Sequence[int]],
) -> Sequence[int]:
    """Calculates the output for a single dimension of a convolutional layer.
    https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html#torch.nn.Conv1d

    :param input_shape: input shape, either int or (int, int)
    :type input_shape: int | Sequence[int]
    :param padding: padding
    :type padding: int | Sequence[int]
    :param dilation: dilation
    :type dilation: int | Sequence[int]
    :param kernel_size: kernel size
    :type kernel_size: int | Sequence[int]
    :param stride: stride
    :type stride: int | Sequence[int]

    :return: output shape
    :rtype: Sequence[int]
    """
    if isinstance(input_shape, (int, np.integer)):
        ndim = 1
    else:
        ndim = len(input_shape)
    if isinstance(padding, str) and padding == "valid":
        padding = [0] * ndim
    shapes = []
    for i in range(ndim):
        if isinstance(padding, str) and padding == "same":
            shape = _index_tuple(input_shape, i)
        else:
            shape = np.floor(
                (
                    _index_tuple(input_shape, i)
                    + 2 * _index_tuple(padding, i)
                    - _index_tuple(dilation, i) * (_index_tuple(kernel_size, i) - 1)
                    - 1
                )
                / _index_tuple(stride, i)
                + 1
            )
        shapes.append(int(shape.item()))
    return np.array(shapes)


def calc_flatten_output(input_shape: Sequence[int], start_dim: int, end_dim: int):
    start_shape = np.array(input_shape[:start_dim]) if start_dim != 0 else []
    middle_shape = (
        np.prod(input_shape[start_dim : end_dim + 1])
        if end_dim != -1
        else np.prod(input_shape[start_dim:])
    )
    end_shape = (
        np.array(input_shape[end_dim + 1 :])
        if end_dim != -1 and end_dim != len(input_shape) - 1
        else []
    )
    return np.array(
        [
            *start_shape,
            middle_shape,
            *end_shape,
        ]
    )


def _index_tuple(tuple: Union[int, Sequence[int]], index: int) -> np.ndarray:
    """If the input is a tuple/array, index it.

    Otherwise, return it as-is.
    """
    if isinstance(tuple, np.ndarray):
        return tuple[index]
    elif isinstance(tuple, Sequence):
        return np.array(tuple[index])
    elif isinstance(tuple, (int, np.integer)):
        return np.array([tuple])
    else:
        try:
            return tuple[index]
        except TypeError:
            raise TypeError(f"tuple must be int or np.ndarray, not {type(tuple)}")


def ensure_str(a: Union[str, bytes]) -> str:
    if isinstance(a, bytes):
        return a.decode("utf8")
    elif isinstance(a, str):
        return a
    else:
        raise TypeError(f"Unexpected non-string type encountered: {type(a)}")
