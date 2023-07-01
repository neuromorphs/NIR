from dataclasses import dataclass

import numpy as np

@dataclass
class LeakyIntegrator:
  """Leaky integrator neuron model."""

  tau: np.ndarray # Time constant
  alpha: np.ndarray # Gain
  beta: np.ndarray # Bias
  v_leak: np.ndarray # Leak voltage


@dataclass
class Linear:

  weights: np.ndarray # Weights M * N
  bias: np.ndarray # Bias M


@dataclass
class Conv1d:
    """Convolutional layer in 1d"""
  
    weights: np.ndarray # Weights C_out * C_in * X
    stride: int # Stride
    padding: int # Padding
    dilation: int # Dilation
    groups: int # Groups
    bias: np.ndarray # Bias C_out


@dataclass
class Conv2d:
    """Convolutional layer in 2d"""
  
    weights: np.ndarray # Weights C_out * C_in * X * Y
    stride: int # Stride
    padding: int # Padding
    dilation: int # Dilation
    groups: int # Groups
    bias: np.ndarray # Bias C_out