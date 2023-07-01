from dataclasses import dataclass
import typing

import numpy as np


Connectivity = typing.NewType("Connectivity", list(typing.Tuple[int, int]))


@dataclass
class NIR:
  """Neural Intermediate Representation (NIR)"""
  units: typing.List[typing.Any] # List of units
  connectivity: Connectivity


@dataclass
class LeakyIntegrator:
  """Leaky integrator neuron model.
  
  The leaky integrator neuron model is defined by the following equation:
  $$
  \tau \dot{v} = \alpha(v_{leak} - v) + \beta I
  $$
  Where $\tau$ is the time constant, $v$ is the membrane potential,
  $\alpha$ is the gain, $v_{leak}$ is the leak voltage, and $\beta$ is the bias for 
  the input current $I$.
  """

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