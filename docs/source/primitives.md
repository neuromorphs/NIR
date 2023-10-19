# Primitives

NIR defines 16 fundamental primitives listed in the table below, which backends are free to implement as they want, leading to varying outputs across platforms. While discrepancies could be minimized by constraining implementations or making backends aware of each other's discretization choices, NIR does not do this since it is declarative, specifying only the necessary inputs and outputs. Constraining implementations would cause hardware incompatibilities and making backends aware of each other could create large O(N^2) overhead for N backends. The primitives are already computationally expressive and able to solve complex PDEs. 

| Primitive | Parameters | Computation | Reset |
|-|-|-|-|  
| **Input** | Input shape | - | - |
| **Output** | Output shape | - | - |
| **Affine** | $W, b$ | $ W*I + b$ | - |
| **Convolution** | $W$, Stride, Padding, Dilation, Groups, Bias | $f \star g$ | - |
| **Current-based leaky integrate-and-fire** | $\tau_\text{syn}$, $\tau_\text{mem}$, R, $v_\text{leak}$, $v_\text{thr}$, $w_\text{in}$ | **LI**_1_; **Linear**; **LIF**_2_ | $\begin{cases} v_\text{LI\_2}-v_\text{thr} & \text{Spike} \\ v & \text{else} \end{cases}$ |
| **Delay** | $\tau$ | $I(t - \tau)$ | - |  
| **Flatten** | Input shape, Start dim., End dim. | - | - |
| **Integrator** | $\text{R}$ | $\dot{v} = R I$ | - |
| **Integrate-and-fire** | $\text{R}, v_\text{thr}$ | **Integrator**; **Threshold** | $\begin{cases} v-v_\text{thr} & \text{Spike} \\ v & \text{else} \end{cases}$ |
| **Leaky integrator (LI)** | $\tau, \text{R}, v_\text{leak}$ | $\tau \dot{v} = (v_\text{leak} - v) + R I$ | - |
| **Linear** | $W$ | $W I$ | - |
| **Leaky integrate-fire (LIF)** | $\tau, \text{R}, v_\text{leak}, v_\text{thr}$ | **LI**; **Threshold** | $\begin{cases} v-v_\text{thr} & \text{Spike} \\ v & \text{else} \end{cases}$ |
| **Scale** | $s$ | $s I$ | - |
| **SumPooling** | $p$ | $\sum_{j} x_j$ |  |
| **Threshold** | $\theta_\text{thr}$ | $H(I - \theta_\text{thr})$ | - |

Each primitive is defined by their own dynamical equation, specified in the [API docs](https://nnir.readthedocs.io/en/latest/modindex.html).

## Connectivity 

Each computational unit is a node in a static graph.
Given 3 nodes $A$ which is a LIF node, $B$ which is a Linear node and $C$ which is another LIF node, we can define edges in the graph such as:

$$
    A \rightarrow B \\
    B \rightarrow C
$$

## Format
The intermediate represenation can be stored as hdf5 file, which benefits from compression. 