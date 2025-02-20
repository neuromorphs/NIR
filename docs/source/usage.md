# How to use NIR

NIR itself is a standard: a way to formalize physical neural networks so they are completely unambiguous.
The unambiguous part is crucial because anyone who reads or writes to NIR cannot be confused about what the network does and does not do.

As a user, you rarely need to understand what's inside NIR.
Rather, you would use NIR as a middleman between, say, a neuron simulator and some piece of [neuromorphic](https://en.wikipedia.org/wiki/Neuromorphic_engineering) hardware such as [SynSense's Speck chip](https://www.synsense.ai/products/speck-2/) or [Intel's Loihi platform](https://www.intel.com/content/www/us/en/newsroom/news/intel-unveils-neuromorphic-loihi-2-lava-software.html).
Therefore, for this guide, we focus on how to use NIR graphs as objects, without fully understanding what's inside them.

![NIR platforms](nir_platforms.png)

## Reading and writing Python graphs
```{admonition} See also
:class: info
Read more on [Working with NIR graphs](#working_with_nir) and our [Code Examples](#examples/index).
```

NIR is excellent as an exchange format, and many frameworks provide functions that lets you do this without having to modify or work with NIR yourself.
We present an example from two platforms (Norse and Sinabs) below, but we have many more examples from other tools [in Code Examples](#examples/index) and
you can see more code [in the repository for our paper](https://github.com/neuromorphs/NIR/tree/main/paper/).

Note that Norse is based on PyTorch and uses [NIRTorch](#dev_pytorch) to convert PyTorch models to NIR.
You can also do this manually, by constructing your own NIR graphs as shown in our [API design documentation](#api_desige).

### Part 1: Convert Norse model to NIR
```python
import torch
import norse.torch as norse

# Define our neural network model
model = ...

# Convert model to NIR
nir_model = norse.to_nir(model)
```

### Part 2: Convert NIR model to chip
```python
import sinabs
from sinabs.backend.dynapcnn import DynapcnnNetwork

# Convert NIR model to Sinabs
batch_size = ... # Define batch size to your liking
sinabs.from_nir(nir_model, batch_size=batch_size)
# Convert Sinabsmodel to chip-supported CNN
dynapcnn_model = DynapcnnNetwork(sinabs_model, input_shape=sample_data.shape[-1])
# Move model to chip!
dynapcnn_model.to("speck2fdevkit")
```

## Reading and writing NIR files

```{admonition} See also
:class: info
See more details on [Working with NIR graphs](#working_with_nir)
```

You can also manually write and read NIR files.
This is useful if you want to save a model to disk and use it later.
Or if you want to load in a model that someone else has created.

### Writing a NIR file
[NIR consists of graphs](#working_with_nir) that describe the structure of a neural network.
Our reference implementation uses Python to describe these graphs, so you can imagine having a graph in an object, say `nir_model`.
To write this graph to file, you can use

```python
import nir
nir.write(nir_model, "my_model.nir")
```

### Reading a NIR file
Reading a NIR file is similarly easy and will give you a graph object that you can use in your code.

```python
import nir
nir_model = nir.read("my_model.nir")
```

Note that the graph object (`nir_model`) doesn't do anything by itself.
You still need to convert it to a format that your hardware or simulator can understand.
Read more about this in the [Using NIR in hardware guide](#porting_nir).