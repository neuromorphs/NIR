# How to use NIR

NIR itself is a standard: a way to formalize physical neural networks so they are completely unambiguous.
The unambiguous part is crucial because anyone who reads or writes to NIR cannot be confused about what the network does and does not do.

As a user, you rarely need to understand what's inside NIR.
Rather, you would use NIR as a middleman between, say, a neuron simulator and some piece of [neuromorphic](https://en.wikipedia.org/wiki/Neuromorphic_engineering) hardware such as [SynSense's Speck chip](https://www.synsense.ai/products/speck-2/) or [Intel's Loihi platform](https://www.intel.com/content/www/us/en/newsroom/news/intel-unveils-neuromorphic-loihi-2-lava-software.html).

![NIR platforms](nir_platforms.png)

Below, we listed a few concrete examples on how to use NIR.
Please refer to the **Examples** section in the sidebar for code for each supported platform.
More code examples are available [in the repository for our paper](https://github.com/neuromorphs/NIR/tree/main/paper/).

## Example: Norse model to Sinabs Speck
This example demonstrates how to convert a Norse model to a Sinabs model and then to a Speck chip.
Note that Norse is based on PyTorch and uses [NIRTorch](#dev_pytorch) to convert PyTorch models to NIR.
You can also do this manually, by constructing your own NIR graphs as shown in our [API design documentation](#api_desige).

### Part 1: Convert Norse model to NIR
```python
import torch
import norse.torch as norse

# Define our neural network model
model = ...

# Convert model to NIR
#   Note that we use some sample data to "trace" the graph in PyTorch.
#   You need to ensure that shape of the data fits your model
batch_size = 1
sample_data = torch.randn(batch_size, 10)
nir_model = norse.to_nir(model, sample_data)
```

### Part 2: Convert NIR model to chip
```python
import sinabs
from sinabs.backend.dynapcnn import DynapcnnNetwork

# Convert NIR model to Sinabs
sinabs.from_nir(nir_model, batch_size=batch_size)
# Convert Sinabsmodel to chip-supported CNN
dynapcnn_model = DynapcnnNetwork(sinabs_model, input_shape=sample_data.shape[-1])
# Move model to chip!
dynapcnn_model.to("speck2fdevkit")
```

## Example: Manually writing and reading NIR files
You can also manually write and read NIR files.
This is useful if you want to save a model to disk and use it later.
Or if you want to load in a model that someone else has created.

### Writing a NIR file
[NIR consists of graphs](#primitives) that describe the structure of a neural network.
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