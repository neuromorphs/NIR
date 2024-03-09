# CNN task: N-MNIST

The NIR graph used for this task is exported from Sinabs (to ensure compatibility with the Speck chip). 
The file `sinabs.py` defines an ANN, loads pretrained weights, converts it to SNN and then exports the corresponding NIR graph to `cnn_sinabs.nir`.

The final comparison plots for the paper are generated in `cnn_plots.ipynb`, including the comparison table.

Hardware to run on:
- [x] Loihi 2
- [x] SpiNNaker2
- [x] Speck

Simulators to run on:
- [x] Lava
- [x] Spyx
- [x] Nengo
- [x] Norse
- [x] Sinabs
- [x] snnTorch
- [x] Spyx

## Checklist per simulator

1. Load the graph from `cnn_sinabs.nir`
2. Use the **test** dataset of N-MNIST from [Tonic](https://tonic.readthedocs.io/en/latest/generated/tonic.datasets.NMNIST.html#tonic.datasets.NMNIST) and compute the accuracy. Store that as a single number in `{platform}_accuracy.npy`.
3. Load the data from `cnn_numbers.npy`, which contains input data from 10 digits in the shape `(300, 10, 2, 34, 34)`. Pass these through the first two layers in the model, generating output data from the first convolution + neuron layer (see model definition below). That output data should have shape `(300, 10, 16, 16, 16)`. Store that in `{platform}_activity.npy`.

## Model definition
```
nn.Sequential(
  nn.Conv2d(
      2, 16, kernel_size=(5, 5), stride=(2, 2), padding=(1, 1), bias=False
  ),  # 16, 18, 18
  nn.ReLU(),
  nn.Conv2d(
      16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
  ),  # 8, 18,18
  nn.ReLU(),
  sl.SumPool2d(kernel_size=(2, 2)),  # 8, 17,17
  nn.Conv2d(
      16, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
  ),  # 8, 9, 9
  nn.ReLU(),
  sl.SumPool2d(kernel_size=(2, 2)),
  nn.Flatten(),
  nn.Linear(4 * 4 * 8, 256, bias=False),
  nn.ReLU(),
  nn.Linear(256, 10, bias=False),
  nn.ReLU(),
)
```
