# CNN task: N-MNIST

NIR graph exported from sinabs implementation (compatible with Speck).

hardware to run on: Loihi 2, SpiNNaker2, Speck 

simulators to run on: Sinabs (+ others?)

## Checklist per simulator

1. Load the graph from `scnn_mnist.nir`
2. Use the [NMNIST test dataset from Tonic](https://tonic.readthedocs.io/en/latest/generated/tonic.datasets.NMNIST.html#tonic.datasets.NMNIST) (emphasis on **test**) to provide an accuracy. Store that as a single number in `<PLATFORM>_accuracy.npy`.
3. Pass the data from `val_numbers.npy` through the first two layers in the model, generating output data from the first convolution + neuron layer. That data should have (300, 10, 16, 16, 16) output. Store that in `<PLATFORM>_activity.npy`.

## Things to visualize/compare
- intermediate layer activations (which layer? perhaps first layer, for better visual features)
  - TODO: description of which sample, which timesteps, etc.
  - e.g. 1ms timebins, N-MNIST
- output logits

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

**Model extraction**
Execute the `mnist_sinabs_to_nir.py` which defines an ANN, loads pretrained weights, converts it to SNN and then generates the corresponding `NIRGraph`.