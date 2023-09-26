# CNN task: N-MNIST

NIR graph exported from sinabs implementation (compatible with Speck).

hardware to run on: Loihi 2, SpiNNaker2, Speck 

simulators to run on: Sinabs (+ others?)

**Things to visualize/compare**:
- intermediate layer activations (which layer? perhaps first layer, for better visual features)
  - TODO: description of which sample, which timesteps, etc.
  - e.g. 1ms timebins, N-MNIST
- output logits

**Model definition** (from sinabs, still converted to spiking model)
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