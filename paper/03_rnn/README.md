# RNN task: Braille letter reading

NIR graph exported from snnTorch.

Hardware to run on: Xylo, Loihi 2, SpiNNaker2

Simulators to run on: snnTorch, Norse, Lava-DL, Nengo (?)

**Note** the graphs use `dt=1e-4`

## The dataset

The dataset for this task was produced from the [original one for Braille letter reading](https://zenodo.org/records/7050094) extracting a subset of letters containing the `Space`, `A`, `E`, `I`, `O`, `U` and `Y` characters. The encoding scheme with $\theta = 1$ was adopted. \
A reduced input dimension was used with respect to what described in the [related paper](https://www.frontiersin.org/articles/10.3389/fnins.2022.951164/full): by applying the `OR` logic to the spike-encoded data, a spike was associated, with a unipolar approach, at each time step where an ON or OFF event was present. This led to input data having the same dimension as the number of taxels in the artificial fingertip used top record the dataset, namely 12.\
As in the related paper, the binning procedure resulting in a $\Delta t = 5\text{ ms}$ as time step within the individual samples was taken into account.


## Checklist per simulator

0. Choose whether to use reset by subtraction or reset to zero for the CuBa-LIF neurons of the model
1. Load the graph from `braille_subtract.nir` (in the first case) or `braille_zero.nir` (in the second one)
2. Use the `ds_test.pt` subset (from the `data` folder) to provide test accuracy. Store that as a single number in `<PLATFORM>_accuracy_zero.npy` and `<PLATFORM>_accuracy_subtract.npy` for the two zero/subtract graphs.
3. Pass the data from the first element in the `ds_test.pt` dataset (`torch.load("data/ds_test.pt")[0][0]`) through the model, and store the output spikes of the first LIF layer (the `"lif1.lif"` node in the graph). That data should have (256, 38) output for the `_zero` graph and (256, 55) for the `_subtract` graph. Store that in `<PLATFORM>_activity_zero.npy` or `<PLATFORM>_activity_subtract.npy`.
