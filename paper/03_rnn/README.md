# SRNN task: Braille letter reading

NIR graph exported after training in snnTorch.

Hardware to run on: Xylo, Loihi 2, SpiNNaker2

Simulators to run on: snnTorch, Norse, Lava-DL, (Nengo)

**Note:** the graphs use `dt=1e-4`

## The dataset

The dataset for this task was produced from the [original one for Braille letter reading](https://zenodo.org/records/7050094) extracting a subset of letters containing the `Space`, `A`, `E`, `I`, `O`, `U` and `Y` characters. The encoding scheme with $\theta = 1$ was adopted. \
A reduced input dimension was used with respect to what described in the [related paper](https://www.frontiersin.org/articles/10.3389/fnins.2022.951164/full): by applying the `OR` logic to the spike-encoded data, a spike was associated, with a unipolar approach, at each timestep where an ON or OFF event was present. This led to input data having the same dimension as the number of taxels in the artificial fingertip used to record the dataset, namely 12.\
As in the above mentioned paper, the binning procedure resulting in a $\Delta t = 5\text{ ms}$ as timestep within the individual samples was used.

## The models
We apply two models to the Braille dataset
1. `braille_noDelay_bias_zero.nir`
    * Membrane reset to zero without delay (in the same timestep), *with* bias
    * A version with subgraph is available in `braille_noDelay_bias_zero_subgraph.nir`
3. `braille_noDelay_noBias_subtract.nir`
    * Membrane reset by subtraction without delay (in the same timestep), *without* bias
    * A version with subgraph is available in `braille_noDelay_noBias_subtract_subgraph.nir`

Please use as many graphs as your platform supports. The more data in the paper, the better.

## Checklist per simulator

Examples are available in:
 - `Braille_training_.ipynb` to train a SRNN in snnTorch with optimized hyperparameters
 - `Braille_inference.ipynb` to perform inference with a pre-trained model and export the graph
 - `Norse inference.ipynb` to load a graph into Norse, make some analysis and inference 

### Procedure

0. Choose whether to use reset by subtraction or reset to zero for the CuBa-LIF neurons of the model
1. Load the two graphs above
   * **Note:** the graphs use `dt=1e-4`
3. Use the `ds_test.pt` subset (from the `data` folder) to provide test accuracy. Store that as a single number in `<PLATFORM>_accuracy_<MODEL>.npy` (e. g. `norse_accuracy_noDelay_bias_zero.nir`)
4. Pass the data from the first element in the `ds_test.pt` dataset (`torch.load("data/ds_test.pt")[0][0]`) through the models, and store the output spikes of the first LIF layer (the `"lif1.lif"` node in the graph). That data should have (256, 38) output for the `noDelay_bias_zero` graph and (256, 40) for the `noDelay_noBias_subtract` graph. Store that in `<PLATFORM>_activity_<MODEL>.npy`.


## *Additional information for training*

To train the Braille reading model in snnTorch, the `Braille_training.ipynb` notebook is available. By setting the `reset_mechanism`, `reset_delay` and `parameters_filename` variables, the different models (and corresponding hyperparameters) can be selected. `use_bias` allows instead to specify if bias can be used or not depending on the target platform. At the very beginning of the notebook, the `store_weights` can be set as True or False according to what is needed. A cell for GPU usage is also present.
