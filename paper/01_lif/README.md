# Dynamics of a single LIF neuron

The LIF example is run on all hardware platforms and all simulators.

The NIR graph of the LIF neuron is created in Norse, see `lif_norse.ipynb`, and exported to `lif_norse.nir`. The `lif_norse.nir` file is imported by every other platform in the respective `lif_{platform}.{ipynb|py}` file. The voltage and spike traces are exported to the respective `lif_{platform}.csv` file. Each platform may also export their loaded graph to NIR again into `lif_{platform}.nir`, and export a plot of the voltage and spike traces to `lif_{platform}.png`. 

We also provide an exact simulation of the LIF dynamics in `lif_exact_sim.ipynb`, with the respective time traces in `lif_exact.csv`.

The final comparison of LIF dynamics across all platforms is created in `lif_comparison.ipynb` and stored in `../figures/lif_comparison.pdf`.

- [x] Lava (CPU fixed, CPU float, Loihi fixed)
- [x] Nengo
- [x] Norse
- [x] Rockpool
- [x] Sinabs
- [x] snnTorch
- [x] SpiNNaker2
- [x] Spyx

*Note: We use a timestep of dt=0.0001*
