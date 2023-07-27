# SpiNNaker2 <-- NIR

SpiNNaker2 is a neuromorphic chip based around a grid of ARM Cortex-M4F processors which are tighly coupled with accelerators and a network-on-chip optimized for, but not limited to transmission of spikes.

Running this requires the following library and installed SpiNNaker2 hardware: [py-spinnaker2](https://gitlab.com/spinnaker2/py-spinnaker2)

Examples:
* Create a graph first: [generate_model.py](https://github.com/neuromorphs/nir/tree/main/example/spinnaker2/generate_model.py)
* Import a NIR graph to SpiNNaker2 and run it on the hardware: [import_to_spinnaker2.py](https://github.com/neuromorphs/nir/tree/main/example/spinnaker2/import_to_spinnaker2.py)

## Credits

* Matthias Jobst