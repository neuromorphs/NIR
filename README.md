<picture>
<source media="(prefers-color-scheme: dark)" srcset="https://github.com/neuromorphs/NIR/raw/main/docs/logo_dark.png">
<img alt="NIR Logo" src="https://github.com/neuromorphs/NIR/raw/main/docs/logo_light.png">
</picture>

# NIR - Neuromorphic Intermediate Representation

NIR is a set of computational primitives, shared across different neuromorphic frameworks and technology stacks.
**NIR is currently supported by 7 simulators and 4 hardware platforms**, allowing users to seamlessly move between any of these platforms.
The goal of NIR is to decouple the evolution of neuromorphic hardware and software, ultimately increasing the interoperability between platforms and improving accessibility to neuromorphic technologies.

## Installation
NIR is installable via [pip](https://pypi.org/)
```bash 
pip install nir
```

Check your [local framework](https://nnir.readthedocs.io/en/latest/support.html) for NIR support.

## Usage
> Read more in our [documentation about NIR usage](https://nnir.readthedocs.io/en/latest/usage.html)

To end-users, NIR is just a declarative format that sits between formats and will hopefully be as invisible as possible.
However, it is possible to export Python objects or NIR files.

```python
import nir
# Write to file
nir.write("my_graph.nir", nir_graph) 

# Read file
imported_graph = nir.read("my_graph.nir")
```

## About NIR
> Read more in our [documentation about NIR primitives](https://nnir.readthedocs.io/en/latest/primitives.html)

On top of popular primitives such as convolutional or fully connected/linear computations, we define additional compuational primitives that are specific to neuromorphic computing and hardware implementations thereof. 
Computational units that are not specifically neuromorphic take inspiration from the Pytorch ecosystem in terms of naming and parameters (such as Conv2d that uses groups/strides).


## Frameworks that currently support NIR

| **Framework** | **Write to NIR** | **Read from NIR** | **Examples** |
| --------------- | :--: | :--: | :------: |
| [Lava-DL](https://github.com/lava-nc/lava-dl) | ✓ | ✓ | [Lava example](https://nnir.readthedocs.io/en/latest/examples/lava/nir-conversion.html)
| [Nengo](https://nengo.ai) | ✓ | ✓ | [Nengo example](https://nnir.readthedocs.io/en/latest/examples/nengo/nir-conversion.html) |
| [Norse](https://github.com/norse/norse) | ✓ | ✓ | [Norse example](https://nnir.readthedocs.io/en/latest/examples/norse/nir-conversion.html) |
| [Rockpool](https://rockpool.ai) ([SynSense Xylo chip](https://www.synsense.ai/products/xylo/)) | ✓ | ✓ | [Rockpool example](https://nnir.readthedocs.io/en/latest/examples/rockpool/nir-conversion.html)
| [Sinabs](https://sinabs.readthedocs.io) ([SynSense Speck chip](https://www.synsense.ai/products/speck-2/)) | ✓ | ✓ | [Sinabs example](https://nnir.readthedocs.io/en/latest/examples/sinabs/nir-conversion.html)
| [snnTorch](https://github.com/jeshraghian/snntorch/) | ✓ | ✓ | [snnTorch example](https://nnir.readthedocs.io/en/latest/examples/snntorch/nir-conversion.html)
| [SpiNNaker2](https://spinncloud.com/portfolio/spinnaker2/) | ⬚ | ✓ | [SpiNNaker2 example](https://nnir.readthedocs.io/en/latest/examples/spinnaker2/import.html)
| [Spyx](https://github.com/kmheckel/spyx) | ⬚ | ✓ | [Spyx example](https://nnir.readthedocs.io/en/latest/examples/spyx/conversion.ipynb)


## Acknowledgements
This work was originally conceived at the [Telluride Neuromorphic Workshop 2023](tellurideneuromorphic.org) by the authors below (in alphabetical order):
* [Steven Abreu](https://github.com/stevenabreu7)
* [Felix Bauer](https://github.com/bauerfe)
* [Jason Eshraghian](https://github.com/jeshraghian)
* [Matthias Jobst](https://github.com/matjobst)
* [Gregor Lenz](https://github.com/biphasic)
* [Jens Egholm Pedersen](https://github.com/jegp)
* [Sadique Sheik](https://github.com/sheiksadique)
* [Peng Zhou](https://github.com/pengzhouzp)

If you use NIR in your work, please cite the [following arXiv preprint](https://arxiv.org/abs/2311.14641)

```
@inproceedings{NIR2023,
  title={Neuromorphic Intermediate Representation: A Unified Instruction Set for Interoperable Brain-Inspired Computing},
  author={Jens E. Pedersen and Steven Abreu and Matthias Jobst and Gregor Lenz and Vittorio Fra and Felix C. Bauer and Dylan R. Muir and Peng Zhou and Bernhard Vogginger and Kade Heckel and Gianvito Urgese and Sadasivan Shankar and Terrence C. Stewart and Jason K. Eshraghian and Sadique Sheik},
  year={2023},
  doi={https://doi.org/10.48550/arXiv.2311.14641}
  archivePrefix={arXiv},
  primaryClass={cs.NE}
}
```