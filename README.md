<picture>
<source media="(prefers-color-scheme: dark)" srcset="https://github.com/neuromorphs/NIR/raw/main/docs/logo_dark.png">
<img alt="NIR Logo" src="https://github.com/neuromorphs/NIR/raw/main/docs/logo_light.png">
</picture>

# NIR - Neuromorphic Intermediate Representation

[![Nature Communications Paper](https://zenodo.org/badge/DOI/10.1038/s41467-024-52259-9.svg)](https://doi.org/10.1038/s41467-024-52259-9)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/nir?logo=pypi)](https://pypi.org/project/nir/)
[![GitHub Tag](https://img.shields.io/github/v/tag/neuromorphs/nir?logo=github)](https://github.com/neuromorphs/NIR/releases)
[![Discord](https://img.shields.io/discord/1209533869733453844?logo=discord)](https://discord.gg/JRMRGP9h3c)

NIR is a set of computational primitives, shared across different neuromorphic frameworks and technology stacks.
**NIR is currently supported by 7 simulators and 4 hardware platforms**, allowing users to seamlessly move between any of these platforms.

NIR is useful when you want to move a model from one platform to another, for instance from a simulator to a hardware platform.

> Read more about NIR in our [documentation about NIR primitives](https://neuroir.org/docs/primitives.html)

> See [which frameworks are currently supported by NIR](https://neuroir.org/docs/support.html).

## Usage
> Read more in our [documentation about NIR usage](https://neuroir.org/docs) and see more examples in our [examples section](https://neuroir.org/docs/examples)

NIR serves as a format between neuromorphic platforms and will be installed alongside your framework of choice.
Using NIR is typically a part of your favorite framework's workflow, but follows the same pattern when you want to move from a *source* to a *target* platform:

```python
# Define a model
my_model = ...
# Save the model (source platform)
nir.write("my_graph.nir", my_model) 
# Load the model (target platform)
imported_graph = nir.read("my_graph.nir")
```

See our [example section](https://neuroir.org/docs/examples) for how to use NIR with your favorite framework.

## Frameworks that currently support NIR
> Read more in our [documentation about NIR support](https://neuroir.org/docs/support.html)

| **Framework** | **Write to NIR** | **Read from NIR** | **Examples** |
| --------------- | :--: | :--: | :------: |
| [Lava-DL](https://github.com/lava-nc/lava-dl) | ✓ | ⬚ | [Lava/Loihi examples](https://neuroir.org/docs/examples/lava/nir-conversion.html) |
| [Nengo](https://nengo.ai) | ✓ | ✓ | [Nengo examples](https://neuroir.org/docs/examples/nengo/nir-conversion.html) |
| [Norse](https://github.com/norse/norse) | ✓ | ✓ | [Norse examples](https://neuroir.org/docs/examples/norse/nir-conversion.html) |
| [Rockpool](https://rockpool.ai) ([SynSense Xylo chip](https://www.synsense.ai/products/xylo/)) | ✓ | ✓ | [Rockpool/Xylo examples](https://neuroir.org/docs/examples/rockpool/nir-conversion.html)
| [Sinabs](https://sinabs.readthedocs.io) ([SynSense Speck chip](https://www.synsense.ai/products/speck-2/)) | ✓ | ✓ | [Sinabs/Speck examples](https://neuroir.org/docs/examples/sinabs/nir-conversion.html) |
| [snnTorch](https://github.com/jeshraghian/snntorch/) | ✓ | ✓ | [snnTorch examples](https://neuroir.org/docs/examples/snntorch/nir-conversion.html) |
| [SpiNNaker2](https://spinncloud.com/portfolio/spinnaker2/) | ⬚ | ✓ | [SpiNNaker2 examples](https://neuroir.org/docs/examples/spinnaker2/import.html) |
| [Spyx](https://github.com/kmheckel/spyx) | ✓ | ✓ | [Spyx examples](https://neuroir.org/docs/examples/spyx/conversion.html)


## Acknowledgements
This work was originally conceived at the [Telluride Neuromorphic Workshop 2023](https://tellurideneuromorphic.org) by the authors below (in alphabetical order):
* [Steven Abreu](https://github.com/stevenabreu7)
* [Felix Bauer](https://github.com/bauerfe)
* [Jason Eshraghian](https://github.com/jeshraghian)
* [Matthias Jobst](https://github.com/matjobst)
* [Gregor Lenz](https://github.com/biphasic)
* [Jens Egholm Pedersen](https://github.com/jegp)
* [Sadique Sheik](https://github.com/sheiksadique)
* [Peng Zhou](https://github.com/pengzhouzp)

If you use NIR in your work, please cite the [following paper](https://www.nature.com/articles/s41467-024-52259-9)

```
article{NIR2024, 
    title={Neuromorphic intermediate representation: A unified instruction set for interoperable brain-inspired computing}, 
    author={Pedersen, Jens E. and Abreu, Steven and Jobst, Matthias and Lenz, Gregor and Fra, Vittorio and Bauer, Felix Christian and Muir, Dylan Richard and Zhou, Peng and Vogginger, Bernhard and Heckel, Kade and Urgese, Gianvito and Shankar, Sadasivan and Stewart, Terrence C. and Sheik, Sadique and Eshraghian, Jason K.}, 
    rights={2024 The Author(s)},
    DOI={10.1038/s41467-024-52259-9}, 
    number={1},
    journal={Nature Communications}, 
    volume={15},
    year={2024}, 
    month=sep, 
    pages={8122},
}
```
