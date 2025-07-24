# Supported simulators and hardware

**NIR is currently supported by 8 simulators and 5 hardware platforms**, allowing users to seamlessly move between any of these platforms.
The 8 simulators include jaxsnn, Lava-DL, Nengo, Norse, Rockpool, Sinabs, snnTorch, and Spyx.
The 5 hardware platforms include BrainScaleS-2, Intel Loihi (via Lava-DL), Speck, SpiNNaker2, and Xylo.

The table below shows the integration progress for the respective frameworks.
By "reading" a NIR graph, we mean converting it into a platform-specific representation.
"Writing" a NIR graph means converting a platform-specific representation into a NIR graph.

| **Framework** | **Write to NIR** | **Read from NIR** | **Examples** |
| --------------- | :--: | :--: | :------: |
| [jaxsnn](https://github.com/electronic-visions/jaxsnn) ([BrainScaleS-2](https://wiki.ebrains.eu/bin/view/Collabs/neuromorphic/BrainScaleS/)) | ⬚ | ✓ | [jaxsnn examples](https://neuroir.org/docs/examples/jaxsnn/nir-conversion.html) |
| [Lava-DL](https://github.com/lava-nc/lava-dl) | ✓ | ⬚ | [Lava/Loihi examples](https://neuroir.org/docs/examples/lava/nir-conversion.html) |
| [Nengo](https://nengo.ai) | ✓ | ✓ | [Nengo examples](https://neuroir.org/docs/examples/nengo/nir-conversion.html) |
| [Norse](https://github.com/norse/norse) | ✓ | ✓ | [Norse examples](https://neuroir.org/docs/examples/norse/nir-conversion.html) |
| [Rockpool](https://rockpool.ai) ([SynSense Xylo chip](https://www.synsense.ai/products/xylo/)) | ✓ | ✓ | [Rockpool/Xylo examples](https://neuroir.org/docs/examples/rockpool/nir-conversion.html)
| [Sinabs](https://sinabs.readthedocs.io) ([SynSense Speck chip](https://www.synsense.ai/products/speck-2/)) | ✓ | ✓ | [Sinabs/Speck examples](https://neuroir.org/docs/examples/sinabs/nir-conversion.html) |
| [snnTorch](https://github.com/jeshraghian/snntorch/) | ✓ | ✓ | [snnTorch examples](https://neuroir.org/docs/examples/snntorch/nir-conversion.html) |
| [SpiNNaker2](https://spinncloud.com/portfolio/spinnaker2/) | ⬚ | ✓ | [SpiNNaker2 examples](https://neuroir.org/docs/examples/spinnaker2/import.html) |
| [Spyx](https://github.com/kmheckel/spyx) | ✓ | ✓ | [Spyx examples](https://neuroir.org/docs/examples/spyx/conversion.html)

## Why are some platforms only reading or writing but not both?
Some platforms support both reading and writing, but in other cases it does not make sense to both read *and* write NIR graphs.
For example hardware platforms are meant as a *runtime* of NIR graphs, so it rarely makes sense to convert the hardware representation back into NIR.

## What about other simulators and hardware platforms?
NIR is a recent invention, and we are working hard to integrate it with as many simulators and hardware platforms as possible.
If you know of a simulator or hardware accelerator we should include, please get in touch with us here on GitHub.
