# Supported simulators and hardware

**NIR is currently supported by 6 simulators and 4 hardware platforms**, allowing users to seamlessly move between any of these platforms.
The 6 simulators include Lava-DL, Nengo, Norse, Rockpool, Sinabs, and snnTorch.
The 4 hardware platforms include Intel Loihi (via Lava-DL), Xylo, Speck, and SpiNNaker2.

The table below shows the integration progress for the respective frameworks.
By "reading" a NIR graph, we mean converting it into a platform-specific representation.
"Writing" a NIR graph means converting a platform-specific representation into a NIR graph.

| **Framework** | **Write to NIR** | **Read from NIR** | **Examples** |
| --------------- | :--: | :--: | :------: |
| [Lava-DL](https://github.com/lava-nc/lava-dl) | ✓ | ⬚ |
| [Nengo](https://nengo.ai) | ✓ | ✓ | [Nengo examples](https://github.com/neuromorphs/nir/tree/main/example/nengo/) |
| [Norse](https://github.com/norse/norse) | ✓ | ✓ | [Norse examples](https://github.com/neuromorphs/nir/tree/main/example/norse/) |
| [Rockpool](https://rockpool.ai) ([SynSense Xylo chip](https://www.synsense.ai/products/xylo/)) | ✓ | ✓ |
| [Sinabs](https://sinabs.readthedocs.io) ([SynSense Speck chip](https://www.synsense.ai/products/speck-2/)) | ✓ | ✓ |
| [snnTorch](https://github.com/jeshraghian/snntorch/) | ✓ | ⬚ |
| [SpiNNaker2](https://spinncloud.com/portfolio/spinnaker2/) | ⬚ | ✓ |

## Why are some platforms only reading or writing but not both?
Some platforms support both reading and writing, but in other cases it does not make sense to both read *and* write NIR graphs.
For example hardware platforms are meant as a *runtime* of NIR graphs, so it rarely makes sense to convert the hardware representation back into NIR.

## What about other simulators and hardware platforms?
NIR is a recent invention, and we are working hard to integrate it with as many simulators and hardware platforms as possible.
If you know of a simulator or hardware accelerator we should include, please get in touch with us here on GitHub.