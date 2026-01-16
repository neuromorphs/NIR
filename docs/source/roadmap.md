# NIR roadmap

NIR is in its early stages but is developing fast.
Here, we present an organizational roadmap for the immediate future of NIR governance, followed by a list of concrete projects that are currently developing NIR.

```{admonition} Community involvement
:class: tip
We are dedicated to interacting and paying back to the community and we are continuously improving the accessibility of NIR by (1) releasing educational material, (2) updating our docs, and (3) engaging with the community in open events.

For more information, see our [events](events) page and join the discussion on our [Discord](https://discord.gg/JRMRGP9h3c) server.
```


## Active projects

The following projects are concrete initiatives that are working to improve NIR.
Feel free to reach out to the project owners if you are interested in contributing or if you have any projects you would like to add.

### Extending SpiNNaker 2 support
**Owner**: [Bernhard Vogginger](https://github.com/bvogginger) </br>
**Description**: The current implementation for SpiNNaker2 is static in the sense that we cannot stream input to the chip. We are working to support streaming inputs and, at the same time, speedup SNN processing. More information can be found at the [SpiNNaker 2 project milestone 5](https://gitlab.com/spinnaker2/py-spinnaker2/-/milestones/5).

### Adding Nengo support
**Owner**: [P. Michael Furlong](https://furlong.gitlab.io/) </br>
**Description**: We are working to add Nengo support to NIR. This will allow users to export Nengo networks to NIR and for NIR users to run graphs via Nengo-supported hardware, such as Loihi and CPU. More information can be found at the [Nengo GitHub repository](https://furlong.gitlab.io/)

## Adding SpiNNaker 1 support
**Owner**: [Andrew Rowley](https://research.manchester.ac.uk/en/persons/andrew-rowley/) </br>
**Description**: We are working to add SpiNNaker 1 support to NIR. This will allow users to export NIR graphs to SpiNNaker 1 hardware. More information can be found at the [SpiNNaker 1 GitHub repository](https://spinnakermanchester.github.io/)

### Energy comparisons on different computational substrates
**Owner**: [Sadasivan Shankar](https://profiles.stanford.edu/sadasivan-shankar) </br>
**Description**: It would be helpful to understand exactly how much energy we can save by lifting a NIR graph to neuromorphic hardware. In this project, we are developing a mapping from NIR graph to energy efficiency, given a computational platform. That is, we can take a NIR graph and calculate a rough estimate of the energy consumption of the graph on a given platform. This will help us to understand the potential energy savings **before** it is deployed.

### NIRData: an exchange format for data
**Owner**: [Ben Kroehs](https://github.com/benkroehs/) </br>
**Description**: To train specialized neuromorphic hardware, we often need to cycle back and forth between NIR graphs and data. Particularly for learning with hardware-in-the-loop, NIRData will be helpful: a standardized format for exchanging data, coupled to individual computational nodes in the NIR graph.

### Optimizing NIR graphs
**Description**: We are working on optimizing NIR graphs to reduce the number of primitives and connections. This will help to reduce the energy consumption of the graph and make it more efficient to run on neuromorphic hardware.