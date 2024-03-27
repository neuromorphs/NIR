# NIR roadmap

NIR is in its early stages but is developing fast.
Here, we present an organizational roadmap for the immediate future of NIR governance, followed by a list of concrete projects that are currently developing NIR.

## Organizational roadmap

With [the NIR preprint](https://arxiv.org/abs/2311.14641), we have demonstrated the feasibility of NIR.
Not only as a scientific idea, but as a practical and valuable tool for the development of neuromorphic systems.
However, NIR has been born in a research environment.
To make it self-sustaining, we are preparing to launch a series of projects that will ensure the continued development of NIR, while continuing to increase its value to the community.

The most important pillar of NIR is the joint value for both industrial and academic partners alike.
Therefore, the first phase of our roadmap will be to reach out to potential partners and invite them to joint projects for which we can raise funds---either through commercial or public sources.
The coming phases will revolve around the fundraising and organization of these projects.
Eventually, we will kick off the projects and establish a steering group for NIR as the projects mature and further direction is needed.

```
 _______________       _______________      _______________      _______________ 
|    Phase 1    | ->  |    Phase 2    | -> |    Phase 3    | -> |    Phase 4    | 
|   May  2024   |     |  July  2024   |    |   Sep. 2024   |    |   Dec. 2024   | 
|   Outreach    |     |  Fundraising  |    |   Kickoff     |    | Steering grp. | 
|_______________|     |_______________|    |_______________|    |_______________|  
```

## Active projects

The follwing projects are concrete intiatives that are working to improve NIR.
Feel free to reach out to the project owners if you are interested in contributing or if you have any projects you would like to add.

### Improved SpiNNaker 2 support
**Duration**: August 2024 </br>
**Owner**: [Bernhard Vogginger](https://github.com/bvogginger) </br>
**Description**: We are working on improving the support for the SpiNNaker 2 platform. This includes speeding up SNN processing with the machine learning accelerator,  supporting more neuron models and streaming data.

## Improved Lava & Loihi support
**Duration**: October 2024 </br>
**Owner**: [Steven Abreu](https://github.com/stevenabreu7/) </br>
**Description**: We are working on improving the support for the Lava and Loihi platforms, focusing on neuron model support and applications.

### Energy comparisons on different computational substrates
**Duration**: Until July 2024 </br>
**Owner**: [Sadasivan Shankar](https://profiles.stanford.edu/sadasivan-shankar) </br>
**Description**: It would be helpful to understand exactly how much energy we can save by lifting a NIR graph to neuromrophic hardware. In this project, we are developing a mapping from NIR graph to energy efficiency, given a computational platform. That is, we can take a NIR graph and calculate a rough estimate of the energy consumption of the graph on a given platform. This will help us to understand the potential energy savings **before** it is deployed.

## Additional neuron models and plasticity
**Timeline**: Start October 2024 </br>
**Owner**: [Jens Egholm Pedersen](https://github.com/jegp) </br>
**Description**: We are working on adding additional neuron models and plasticity rules to NIR, including the Fitzhugh-Nagumo model. Plasticity will require additional primitives that can, themselves, modify primitive parameters.


