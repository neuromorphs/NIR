# About

The Neuromorphic Intermediate Representation (NIR) is a joint effort between researchers and engineers working to spread the usage and accelerate the innnovation of neuromorphic technologies.
It is a product of many clever actors, thinking deeply about the interplay between computation and physics, and it would be incorrect to claim that NIR is owned by any single actor.
Rather, NIR is a community effort that serves to simplify the development of neuromorphic algorithms and applications, and it is only possible because of the many contributors who put in years of work across many organizations and universities.

```{admonition} NIR Community 
Connect to our community via [Discord](https://discord.gg/JRMRGP9h3c) or one of our many [online events](events).
```

## Design principles
NIR is designed according to two founding principles:

1. Closeness to physics
    * We restrict ourselves to stay close to physical primitives, because that is what the world is made of. Any computation has to be embedded in a physical substrate, and we believe the power of NIR comes from the fact that we can work as close to physics as possible.
2. Balance between productivity and correctness
    * We are entering an area of research where scientific giants have made many lasting marks. We are aware that we will not get everything right in the first go, but we are compelled to try. We are equally compelled to keep learning and improving, and the NIR spec will change as we grow.

```{admonition} NIR Roadmap
See more about the development and progress of NIR in our [roadmap](roadmap).
```

## Connectivity 
Each computational unit is a node in a static graph.
Given 3 nodes $A$ which is a LIF node, $B$ which is a Linear node and $C$ which is another LIF node, we can define edges in the graph such as:

## Format
The intermediate represenation can be stored as hdf5 file, which benefits from compression. 

## Authors
Authors (in alphabetical order):
* [Steven Abreu](https://github.com/stevenabreu7)
* [Felix Bauer](https://github.com/bauerfe)
* [Jason Eshraghian](https://github.com/jeshraghian)
* [Matthias Jobst](https://github.com/matjobst)
* [Gregor Lenz](https://github.com/biphasic)
* [Jens Egholm Pedersen](https://github.com/jegp)
* [Sadique Sheik](https://github.com/sheiksadique)

If you use NIR in your work, please cite the [following Zenodo reference](https://zenodo.org/record/8105042)

```
@software{nir2023,
  author       = {Abreu, Steven and
                  Bauer, Felix and
                  Eshraghian, Jason and
                  Jobst, Matthias and
                  Lenz, Gregor and
                  Pedersen, Jens Egholm and
                  Sheik, Sadique},
  title        = {Neuromorphic Intermediate Representation},
  month        = jul,
  year         = 2023,
  publisher    = {Zenodo},
  version      = {0.2},
  doi          = {10.5281/zenodo.8105042},
  url          = {https://doi.org/10.5281/zenodo.8105042}
}
```
