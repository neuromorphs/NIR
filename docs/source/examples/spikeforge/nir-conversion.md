# spikeforge

[spikeforge](https://github.com/Capsize-Games/spikeforge) is a toolkit for
building, training, and deploying spiking neural networks. Its NIR bridge can
export a topology, write the graph to a NIR file, and compare the topology's
output with an independent NIR interpreter.

Install spikeforge with its NIR dependencies before running the example:

```shell
pip install "spikeforge[nir]"
```

## Export a topology to NIR

This example uses SpikeForge's deterministic `conv_net` fixture. It does not
train a model or download a dataset.

```python
import nir

from spikeforge.cli import fixture
from spikeforge.nir_bridge import graph_summary, to_nir

topology = "conv_net"
spec, module, spikes = fixture.synthetic_input(
    topology, steps=4, batch=1, seed=0
)

graph = to_nir(spec, module)
nir.write("conv_net.nir", graph)

summary = graph_summary(graph)
print("topology:", topology, "| input:", tuple(spikes.shape))
print("nodes:", len(summary["nodes"]))
print("kinds:", [node["kind"] for node in summary["nodes"]])
```

`graph_summary` returns the node and edge information without tensor objects,
so it can also be used when a graph needs to be inspected or sent over a
protocol.

## Validate the exported graph

SpikeForge can run the same input through the source module and the exported
graph. The validation report includes per-layer drift and an overall
`within_tolerance` result.

```python
from spikeforge.nir_bridge import validate

report = validate(spec, module, spikes, graph=graph)
print("within_tolerance:", report["within_tolerance"])
print("worst:", report["worst"])
```

The `graph` argument makes the comparison use the graph created in the first
section. The interpreter is separate from the source module, so this checks
the exported representation rather than only checking that export succeeded.
