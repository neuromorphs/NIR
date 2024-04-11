"""Flatten Braille graph for SpiNNaker2 and fix some shapes."""

import nir

nir_model = nir.read("braille.nir")

nodes_to_copy = {
    "fc1": "fc1",
    "fc2": "fc2",
    "input": "input",
    "lif1.lif": "lif1",
    "lif1.w_rec": "w_rec",
    "lif2": "lif2",
    "output": "output",
}

new_edges = [
    # forward
    ("input", "fc1"),
    ("fc1", "lif1"),
    ("lif1", "fc2"),
    ("fc2", "lif2"),
    ("lif2", "output"),
    # recurrent
    ("lif1", "w_rec"),
    ("w_rec", "lif1"),
]

new_nodes = {}
for old_name, new_name in nodes_to_copy.items():
    new_nodes[new_name] = nir_model.nodes[old_name]

new_model = nir.NIRGraph(new_nodes, new_edges)

# modify shapes of input node
input_node = new_model.nodes["input"]

old_input_shape = input_node.input_type["input"]
if old_input_shape.size == 2:
    input_node.input_type["input"] = old_input_shape[1:]

old_output_shape = input_node.output_type["output"]
if old_output_shape.size == 2:
    input_node.output_type["output"] = old_output_shape[1:]

print("nodes:")
for nodekey, node in new_model.nodes.items():
    print(
        "\t",
        nodekey,
        node.__class__.__name__,
        node.input_type["input"],
        node.output_type["output"],
    )
print("edges:")
for edge in new_model.edges:
    print("\t", edge)

nir.write("braille_flattened.nir", new_model)
