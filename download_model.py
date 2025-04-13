import nir

graph = nir.hub.download(
    model_id_or_name="braille-rnn",
    hub_url="http://localhost:8818",
)
print(graph.nodes.keys())