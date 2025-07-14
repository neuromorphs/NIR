# NIR Hub

NIR Hub is a platform for sharing and discovering neuromorphic models in the NIR format. It allows researchers and developers to:

1. Upload their NIR models to a central repository
2. Download models created by others
3. Search for models by tags, framework origin, or compatible platforms

## Getting Started

NIR Hub can be used both programmatically through the Python API and via a command-line interface.

### Starting a Hub Server

For local testing or private hubs, you can start a server with:

```bash
python -m nir.hub.cli server --debug
```

This will start a server on [http://localhost:8080](http://localhost:8080).

You can specify a different port using the `--port` option (especially useful on macOS where port 5000 is reserved):

```bash
python -m nir.hub.cli server --port 9090 --debug
```

### Uploading a Model

```python
import nir

# Load or create a model
graph = nir.read("path/to/model.nir")  # or create one programmatically

# Upload to hub
response = nir.hub.upload(
    graph=graph,
    model_name="My Model",
    description="A detailed description of my model",
    tags=["vision", "classification"],
    framework_origin="snntorch",
    compatible_platforms=["norse", "nengo"],
    hub_url="http://localhost:9090",  # custom url for the hub
)
result = response.json()

print(f"Model uploaded with ID: {result['model_id']}")
```

### Downloading a Model

```python
import nir

# Download by ID or name
graph = nir.hub.download(
    model_id_or_name="My Model",  # or use the UUID
    output_dir="./downloaded_models",  # optional
    check_compatibility=True,  # default
    hub_url="http://localhost:9090",  # custom url for the hub
)

# Use the downloaded model
print(f"Downloaded model with {len(graph.nodes)} nodes")
```

## Command-Line Interface

NIR Hub provides a command-line interface for common operations:

```bash
# Upload a model
nir-hub upload path/to/model.nir --name "My Model" --description "Description"

# Download a model
nir-hub download "My Model" --output ./downloaded_models

# Search for models
nir-hub search --tag vision --framework snntorch

# Start a hub server
nir-hub server --host 0.0.0.0 --port 5000 --debug
```

## Version Compatibility

When downloading models, NIR Hub performs compatibility checks by default to ensure that the model version is compatible with your current NIR version. The compatibility rules are:

1. Major versions must match exactly
2. The model's minor version must be less than or equal to your current minor version
3. If minor versions match, the model's patch version must be less than or equal to your current patch version

You can disable compatibility checks with `check_compatibility=False` in the API or `--no-check` in the CLI.

## Use Cases

### Cross-Framework Collaboration

NIR Hub enables collaboration between users of different neuromorphic frameworks. For example:

- A researcher develops a model in SNNTorch
- They upload it to NIR Hub
- Another researcher downloads the model and uses it in Norse
- The model performs identically in both frameworks

### Model Zoo

NIR Hub can serve as a model zoo for neuromorphic networks, similar to how PyTorch Hub and TensorFlow Hub work for traditional neural networks.

### Reproducible Research

By sharing exact model configurations through NIR Hub, researchers can ensure that others can reproduce their results exactly, even if they prefer different frameworks.