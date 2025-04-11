"""Client-side functionality for the NIR model hub."""

import json
import os
import pathlib
import tempfile
from typing import Dict, List, Optional, Union, Any

import requests

import nir
from nir import NIRGraph
from .version import check_compatibility as check_compatibility_fn


def upload(
    graph: NIRGraph,
    model_name: str,
    description: str = "",
    tags: List[str] = None,
    framework_origin: str = "",
    compatible_platforms: List[str] = None,
    hub_url: str = "http://localhost:8080",
) -> requests.Response:
    """Upload a NIR graph to the hub.
    
    Args:
        graph: The NIR graph to upload
        model_name: A descriptive name for the model
        description: Optional detailed description
        tags: Optional list of tags for categorization
        framework_origin: Optional framework used to create the model
        compatible_platforms: Optional list of platforms tested with this model
        hub_url: URL of the hub server (default: local server)
        
    Returns:
        The response object from the upload request
    """
    if tags is None:
        tags = []
    if compatible_platforms is None:
        compatible_platforms = []
    
    # Create model config
    config = {
        "nir_version": nir.version,
        "model_name": model_name,
        "description": description,
        "tags": tags,
        "framework_origin": framework_origin,
        "compatible_platforms": compatible_platforms,
    }
    
    # Create temporary files for upload
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save NIR graph
        nir_path = os.path.join(temp_dir, f"{model_name}.nir")
        nir.write(nir_path, graph)
        
        # Save config
        config_path = os.path.join(temp_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        
        # Prepare files for upload
        files = {
            "nir_file": open(nir_path, "rb"),
            "config_file": open(config_path, "rb"),
        }
        
        # Upload to hub
        try:
            response = requests.post(
                f"{hub_url}/api/models",
                files=files,
            )
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            raise ConnectionError(f"Failed to upload model to hub: {e}")
        finally:
            # Close file handlers
            for f in files.values():
                f.close()


def download(
    model_id_or_name: str,
    output_dir: Optional[str] = None,
    check_compatibility: bool = True,
    hub_url: str = "http://localhost:8080",
) -> NIRGraph:
    """Download a NIR graph from the hub.
    
    Args:
        model_id_or_name: ID or name of the model to download
        output_dir: Optional directory to save the downloaded files
        check_compatibility: Whether to check version compatibility
        hub_url: URL of the hub server (default: local server)
        
    Returns:
        The downloaded NIR graph
    """
    try:
        # Get model metadata first
        response = requests.get(f"{hub_url}/api/models/{model_id_or_name}")
        response.raise_for_status()
        model_info = response.json()
        
        # Download model file
        nir_response = requests.get(f"{hub_url}/api/models/{model_id_or_name}/download")
        nir_response.raise_for_status()
        
        # Create temp directory for downloads if output_dir not specified
        if output_dir is None:
            temp_dir = tempfile.mkdtemp()
            output_dir = temp_dir
        else:
            os.makedirs(output_dir, exist_ok=True)
        
        # Save downloaded files
        nir_path = os.path.join(output_dir, f"{model_info['model_name']}.nir")
        with open(nir_path, "wb") as f:
            f.write(nir_response.content)
            
        # Save config
        config_path = os.path.join(output_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(model_info, f, indent=2)
        
        # Check compatibility if requested
        if check_compatibility:
            model_version = model_info.get("nir_version")
            if not check_compatibility_fn(model_version, nir.version):
                print(f"WARNING: Model version {model_version} may not be compatible with current NIR version {nir.version}")
        
        # Load the NIR graph
        return nir.read(nir_path)
        
    except requests.RequestException as e:
        raise ConnectionError(f"Failed to download model from hub: {e}")
