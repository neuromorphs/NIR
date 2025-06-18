"""Server implementation for the NIR model hub.

This module provides a minimal server implementation for hosting NIR models.
"""

import json
import os
import pathlib
import shutil
import tempfile
import uuid
import logging
from typing import Dict, List, Optional, Union, Any

from flask import Flask, jsonify, request, send_file

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('nir.hub.server')

# Storage location for models
STORAGE_DIR = os.path.expanduser("~/.nir/hub/models")
INDEX_PATH = os.path.join(os.path.dirname(STORAGE_DIR), "model_index.json")

# Create the app
app = Flask(__name__)

# Create storage directory if it doesn't exist
os.makedirs(STORAGE_DIR, exist_ok=True)

# Load model index from disk if it exists
model_index = {}
try:
    if os.path.exists(INDEX_PATH):
        with open(INDEX_PATH, 'r') as f:
            model_index = json.load(f)
        logger.info(f"Loaded {len(model_index)} models from index")
except Exception as e:
    logger.error(f"Error loading model index: {e}")
    # Continue with empty index

# Function to save the index
def save_index():
    """Save the model index to disk."""
    try:
        with open(INDEX_PATH, 'w') as f:
            json.dump(model_index, f, indent=2)
        logger.info(f"Saved {len(model_index)} models to index")
    except Exception as e:
        logger.error(f"Error saving model index: {e}")


@app.route("/api/models", methods=["GET"])
def list_models():
    """List all available models."""
    return jsonify(list(model_index.values()))


@app.route("/api/models", methods=["POST"])
def upload_model():
    """Upload a new model to the hub."""
    # Check for required files
    if "nir_file" not in request.files or "config_file" not in request.files:
        return jsonify({"error": "Missing required files: nir_file and config_file"}), 400
    
    nir_file = request.files["nir_file"]
    config_file = request.files["config_file"]
    
    # Parse config
    try:
        config = json.load(config_file)
    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON in config file"}), 400
    
    # Validate config
    if "model_name" not in config or "nir_version" not in config:
        return jsonify({"error": "Config must contain 'model_name' and 'nir_version'"}), 400
    
    # Generate unique ID for model
    model_id = str(uuid.uuid4())
    config["model_id"] = model_id
    
    # Create model directory
    model_dir = os.path.join(STORAGE_DIR, model_id)
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model files
    model_path = os.path.join(model_dir, f"{config['model_name']}.nir")
    config_path = os.path.join(model_dir, "config.json")
    
    nir_file.save(model_path)
    
    # Save updated config
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    # Add to index
    model_index[model_id] = config
    
    # Save index to disk
    save_index()
    
    return jsonify(config), 201


@app.route("/api/models/<model_id_or_name>", methods=["GET"])
def get_model(model_id_or_name):
    """Get model information by ID or name."""
    logger.info(f"Looking up model: {model_id_or_name}")
    logger.debug(f"Current index has {len(model_index)} models: {list(model_index.keys())}")
    
    # Search by ID
    if model_id_or_name in model_index:
        logger.info(f"Found model by ID: {model_id_or_name}")
        return jsonify(model_index[model_id_or_name])
    
    # Search by name
    for model in model_index.values():
        if model["model_name"] == model_id_or_name:
            logger.info(f"Found model by name: {model_id_or_name}")
            return jsonify(model)
    
    logger.warning(f"Model not found: {model_id_or_name}")
    return jsonify({"error": "Model not found", "available_models": list(model_index.keys())}), 404


@app.route("/api/models/<model_id_or_name>/download", methods=["GET"])
def download_model(model_id_or_name):
    """Download a model file."""
    logger.info(f"Download request for model: {model_id_or_name}")
    logger.debug(f"Current index has {len(model_index)} models: {list(model_index.keys())}")
    
    # First, get model info
    model_info = None
    if model_id_or_name in model_index:
        logger.info(f"Found model by ID for download: {model_id_or_name}")
        model_info = model_index[model_id_or_name]
    else:
        # Search by name
        for model in model_index.values():
            if model["model_name"] == model_id_or_name:
                logger.info(f"Found model by name for download: {model_id_or_name}")
                model_info = model
                break
    
    if model_info is None:
        logger.warning(f"Model not found for download: {model_id_or_name}")
        return jsonify({"error": "Model not found", "available_models": list(model_index.keys())}), 404
    
    # Construct path to model file
    model_path = os.path.join(STORAGE_DIR, model_info["model_id"], f"{model_info['model_name']}.nir")
    logger.info(f"Model file path: {model_path}")
    
    # Check if file exists
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at path: {model_path}")
        return jsonify({"error": "Model file not found at expected path"}), 404
    
    # Send file
    logger.info(f"Sending model file: {model_path}")
    return send_file(model_path, as_attachment=True, download_name=f"{model_info['model_name']}.nir")


@app.route("/api/models/<model_id_or_name>", methods=["DELETE"])
def delete_model(model_id_or_name):
    """Delete a model from the hub."""
    # Find the model
    model_id = None
    if model_id_or_name in model_index:
        model_id = model_id_or_name
    else:
        # Search by name
        for id, model in model_index.items():
            if model["model_name"] == model_id_or_name:
                model_id = id
                break
    
    if model_id is None:
        return jsonify({"error": "Model not found"}), 404
    
    # Remove from index
    model_info = model_index.pop(model_id)
    
    # Save index to disk
    save_index()
    
    # Delete files
    model_dir = os.path.join(STORAGE_DIR, model_id)
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    
    return jsonify({"message": f"Model '{model_info['model_name']}' deleted"})


@app.route("/api/models/search", methods=["GET"])
def search_models():
    """Search for models based on query parameters."""
    # Get query parameters
    tag = request.args.get("tag")
    framework = request.args.get("framework")
    platform = request.args.get("platform")
    
    # Filter models
    results = list(model_index.values())
    
    if tag:
        results = [m for m in results if "tags" in m and tag in m["tags"]]
    
    if framework:
        results = [m for m in results if "framework_origin" in m and framework == m["framework_origin"]]
    
    if platform:
        results = [m for m in results if "compatible_platforms" in m and platform in m["compatible_platforms"]]
    
    return jsonify(results)


def run_server(host="0.0.0.0", port=8080, debug=True):
    """Run the NIR hub server."""
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run_server()
