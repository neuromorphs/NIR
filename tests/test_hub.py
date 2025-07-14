"""Tests for the NIR Hub functionality."""

import os
import tempfile
from unittest import mock

import pytest
import numpy as np

import nir
from nir import NIRGraph
from nir.hub.version import parse_version, check_compatibility
from tests import mock_affine


def test_parse_version():
    """Test version parsing."""
    assert parse_version("1.2.3") == (1, 2, 3)
    assert parse_version("0.1.0") == (0, 1, 0)
    assert parse_version("1.0") == (1, 0, 0)
    assert parse_version("2") == (2, 0, 0)
    assert parse_version("") == (0, 0, 0)
    assert parse_version("not.a.version") == (0, 0, 0)


def test_version_compatibility():
    """Test version compatibility checking."""
    # Same version
    assert check_compatibility("1.2.3", "1.2.3") is True
    
    # Compatible versions
    assert check_compatibility("1.0.0", "1.0.1") is True
    assert check_compatibility("1.0.0", "1.1.0") is True
    assert check_compatibility("1.0.0", "1.2.0") is True
    assert check_compatibility("1.1.0", "1.1.5") is True
    
    # Incompatible versions
    assert check_compatibility("1.0.0", "2.0.0") is False
    assert check_compatibility("1.2.0", "1.1.0") is False
    assert check_compatibility("1.1.5", "1.1.0") is False


def test_model_serialization():
    """Test model serialization and deserialization for hub usage."""
    # Create a simple sequential model using the mock_affine helper
    a = mock_affine(2, 3)
    b = mock_affine(3, 2)
    
    # Create a sequential model
    model = NIRGraph.from_list(a, b)
    
    # Save and load the model
    with tempfile.NamedTemporaryFile(suffix=".nir") as tmp:
        nir.write(tmp.name, model)
        loaded_model = nir.read(tmp.name)
        
        # Check that the model structure is preserved
        assert isinstance(loaded_model, NIRGraph)
        assert len(loaded_model.nodes) == len(model.nodes)
        assert len(loaded_model.edges) == len(model.edges)


@mock.patch("nir.hub.client.requests.post")
def test_upload(mock_post):
    """Test model upload functionality."""
    # Mock the response
    mock_response = mock.Mock()
    mock_response.json.return_value = {"model_id": "test-id", "model_name": "Test Model"}
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response
    
    # Create a simple sequential model using the mock_affine helper
    a = mock_affine(2, 3)
    b = mock_affine(3, 2)
    
    # Create a sequential model
    model = NIRGraph.from_list(a, b)
    
    # Upload the model
    result = nir.hub.upload(
        graph=model,
        model_name="Test Model",
        description="Test description",
        tags=["test"],
        framework_origin="pytest",
        compatible_platforms=["all"],
        hub_url="http://test-server",
    )
    
    # Check the result - the client.py function returns the response object
    assert mock_post.called
    # Verify the URL is correct in the call
    assert mock_post.call_args[0][0] == "http://test-server/api/models"


@mock.patch("nir.hub.client.requests.get")
def test_download(mock_get):
    """Test model download functionality."""
    # Create a simple sequential model using the mock_affine helper
    a = mock_affine(2, 3)
    b = mock_affine(3, 2)
    
    # Create a sequential model
    model = NIRGraph.from_list(a, b)
    
    # Save the model to use in the mock response
    with tempfile.NamedTemporaryFile(suffix=".nir") as tmp:
        nir.write(tmp.name, model)
        with open(tmp.name, "rb") as f:
            nir_content = f.read()
        
        # Mock the metadata response
        meta_response = mock.Mock()
        meta_response.json.return_value = {
            "model_id": "test-id",
            "model_name": "Test Model",
            "nir_version": nir.version,
        }
        meta_response.raise_for_status.return_value = None
        
        # Mock the download response
        download_response = mock.Mock()
        download_response.content = nir_content
        download_response.raise_for_status.return_value = None
        
        # Configure the mock to return different responses for different URLs
        def get_side_effect(url, *args, **kwargs):
            if url.endswith("/download"):
                return download_response
            return meta_response
        
        mock_get.side_effect = get_side_effect
        
        # Download the model
        with tempfile.TemporaryDirectory() as output_dir:
            downloaded_model = nir.hub.download(
                model_id_or_name="test-id",
                output_dir=output_dir,
                check_compatibility=True,
                hub_url="http://test-server",
            )
            
            # Check the result
            assert isinstance(downloaded_model, NIRGraph)
            assert len(downloaded_model.nodes) == len(model.nodes)