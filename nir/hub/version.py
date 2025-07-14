"""Version compatibility utilities for the NIR model hub."""

from typing import Tuple


def parse_version(version_str: str) -> Tuple[int, int, int]:
    """Parse a version string into a tuple of (major, minor, patch).
    
    Args:
        version_str: Version string in format "x.y.z"
        
    Returns:
        Tuple of (major, minor, patch) version numbers
    """
    try:
        parts = version_str.split('.')
        if len(parts) >= 3:
            return (int(parts[0]), int(parts[1]), int(parts[2]))
        elif len(parts) == 2:
            return (int(parts[0]), int(parts[1]), 0)
        elif len(parts) == 1:
            return (int(parts[0]), 0, 0)
        else:
            return (0, 0, 0)
    except (ValueError, AttributeError):
        return (0, 0, 0)  # Default for unparseable versions


def check_compatibility(model_version: str, current_version: str) -> bool:
    """Check if a model version is compatible with the current NIR version.
    
    The compatibility rules are:
    - Major version must match
    - Model's minor version must be <= current minor version
    - If minor versions match, model's patch version must be <= current patch version
    
    Args:
        model_version: Version string of the model
        current_version: Current NIR version string
        
    Returns:
        True if versions are compatible, False otherwise
    """
    model_major, model_minor, model_patch = parse_version(model_version)
    current_major, current_minor, current_patch = parse_version(current_version)
    
    # Major version must match
    if model_major != current_major:
        return False
    
    # Minor version compatibility check
    if model_minor > current_minor:
        return False
    
    # Patch version check only matters if minor versions match
    if model_minor == current_minor and model_patch > current_patch:
        return False
    
    return True
