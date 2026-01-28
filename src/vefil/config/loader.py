"""Configuration loader from YAML."""

from pathlib import Path
from typing import Any, Dict

import yaml

from .schema import Config


def load_config(yaml_path: str = None) -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        yaml_path: Path to YAML file (defaults to defaults.yaml)
        
    Returns:
        Config object
    """
    if yaml_path is None:
        yaml_path = Path(__file__).parent / "defaults.yaml"

    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    return Config.from_dict(data)


def config_from_dict(data: Dict[str, Any]) -> Config:
    """
    Create config from dictionary.
    
    Args:
        data: Configuration dictionary
        
    Returns:
        Config object
    """
    return Config.from_dict(data)
