"""Configuration loader and validator."""
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str = "configs/default.yaml") -> dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Dictionary containing configuration parameters
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path) as f:
        config = yaml.safe_load(f)

    # Basic validation
    validate_config(config)

    return config


def validate_config(config: dict[str, Any]) -> None:
    """
    Validate configuration parameters.

    Args:
        config: Configuration dictionary

    Raises:
        ValueError: If configuration is invalid
    """
    required_sections = ['data', 'windows', 'similarity', 'vote', 'backtest', 'live']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")

    # Validate window length
    if config['windows']['length'] < 1:
        raise ValueError("Window length must be >= 1")

    # Validate top_k
    if config['similarity']['top_k'] < 1:
        raise ValueError("top_k must be >= 1")

    # Validate threshold
    threshold = config['vote']['threshold']
    if not 0 <= threshold <= 1:
        raise ValueError("Vote threshold must be in [0, 1]")

    # Validate normalization method
    valid_norms = ['zscore', 'rank', 'vol']
    if config['windows']['normalization'] not in valid_norms:
        raise ValueError(f"Normalization must be one of {valid_norms}")

    # Validate similarity metric
    valid_metrics = ['pearson', 'spearman', 'cosine']
    if config['similarity']['metric'] not in valid_metrics:
        raise ValueError(f"Similarity metric must be one of {valid_metrics}")

    # Validate vote scheme
    valid_schemes = ['majority', 'similarity_weighted']
    if config['vote']['scheme'] not in valid_schemes:
        raise ValueError(f"Vote scheme must be one of {valid_schemes}")


def get_param(config: dict[str, Any], *keys, default=None) -> Any:
    """
    Safely get nested configuration parameter.

    Args:
        config: Configuration dictionary
        *keys: Nested keys to traverse
        default: Default value if key not found

    Returns:
        Configuration value or default
    """
    result = config
    for key in keys:
        if isinstance(result, dict) and key in result:
            result = result[key]
        else:
            return default
    return result
