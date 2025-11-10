"""
Configuration Utilities
Provides functions for loading and managing configuration files
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
from copy import deepcopy


class Config:
    """
    Configuration manager with dot-notation access.

    Allows accessing configuration values using dot notation
    (e.g., config.model.layers instead of config['model']['layers']).
    """

    def __init__(self, config_dict: Dict[str, Any]):
        """
        Initialize configuration.

        Args:
            config_dict: Configuration dictionary
        """
        self._config = deepcopy(config_dict)
        self._convert_to_dot_notation(self._config)

    def _convert_to_dot_notation(self, d: Dict[str, Any]):
        """
        Convert nested dicts to allow dot notation access.

        Args:
            d: Dictionary to convert
        """
        for key, value in d.items():
            if isinstance(value, dict):
                self.__dict__[key] = Config(value)
            else:
                self.__dict__[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value with default.

        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key doesn't exist

        Returns:
            Configuration value or default

        Example:
            >>> config = Config({'model': {'layers': 4}})
            >>> layers = config.get('model.layers', 3)
            >>> dropout = config.get('model.dropout', 0.5)  # Uses default
        """
        keys = key.split('.')
        value = self._config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key: str, value: Any):
        """
        Set configuration value.

        Args:
            key: Configuration key (supports dot notation)
            value: Value to set

        Example:
            >>> config = Config({'model': {'layers': 4}})
            >>> config.set('model.dropout', 0.5)
        """
        keys = key.split('.')
        d = self._config

        for k in keys[:-1]:
            if k not in d:
                d[k] = {}
            d = d[k]

        d[keys[-1]] = value
        self._convert_to_dot_notation(self._config)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert config to dictionary.

        Returns:
            Configuration as dictionary
        """
        return deepcopy(self._config)

    def update(self, other: Union[Dict[str, Any], 'Config']):
        """
        Update configuration with another dict or Config.

        Args:
            other: Dictionary or Config to merge

        Example:
            >>> config = Config({'model': {'layers': 4}})
            >>> config.update({'model': {'dropout': 0.5}})
        """
        if isinstance(other, Config):
            other = other.to_dict()

        self._deep_update(self._config, other)
        self._convert_to_dot_notation(self._config)

    def _deep_update(self, d: Dict, u: Dict) -> Dict:
        """
        Recursively update nested dictionary.

        Args:
            d: Dictionary to update
            u: Dictionary with updates

        Returns:
            Updated dictionary
        """
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._deep_update(d[k], v)
            else:
                d[k] = v
        return d

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access."""
        return self._config[key]

    def __setitem__(self, key: str, value: Any):
        """Allow dictionary-style setting."""
        self._config[key] = value
        self._convert_to_dot_notation(self._config)

    def __contains__(self, key: str) -> bool:
        """Check if key exists."""
        return key in self._config

    def __repr__(self) -> str:
        """String representation."""
        return f"Config({self._config})"


def load_config(config_path: Union[str, Path]) -> Config:
    """
    Load configuration from YAML or JSON file.

    Args:
        config_path: Path to configuration file

    Returns:
        Config object

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If file extension is not .yaml, .yml, or .json

    Example:
        >>> config = load_config('configs/mlflow/default_config.yaml')
        >>> print(config.model.layers)
        >>> print(config.training.learning_rate)
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    if not config_path.is_file():
        raise ValueError(f"Config path is not a file: {config_path}")

    # Load based on file extension
    if config_path.suffix in ['.yaml', '.yml']:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
    elif config_path.suffix == '.json':
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
    else:
        raise ValueError(
            f"Unsupported config format: {config_path.suffix}. "
            "Use .yaml, .yml, or .json"
        )

    return Config(config_dict)


def save_config(config: Union[Config, Dict], save_path: Union[str, Path]):
    """
    Save configuration to YAML or JSON file.

    Args:
        config: Config object or dictionary
        save_path: Path to save configuration

    Example:
        >>> config = Config({'model': {'layers': 4}})
        >>> save_config(config, 'configs/my_config.yaml')
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(config, Config):
        config_dict = config.to_dict()
    else:
        config_dict = config

    if save_path.suffix in ['.yaml', '.yml']:
        with open(save_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    elif save_path.suffix == '.json':
        with open(save_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    else:
        raise ValueError(
            f"Unsupported config format: {save_path.suffix}. "
            "Use .yaml, .yml, or .json"
        )


def merge_configs(*configs: Union[Config, Dict]) -> Config:
    """
    Merge multiple configurations.

    Later configs override earlier ones.

    Args:
        *configs: Variable number of Config objects or dictionaries

    Returns:
        Merged Config object

    Example:
        >>> base_config = Config({'model': {'layers': 4, 'dropout': 0.5}})
        >>> override = {'model': {'layers': 6}}
        >>> merged = merge_configs(base_config, override)
        >>> print(merged.model.layers)  # 6
        >>> print(merged.model.dropout)  # 0.5
    """
    merged = Config({})

    for config in configs:
        merged.update(config)

    return merged


def validate_config(
    config: Config,
    required_keys: list,
    schema: Optional[Dict[str, type]] = None
) -> bool:
    """
    Validate configuration has required keys and types.

    Args:
        config: Configuration to validate
        required_keys: List of required keys (supports dot notation)
        schema: Optional dictionary mapping keys to expected types

    Returns:
        True if valid

    Raises:
        ValueError: If validation fails

    Example:
        >>> config = Config({'model': {'layers': 4}})
        >>> required = ['model.layers', 'model.dropout']
        >>> schema = {'model.layers': int, 'model.dropout': float}
        >>> validate_config(config, required, schema)
    """
    # Check required keys
    for key in required_keys:
        value = config.get(key)
        if value is None:
            raise ValueError(f"Missing required config key: {key}")

    # Check types
    if schema:
        for key, expected_type in schema.items():
            value = config.get(key)
            if value is not None and not isinstance(value, expected_type):
                raise ValueError(
                    f"Config key '{key}' has wrong type. "
                    f"Expected {expected_type.__name__}, "
                    f"got {type(value).__name__}"
                )

    return True


class ConfigManager:
    """
    Singleton configuration manager.

    Ensures only one global configuration exists.
    """

    _instance = None
    _config = None

    def __new__(cls):
        """Create singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(self, config_path: Union[str, Path]) -> Config:
        """
        Load configuration.

        Args:
            config_path: Path to configuration file

        Returns:
            Config object
        """
        self._config = load_config(config_path)
        return self._config

    @property
    def config(self) -> Config:
        """
        Get current configuration.

        Returns:
            Config object

        Raises:
            RuntimeError: If config not loaded
        """
        if self._config is None:
            raise RuntimeError(
                "Configuration not loaded. Call load() first."
            )
        return self._config

    def is_loaded(self) -> bool:
        """Check if configuration is loaded."""
        return self._config is not None


if __name__ == '__main__':
    # Test Config class
    config_dict = {
        'model': {
            'architecture': 'CustomCNN',
            'layers': 4,
            'dropout': 0.5
        },
        'training': {
            'epochs': 20,
            'batch_size': 32,
            'learning_rate': 0.001
        }
    }

    config = Config(config_dict)

    # Test dot notation access
    print(f"Model architecture: {config.model.architecture}")
    print(f"Learning rate: {config.training.learning_rate}")

    # Test get with default
    print(f"Dropout: {config.get('model.dropout', 0.0)}")
    print(f"Unknown: {config.get('model.unknown', 'default')}")

    # Test set
    config.set('model.new_param', 100)
    print(f"New param: {config.model.new_param}")

    # Test update
    config.update({'training': {'epochs': 30}})
    print(f"Updated epochs: {config.training.epochs}")

    # Test to_dict
    print(f"\nConfig dict: {config.to_dict()}")

    # Test validation
    required = ['model.architecture', 'training.epochs']
    schema = {'model.layers': int, 'training.learning_rate': float}
    validate_config(config, required, schema)
    print("Config validation passed!")
