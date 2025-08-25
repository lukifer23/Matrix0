"""
Unified Configuration Utilities for Matrix0
Centralizes configuration access patterns and validation.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Optional, Union, TypeVar
from dataclasses import dataclass
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class ConfigPath:
    """Represents a configuration path for structured access."""
    section: Optional[str] = None
    subsection: Optional[str] = None
    key: Optional[str] = None

    def __str__(self) -> str:
        parts = []
        if self.section:
            parts.append(self.section)
        if self.subsection:
            parts.append(self.subsection)
        if self.key:
            parts.append(self.key)
        return ".".join(parts)


class ConfigManager:
    """Centralized configuration manager with unified access patterns."""

    def __init__(self):
        self._config_cache: Dict[str, Any] = {}
        self._lock = threading.RLock()
        self._current_config: Optional[Any] = None

    def set_config(self, config: Any) -> None:
        """Set the current configuration object."""
        with self._lock:
            self._current_config = config
            self._config_cache.clear()  # Clear cache when config changes

    def get_config(self) -> Optional[Any]:
        """Get the current configuration object."""
        return self._current_config

    def get(self, path: Union[str, ConfigPath], default: Any = None, config: Optional[Any] = None) -> Any:
        """Unified configuration access with caching and error handling."""
        if config is None:
            config = self._current_config

        if config is None:
            logger.warning(f"No configuration available for path: {path}")
            return default

        # Convert string path to ConfigPath
        if isinstance(path, str):
            parts = path.split('.')
            if len(parts) == 1:
                config_path = ConfigPath(key=parts[0])
            elif len(parts) == 2:
                config_path = ConfigPath(section=parts[0], key=parts[1])
            elif len(parts) == 3:
                config_path = ConfigPath(section=parts[0], subsection=parts[1], key=parts[2])
            else:
                logger.error(f"Invalid config path format: {path}")
                return default
        else:
            config_path = path

        # Check cache first
        cache_key = str(config_path)
        with self._lock:
            if cache_key in self._config_cache:
                return self._config_cache[cache_key]

        try:
            value = self._get_value(config, config_path, default)

            # Cache the result
            with self._lock:
                self._config_cache[cache_key] = value

            return value

        except Exception as e:
            logger.debug(f"Error accessing config path {config_path}: {e}")
            return default

    def _get_value(self, config: Any, path: ConfigPath, default: Any) -> Any:
        """Get value from configuration using the specified path."""
        current = config

        # Navigate to section
        if path.section:
            if hasattr(current, path.section):
                current = getattr(current, path.section)()
            elif hasattr(current, 'get'):
                current = current.get(path.section)
            else:
                return default

            if current is None:
                return default

        # Navigate to subsection
        if path.subsection:
            if hasattr(current, path.subsection):
                current = getattr(current, path.subsection)()
            elif hasattr(current, 'get'):
                current = current.get(path.subsection)
            else:
                return default

            if current is None:
                return default

        # Get the final value
        if path.key:
            if hasattr(current, 'get'):
                return current.get(path.key, default)
            elif hasattr(current, path.key):
                return getattr(current, path.key, default)
            else:
                return default

        return current

    def get_typed(self, path: Union[str, ConfigPath], expected_type: type[T], default: T,
                  config: Optional[Any] = None) -> T:
        """Get configuration value with type checking and conversion."""
        value = self.get(path, default, config)

        try:
            if expected_type == bool and isinstance(value, str):
                # Handle string boolean values
                return expected_type(value.lower() in ('true', '1', 'yes', 'on'))
            elif expected_type in (int, float) and isinstance(value, str):
                # Handle string numeric values
                return expected_type(value)
            elif not isinstance(value, expected_type):
                # Attempt conversion
                return expected_type(value)
            else:
                return value
        except (ValueError, TypeError) as e:
            logger.warning(f"Could not convert config value {value} to {expected_type.__name__}: {e}")
            return default

    def get_section(self, section: str, config: Optional[Any] = None) -> Dict[str, Any]:
        """Get an entire configuration section as a dictionary."""
        if config is None:
            config = self._current_config

        try:
            if hasattr(config, section):
                section_config = getattr(config, section)()
                if hasattr(section_config, 'get'):
                    # Convert to dictionary if it's a config object
                    return dict(section_config)
                return section_config
        except Exception as e:
            logger.debug(f"Could not get section {section}: {e}")

        return {}

    def validate_required(self, requirements: Dict[str, type], config: Optional[Any] = None) -> bool:
        """Validate that required configuration values are present and of correct types."""
        if config is None:
            config = self._current_config

        missing = []
        wrong_type = []

        for path, expected_type in requirements.items():
            value = self.get(path, config=config)
            if value is None:
                missing.append(path)
            elif not isinstance(value, expected_type):
                try:
                    # Try conversion
                    expected_type(value)
                except (ValueError, TypeError):
                    wrong_type.append((path, type(value).__name__, expected_type.__name__))

        if missing:
            logger.error(f"Missing required configuration: {missing}")
        if wrong_type:
            for path, actual, expected in wrong_type:
                logger.error(f"Wrong type for {path}: expected {expected}, got {actual}")

        return len(missing) == 0 and len(wrong_type) == 0


# Global configuration manager instance
config_manager = ConfigManager()


# Convenience functions for global access
def set_global_config(config: Any) -> None:
    """Set the global configuration."""
    config_manager.set_config(config)


def get_global_config() -> Optional[Any]:
    """Get the global configuration."""
    return config_manager.get_config()


def config_get(path: Union[str, ConfigPath], default: Any = None) -> Any:
    """Get configuration value from global config."""
    return config_manager.get(path, default)


def config_get_typed(path: Union[str, ConfigPath], expected_type: type[T], default: T) -> T:
    """Get typed configuration value from global config."""
    return config_manager.get_typed(path, expected_type, default)


def config_get_section(section: str) -> Dict[str, Any]:
    """Get configuration section from global config."""
    return config_manager.get_section(section)


def validate_config_requirements(requirements: Dict[str, type]) -> bool:
    """Validate required configuration values."""
    return config_manager.validate_required(requirements)


class ConfigUtils:
    """Unified configuration utilities for consistent config access."""

    @staticmethod
    def safe_get(config: Any, key: str, default: Any = None, section: Optional[str] = None) -> Any:
        """Safely get a configuration value with proper error handling."""
        try:
            if section:
                if hasattr(config, section):
                    section_config = getattr(config, section)()
                    if hasattr(section_config, 'get'):
                        return section_config.get(key, default)
                return default

            if hasattr(config, 'get'):
                return config.get(key, default)

            if hasattr(config, key):
                return getattr(config, key, default)

            return default

        except Exception as e:
            logger.debug(f"Failed to get config {section}.{key}: {e}")
            return default

    @staticmethod
    def get_nested_config(config: Any, *keys: str, default: Any = None) -> Any:
        """Get nested configuration value using dot notation."""
        try:
            current = config
            for key in keys:
                if hasattr(current, key):
                    current = getattr(current, key)
                elif hasattr(current, 'get'):
                    current = current.get(key)
                else:
                    return default

                if current is None:
                    return default

            return current

        except Exception as e:
            logger.debug(f"Failed to get nested config {'/'.join(keys)}: {e}")
            return default

    @staticmethod
    def validate_config_section(config: Any, section: str, required_keys: list) -> bool:
        """Validate that a configuration section has all required keys."""
        try:
            if not hasattr(config, section):
                logger.error(f"Configuration missing section: {section}")
                return False

            section_config = getattr(config, section)()

            missing_keys = []
            for key in required_keys:
                if hasattr(section_config, 'get'):
                    if key not in section_config:
                        missing_keys.append(key)
                elif not hasattr(section_config, key):
                    missing_keys.append(key)

            if missing_keys:
                logger.error(f"Configuration section '{section}' missing keys: {missing_keys}")
                return False

            return True

        except Exception as e:
            logger.error(f"Configuration validation failed for section '{section}': {e}")
            return False

    @staticmethod
    def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two configuration dictionaries with override taking precedence."""
        merged = base_config.copy()

        for key, value in override_config.items():
            if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
                merged[key] = ConfigUtils.merge_configs(merged[key], value)
            else:
                merged[key] = value

        return merged

    @staticmethod
    def log_config_summary(config: Any, sections: Optional[list] = None) -> None:
        """Log a summary of important configuration values."""
        try:
            sections = sections or ['model', 'training', 'mcts', 'selfplay']

            logger.info("Configuration Summary:")
            for section in sections:
                if hasattr(config, section):
                    section_config = getattr(config, section)()
                    if hasattr(section_config, 'get'):
                        # Log a few key values from each section
                        if section == 'model':
                            logger.info(f"  Model - planes: {section_config.get('planes', 'N/A')}, "
                                      f"channels: {section_config.get('channels', 'N/A')}")
                        elif section == 'training':
                            logger.info(f"  Training - batch_size: {section_config.get('batch_size', 'N/A')}, "
                                      f"lr: {section_config.get('lr', 'N/A')}")
                        elif section == 'mcts':
                            logger.info(f"  MCTS - simulations: {section_config.get('num_simulations', 'N/A')}, "
                                      f"cpuct: {section_config.get('cpuct', 'N/A')}")
                        elif section == 'selfplay':
                            logger.info(f"  Selfplay - workers: {section_config.get('num_workers', 'N/A')}, "
                                      f"simulations: {section_config.get('num_simulations', 'N/A')}")

        except Exception as e:
            logger.debug(f"Could not log config summary: {e}")


# Global instance for easy access
config_utils = ConfigUtils()


def safe_config_get(config: Any, key: str, default: Any = None, section: Optional[str] = None) -> Any:
    """Convenience function."""
    return config_utils.safe_get(config, key, default, section)


def get_nested_config(config: Any, *keys: str, default: Any = None) -> Any:
    """Convenience function."""
    return config_utils.get_nested_config(config, *keys, default=default)


def validate_config_section(config: Any, section: str, required_keys: list) -> bool:
    """Convenience function."""
    return config_utils.validate_config_section(config, section, required_keys)


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function."""
    return config_utils.merge_configs(base_config, override_config)


def log_config_summary(config: Any, sections: Optional[list] = None) -> None:
    """Convenience function."""
    config_utils.log_config_summary(config, sections)
