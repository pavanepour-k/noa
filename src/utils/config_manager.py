"""
Configuration Manager for Japanese Tokenizer

This module provides centralized configuration management using JSON files
with validation, defaults, and singleton pattern for global access.
"""

import json
import os
from typing import Any, Dict, Optional, Union
from pathlib import Path


class ConfigManager:
    """
    Singleton configuration manager for the Japanese tokenizer.
    
    Loads configuration from JSON files, provides validation,
    and offers easy access to nested configuration values.
    """
    
    _instance: Optional['ConfigManager'] = None
    _config: Dict[str, Any] = {}
    _config_path: Optional[str] = None
    
    def __new__(cls) -> 'ConfigManager':
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize configuration manager."""
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._load_default_config()
    
    def _load_default_config(self) -> None:
        """Load default configuration values."""
        self._config = {
            "cache_system": {
                "l1_capacity": 10000,
                "l2_capacity": 2000,
                "prefetch_enabled": True,
                "max_access_patterns": 50000,
                "max_prefetch_queue": 1000
            },
            "hybrid_tokenizer": {
                "confidence_threshold": 0.7,
                "fast_engine_weight": 0.6,
                "context_engine_weight": 0.4
            },
            "context_engine": {
                "context_window_size": 5,
                "semantic_field_threshold": 0.3
            },
            "adaptive_learning": {
                "min_pattern_frequency": 3,
                "confidence_threshold": 0.7,
                "max_memory_size": 10000,
                "pattern_decay_factor": 0.95
            },
            "performance": {
                "max_text_length": 100000,
                "sentence_processing_target_ms": 50,
                "cache_hit_rate_target": 0.85
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file_logging": False,
                "log_file": "tokenizer.log"
            }
        }
    
    def load_config(self, config_path: str) -> None:
        """
        Load configuration from JSON file.
        
        Args:
            config_path: Path to configuration file
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            json.JSONDecodeError: If config file is invalid JSON
            ValueError: If config structure is invalid
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)
            
            # Merge with defaults
            self._merge_config(loaded_config)
            self._config_path = config_path
            
            # Validate configuration
            self.validate_config()
            
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Invalid JSON in config file: {e}")
    
    def _merge_config(self, loaded_config: Dict[str, Any]) -> None:
        """Merge loaded config with defaults."""
        def merge_dict(default: Dict, loaded: Dict) -> Dict:
            result = default.copy()
            for key, value in loaded.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = merge_dict(result[key], value)
                else:
                    result[key] = value
            return result
        
        self._config = merge_dict(self._config, loaded_config)
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to config value (e.g., "cache_system.l1_capacity")
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self._config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value: Any) -> None:
        """
        Set configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to config value
            value: Value to set
        """
        keys = key_path.split('.')
        config = self._config
        
        # Navigate to parent of target key
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        # Set the value
        config[keys[-1]] = value
    
    def validate_config(self) -> None:
        """
        Validate configuration structure and values.
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Validate cache system
        cache_config = self.get("cache_system", {})
        if not isinstance(cache_config, dict):
            raise ValueError("cache_system must be a dictionary")
        
        required_cache_keys = ["l1_capacity", "l2_capacity", "prefetch_enabled"]
        for key in required_cache_keys:
            if key not in cache_config:
                raise ValueError(f"Missing required cache_system key: {key}")
        
        # Validate numeric values
        if not isinstance(cache_config.get("l1_capacity"), int) or cache_config["l1_capacity"] <= 0:
            raise ValueError("l1_capacity must be a positive integer")
        
        if not isinstance(cache_config.get("l2_capacity"), int) or cache_config["l2_capacity"] <= 0:
            raise ValueError("l2_capacity must be a positive integer")
        
        # Validate hybrid tokenizer
        hybrid_config = self.get("hybrid_tokenizer", {})
        if not isinstance(hybrid_config, dict):
            raise ValueError("hybrid_tokenizer must be a dictionary")
        
        confidence = hybrid_config.get("confidence_threshold", 0.7)
        if not isinstance(confidence, (int, float)) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")
        
        # Validate engine weights
        fast_weight = hybrid_config.get("fast_engine_weight", 0.6)
        context_weight = hybrid_config.get("context_engine_weight", 0.4)
        
        if not isinstance(fast_weight, (int, float)) or fast_weight <= 0:
            raise ValueError("fast_engine_weight must be a positive number")
        
        if not isinstance(context_weight, (int, float)) or context_weight <= 0:
            raise ValueError("context_engine_weight must be a positive number")
        
        # Validate performance settings
        perf_config = self.get("performance", {})
        max_text_length = perf_config.get("max_text_length", 100000)
        if not isinstance(max_text_length, int) or max_text_length <= 0:
            raise ValueError("max_text_length must be a positive integer")
        
        # Validate logging
        log_config = self.get("logging", {})
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        log_level = log_config.get("level", "INFO")
        if log_level not in valid_levels:
            raise ValueError(f"logging.level must be one of {valid_levels}")
    
    def save_config(self, output_path: Optional[str] = None) -> None:
        """
        Save current configuration to file.
        
        Args:
            output_path: Path to save config (uses loaded path if None)
        """
        save_path = output_path or self._config_path
        if not save_path:
            raise ValueError("No output path specified and no config file loaded")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(self._config, f, indent=2, ensure_ascii=False)
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire configuration section.
        
        Args:
            section: Section name (e.g., "cache_system")
            
        Returns:
            Configuration section dictionary
        """
        return self.get(section, {})
    
    def update_section(self, section: str, updates: Dict[str, Any]) -> None:
        """
        Update entire configuration section.
        
        Args:
            section: Section name
            updates: Dictionary of updates
        """
        if not isinstance(updates, dict):
            raise ValueError("Updates must be a dictionary")
        
        self.set(section, updates)
    
    def reset_to_defaults(self) -> None:
        """Reset configuration to default values."""
        self._load_default_config()
    
    def get_all_config(self) -> Dict[str, Any]:
        """Get complete configuration dictionary."""
        return self._config.copy()
    
    def is_loaded(self) -> bool:
        """Check if configuration has been loaded from file."""
        return self._config_path is not None
    
    def get_config_path(self) -> Optional[str]:
        """Get path to loaded configuration file."""
        return self._config_path


# Global instance for easy access
config_manager = ConfigManager()
