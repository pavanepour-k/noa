"""
Input Validation Module for Japanese Tokenizer

This module provides comprehensive input validation for all public methods
to prevent crashes and ensure data integrity.
"""

import re
from typing import Any, Optional, Union


class InputValidator:
    """
    Comprehensive input validation for Japanese tokenizer.
    
    Provides static methods for validating various input types
    with appropriate error messages and type checking.
    """
    
    # Constants for validation
    MAX_TEXT_LENGTH = 100000  # Maximum text length
    MAX_WINDOW_SIZE = 50      # Maximum context window size
    MAX_CAPACITY = 100000     # Maximum cache capacity
    
    @staticmethod
    def validate_text(text: Any, max_length: int = MAX_TEXT_LENGTH) -> str:
        """
        Validate text input.
        
        Args:
            text: Input to validate
            max_length: Maximum allowed length
            
        Returns:
            Validated text string
            
        Raises:
            TypeError: If text is not a string
            ValueError: If text is empty or too long
        """
        if not isinstance(text, str):
            raise TypeError(f"Text must be a string, got {type(text).__name__}")
        
        if not text.strip():
            raise ValueError("Text cannot be empty or whitespace only")
        
        if len(text) > max_length:
            raise ValueError(f"Text length ({len(text)}) exceeds maximum ({max_length})")
        
        return text.strip()
    
    @staticmethod
    def validate_confidence(value: Any, name: str = "confidence") -> float:
        """
        Validate confidence value.
        
        Args:
            value: Value to validate
            name: Name of the parameter for error messages
            
        Returns:
            Validated confidence float
            
        Raises:
            TypeError: If value is not numeric
            ValueError: If value is out of range [0.0, 1.0]
        """
        if not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a number, got {type(value).__name__}")
        
        confidence = float(value)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"{name} must be between 0.0 and 1.0, got {confidence}")
        
        return confidence
    
    @staticmethod
    def validate_kanji(char: Any) -> str:
        """
        Validate kanji character.
        
        Args:
            char: Character to validate
            
        Returns:
            Validated kanji character
            
        Raises:
            TypeError: If char is not a string
            ValueError: If char is not a single kanji character
        """
        if not isinstance(char, str):
            raise TypeError(f"Kanji must be a string, got {type(char).__name__}")
        
        if len(char) != 1:
            raise ValueError(f"Kanji must be a single character, got length {len(char)}")
        
        # Check if character is in kanji range
        if not re.match(r'[\u4e00-\u9faf]', char):
            raise ValueError(f"Character '{char}' is not a valid kanji")
        
        return char
    
    @staticmethod
    def validate_window_size(size: Any, max_size: int = MAX_WINDOW_SIZE) -> int:
        """
        Validate context window size.
        
        Args:
            size: Size to validate
            max_size: Maximum allowed size
            
        Returns:
            Validated window size
            
        Raises:
            TypeError: If size is not an integer
            ValueError: If size is out of range
        """
        if not isinstance(size, int):
            raise TypeError(f"Window size must be an integer, got {type(size).__name__}")
        
        if size <= 0:
            raise ValueError(f"Window size must be positive, got {size}")
        
        if size > max_size:
            raise ValueError(f"Window size ({size}) exceeds maximum ({max_size})")
        
        return size
    
    @staticmethod
    def validate_capacity(capacity: Any, max_capacity: int = MAX_CAPACITY) -> int:
        """
        Validate cache capacity.
        
        Args:
            capacity: Capacity to validate
            max_capacity: Maximum allowed capacity
            
        Returns:
            Validated capacity
            
        Raises:
            TypeError: If capacity is not an integer
            ValueError: If capacity is out of range
        """
        if not isinstance(capacity, int):
            raise TypeError(f"Capacity must be an integer, got {type(capacity).__name__}")
        
        if capacity <= 0:
            raise ValueError(f"Capacity must be positive, got {capacity}")
        
        if capacity > max_capacity:
            raise ValueError(f"Capacity ({capacity}) exceeds maximum ({max_capacity})")
        
        return capacity
    
    @staticmethod
    def validate_word_list(words: Any, name: str = "words") -> list:
        """
        Validate word list.
        
        Args:
            words: List to validate
            name: Name of the parameter for error messages
            
        Returns:
            Validated word list
            
        Raises:
            TypeError: If words is not a list
            ValueError: If list contains invalid items
        """
        if not isinstance(words, list):
            raise TypeError(f"{name} must be a list, got {type(words).__name__}")
        
        for i, word in enumerate(words):
            if not isinstance(word, str):
                raise ValueError(f"{name}[{i}] must be a string, got {type(word).__name__}")
            if not word.strip():
                raise ValueError(f"{name}[{i}] cannot be empty")
        
        return words
    
    @staticmethod
    def validate_optional_text(text: Any, max_length: int = MAX_TEXT_LENGTH) -> Optional[str]:
        """
        Validate optional text input.
        
        Args:
            text: Input to validate (can be None)
            max_length: Maximum allowed length
            
        Returns:
            Validated text string or None
            
        Raises:
            TypeError: If text is not a string or None
            ValueError: If text is empty or too long
        """
        if text is None:
            return None
        
        return InputValidator.validate_text(text, max_length)
    
    @staticmethod
    def validate_positive_number(value: Any, name: str = "value") -> float:
        """
        Validate positive number.
        
        Args:
            value: Value to validate
            name: Name of the parameter for error messages
            
        Returns:
            Validated positive number
            
        Raises:
            TypeError: If value is not numeric
            ValueError: If value is not positive
        """
        if not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a number, got {type(value).__name__}")
        
        num_value = float(value)
        if num_value <= 0:
            raise ValueError(f"{name} must be positive, got {num_value}")
        
        return num_value
    
    @staticmethod
    def validate_boolean(value: Any, name: str = "value") -> bool:
        """
        Validate boolean value.
        
        Args:
            value: Value to validate
            name: Name of the parameter for error messages
            
        Returns:
            Validated boolean
            
        Raises:
            TypeError: If value is not a boolean
        """
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be a boolean, got {type(value).__name__}")
        
        return value
    
    @staticmethod
    def validate_database_path(path: Any) -> str:
        """
        Validate database file path.
        
        Args:
            path: Path to validate
            
        Returns:
            Validated path string
            
        Raises:
            TypeError: If path is not a string
            ValueError: If path is empty
        """
        if not isinstance(path, str):
            raise TypeError(f"Database path must be a string, got {type(path).__name__}")
        
        if not path.strip():
            raise ValueError("Database path cannot be empty")
        
        return path.strip()
    
    @staticmethod
    def validate_semantic_field(field: Any) -> str:
        """
        Validate semantic field name.
        
        Args:
            field: Field name to validate
            
        Returns:
            Validated field name
            
        Raises:
            TypeError: If field is not a string
            ValueError: If field is empty
        """
        if not isinstance(field, str):
            raise TypeError(f"Semantic field must be a string, got {type(field).__name__}")
        
        if not field.strip():
            raise ValueError("Semantic field cannot be empty")
        
        return field.strip().lower()
    
    @staticmethod
    def validate_engine_weights(fast_weight: Any, context_weight: Any) -> tuple[float, float]:
        """
        Validate engine weights.
        
        Args:
            fast_weight: Fast engine weight
            context_weight: Context engine weight
            
        Returns:
            Tuple of validated weights
            
        Raises:
            TypeError: If weights are not numeric
            ValueError: If weights are invalid
        """
        fast = InputValidator.validate_positive_number(fast_weight, "fast_weight")
        context = InputValidator.validate_positive_number(context_weight, "context_weight")
        
        total = fast + context
        if total <= 0:
            raise ValueError("Total weight must be positive")
        
        # Normalize weights
        return fast / total, context / total
