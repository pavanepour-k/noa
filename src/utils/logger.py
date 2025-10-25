"""
Logging Module for Japanese Tokenizer

This module provides centralized logging functionality using Python's built-in
logging module with configurable levels, formats, and handlers.
"""

import logging
import sys
from typing import Optional, Dict, Any
from pathlib import Path


class TokenizerLogger:
    """
    Centralized logging for Japanese tokenizer.
    
    Provides configured logger instances with consistent formatting
    and configurable output destinations.
    """
    
    _loggers: Dict[str, logging.Logger] = {}
    _configured: bool = False
    
    @classmethod
    def get_logger(cls, name: str = 'japanese_tokenizer') -> logging.Logger:
        """
        Get configured logger instance.
        
        Args:
            name: Logger name
            
        Returns:
            Configured logger instance
        """
        if name not in cls._loggers:
            cls._loggers[name] = logging.getLogger(name)
            
            # Configure if not already done
            if not cls._configured:
                cls._configure_logging()
        
        return cls._loggers[name]
    
    @classmethod
    def _configure_logging(cls) -> None:
        """Configure logging system."""
        # Import config manager
        try:
            from .config_manager import config_manager
            log_config = config_manager.get_section("logging")
        except ImportError:
            # Fallback configuration
            log_config = {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file_logging": False,
                "log_file": "tokenizer.log"
            }
        
        # Set logging level
        level = getattr(logging, log_config.get("level", "INFO").upper())
        logging.getLogger().setLevel(level)
        
        # Create formatter
        formatter = logging.Formatter(
            log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        
        # Add console handler to root logger
        root_logger = logging.getLogger()
        root_logger.handlers.clear()  # Remove existing handlers
        root_logger.addHandler(console_handler)
        
        # File handler (if enabled)
        if log_config.get("file_logging", False):
            log_file = log_config.get("log_file", "tokenizer.log")
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        
        cls._configured = True
    
    @classmethod
    def reconfigure_logging(cls, config: Dict[str, Any]) -> None:
        """
        Reconfigure logging with new settings.
        
        Args:
            config: Logging configuration dictionary
        """
        # Clear existing loggers
        cls._loggers.clear()
        cls._configured = False
        
        # Update config manager if available
        try:
            from .config_manager import config_manager
            config_manager.update_section("logging", config)
        except ImportError:
            pass
        
        # Reconfigure
        cls._configure_logging()
    
    @classmethod
    def set_level(cls, level: str) -> None:
        """
        Set logging level for all loggers.
        
        Args:
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        log_level = getattr(logging, level.upper())
        
        # Update all existing loggers
        for logger in cls._loggers.values():
            logger.setLevel(log_level)
        
        # Update root logger
        logging.getLogger().setLevel(log_level)
    
    @classmethod
    def add_file_handler(cls, log_file: str, level: Optional[str] = None) -> None:
        """
        Add file handler to all loggers.
        
        Args:
            log_file: Path to log file
            level: Optional logging level for file handler
        """
        # Ensure directory exists
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Create file handler
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        if level:
            file_handler.setLevel(getattr(logging, level.upper()))
        
        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        
        # Add to all loggers
        for logger in cls._loggers.values():
            logger.addHandler(file_handler)
        
        # Add to root logger
        logging.getLogger().addHandler(file_handler)
    
    @classmethod
    def remove_file_handlers(cls) -> None:
        """Remove all file handlers from loggers."""
        for logger in cls._loggers.values():
            # Remove file handlers
            handlers_to_remove = []
            for handler in logger.handlers:
                if isinstance(handler, logging.FileHandler):
                    handlers_to_remove.append(handler)
            
            for handler in handlers_to_remove:
                logger.removeHandler(handler)
                handler.close()
        
        # Remove from root logger
        root_logger = logging.getLogger()
        handlers_to_remove = []
        for handler in root_logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handlers_to_remove.append(handler)
        
        for handler in handlers_to_remove:
            root_logger.removeHandler(handler)
            handler.close()
    
    @classmethod
    def get_logger_stats(cls) -> Dict[str, Any]:
        """
        Get statistics about configured loggers.
        
        Returns:
            Dictionary with logger statistics
        """
        stats = {
            "total_loggers": len(cls._loggers),
            "configured": cls._configured,
            "loggers": {}
        }
        
        for name, logger in cls._loggers.items():
            stats["loggers"][name] = {
                "level": logging.getLevelName(logger.level),
                "handlers_count": len(logger.handlers),
                "handlers": [
                    {
                        "type": type(handler).__name__,
                        "level": logging.getLevelName(handler.level)
                    }
                    for handler in logger.handlers
                ]
            }
        
        return stats


# Convenience functions for common logging operations
def get_logger(name: str = 'japanese_tokenizer') -> logging.Logger:
    """Get logger instance."""
    return TokenizerLogger.get_logger(name)


def log_performance(logger: logging.Logger, operation: str, duration: float, 
                   details: Optional[Dict[str, Any]] = None) -> None:
    """
    Log performance metrics.
    
    Args:
        logger: Logger instance
        operation: Operation name
        duration: Duration in seconds
        details: Optional additional details
    """
    message = f"Performance: {operation} took {duration:.4f}s"
    if details:
        message += f" | Details: {details}"
    
    logger.info(message)


def log_cache_stats(logger: logging.Logger, stats: Dict[str, Any]) -> None:
    """
    Log cache statistics.
    
    Args:
        logger: Logger instance
        stats: Cache statistics dictionary
    """
    hit_rate = stats.get("performance", {}).get("overall_hit_rate", 0)
    l1_hits = stats.get("performance", {}).get("l1_hits", 0)
    l2_hits = stats.get("performance", {}).get("l2_hits", 0)
    l3_hits = stats.get("performance", {}).get("l3_hits", 0)
    misses = stats.get("performance", {}).get("misses", 0)
    
    logger.info(f"Cache Stats - Hit Rate: {hit_rate:.2%}, "
               f"L1: {l1_hits}, L2: {l2_hits}, L3: {l3_hits}, Misses: {misses}")


def log_tokenization_result(logger: logging.Logger, text: str, result_count: int, 
                          processing_time: float, unknown_words: int = 0) -> None:
    """
    Log tokenization results.
    
    Args:
        logger: Logger instance
        text: Input text (truncated for logging)
        result_count: Number of tokens produced
        processing_time: Processing time in seconds
        unknown_words: Number of unknown words
    """
    text_preview = text[:50] + "..." if len(text) > 50 else text
    logger.info(f"Tokenization: '{text_preview}' -> {result_count} tokens "
               f"({processing_time:.4f}s, {unknown_words} unknown)")


def log_error_with_context(logger: logging.Logger, error: Exception, 
                          context: Dict[str, Any]) -> None:
    """
    Log error with context information.
    
    Args:
        logger: Logger instance
        error: Exception that occurred
        context: Context information
    """
    context_str = ", ".join(f"{k}={v}" for k, v in context.items())
    logger.error(f"Error: {type(error).__name__}: {error} | Context: {context_str}")


def log_engine_selection(logger: logging.Logger, word: str, engine: str, 
                        confidence: float, reason: str) -> None:
    """
    Log engine selection decision.
    
    Args:
        logger: Logger instance
        word: Word being processed
        engine: Selected engine
        confidence: Confidence score
        reason: Reason for selection
    """
    logger.debug(f"Engine Selection: '{word}' -> {engine} "
                f"(confidence: {confidence:.3f}, reason: {reason})")


def log_learning_event(logger: logging.Logger, event_type: str, word: str, 
                       confidence: float, pattern_count: int) -> None:
    """
    Log adaptive learning events.
    
    Args:
        logger: Logger instance
        event_type: Type of learning event
        word: Word being learned
        confidence: Confidence in learning
        pattern_count: Number of patterns
    """
    logger.info(f"Learning: {event_type} for '{word}' "
                f"(confidence: {confidence:.3f}, patterns: {pattern_count})")
