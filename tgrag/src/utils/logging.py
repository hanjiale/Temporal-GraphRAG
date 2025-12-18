"""Centralized logging utilities."""

import logging

# Create logger instances for different modules
def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a given module name.
    
    Args:
        name: Logger name, typically __name__ from the calling module
        
    Returns:
        Logger instance configured for the module
    """
    return logging.getLogger(f"temporal-graphrag.{name}")

# Main logger for the package
logger = logging.getLogger("temporal-graphrag")

__all__ = ["logger", "get_logger"]
