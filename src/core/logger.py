"""Logging utility using loguru."""
import sys
from pathlib import Path

from loguru import logger


def get_logger(name: str = "sp500_analogs", log_file: str = None, level: str = "INFO"):
    """
    Get a configured logger instance using loguru.

    Args:
        name: Logger name (used as context)
        log_file: Optional path to log file
        level: Log level (DEBUG, INFO, WARNING, ERROR)

    Returns:
        Configured loguru logger

    Usage:
        logger = get_logger(__name__)
        logger.info("Processing data...")
        logger.debug("Debug details: {}", some_value)
        logger.error("Error occurred: {}", exc)
    """
    # Remove default handler
    logger.remove()

    # Add console handler with formatting
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{extra[name]}</cyan> | <level>{message}</level>",
        level=level,
        colorize=True
    )

    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {extra[name]} | {message}",
            level=level,
            rotation="10 MB",  # Rotate when file reaches 10 MB
            retention="7 days",  # Keep logs for 7 days
            compression="zip"  # Compress rotated logs
        )

    # Bind the name as context
    return logger.bind(name=name)


# Create default logger instance
default_logger = get_logger()
