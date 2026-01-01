"""Logging utility using loguru."""
import sys
from pathlib import Path
from loguru import logger


def get_logger(name: str = "trading_system", log_file: str = None, level: str = "ERROR"):
    logger.remove()

    logger.add(
        sys.stderr,
        format="<red>{time:HH:mm:ss}</red> | <level>{level}</level> | {message}",
        level=level,
        colorize=True
    )

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {extra[name]} | {message}",
            level="INFO",
            rotation="10 MB",
            retention="7 days",
            compression="zip"
        )

    return logger.bind(name=name)


default_logger = get_logger()
