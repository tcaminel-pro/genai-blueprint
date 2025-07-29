"""Logging configuration factory using Loguru.

Provides a centralized way to configure logging across the application
"""

# For more control, see  https://medium.com/python-in-plain-english/mastering-logging-in-python-with-loguru-and-pydantic-settings-a-complete-guide-for-cross-module-a6205a987251

import os
import sys

from loguru import logger

from src.utils.config_mngr import global_config


def setup_logging(level: str | None = None) -> None:
    """Configure the application logger with Loguru.

    Sets up logging with a default format. It can be overridden by setting the LOGURU_FORMAT environment variable.
    """
    LOGURU_FORMAT = "<cyan>{time:HH:mm:ss}</cyan>-<level>{level: <7}</level> | <magenta>{file.name}</magenta>:<green>{line} <italic>{function}</italic></green>- <level>{message}</level>"
    format_str = os.environ.get("LOGURU_FORMAT") or global_config().get_str("logging.format", LOGURU_FORMAT)
    level = level or global_config().get_str("logging.level", "INFO")
    backtrace = global_config().get_bool("logging.backtrace", False)
    logger.remove()
    logger.add(
        sys.stderr,
        colorize=True,
        level=level.upper(),
        format=format_str,
        backtrace=backtrace,
        diagnose=True,
    )
