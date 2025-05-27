"""Logging configuration factory using Loguru.

Provides a centralized way to configure logging across the application with:
- Colorized output
- Custom format including timestamp, level, file, line and function
- Environment variable override for format
"""

import os
import sys

from loguru import logger


def setup_logging() -> None:
    """Configure the application logger with Loguru.

    Sets up logging with a default format that includes:
    - Timestamp
    - Log level
    - Source file and line number
    - Function name
    - Message

    The format can be overridden by setting the LOGURU_FORMAT environment variable.
    """
    LOGURU_FORMAT = "<cyan>{time:HH:mm:ss}</cyan>-<level>{level: <7}</level> | <magenta>{file.name}</magenta>:<green>{line} <italic>{function}</italic></green>- <level>{message}</level>"
    # Workaround "LOGURU_FORMAT" does not seems to be taken into account
    format_str = os.environ.get("LOGURU_FORMAT") or LOGURU_FORMAT
    logger.remove()
    logger.add(
        sys.stderr,
        colorize=True,
        format=format_str,
        backtrace=False,
        diagnose=True,
    )
