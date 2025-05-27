# Add doc AI! 

# See also https://medium.com/python-in-plain-english/mastering-logging-in-python-with-loguru-and-pydantic-settings-a-complete-guide-for-cross-module-a6205a987251


import os
import sys

from loguru import logger


def setup_logging() -> None:
    """
    Configure the logger.
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
