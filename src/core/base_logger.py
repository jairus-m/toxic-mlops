import logging
import sys
from pathlib import Path
from typing import Dict, Any
from logging.handlers import RotatingFileHandler

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def setup_base_logger(name: str, log_config: Dict[str, Any] = None) -> logging.Logger:
    """
    Sets up a basic logger with console and optional file output.

    Args:
        name (str): Logger name
        log_config (Dict[str, Any], optional): Logging configuration

    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Always add console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Add file handler if configured
    if log_config and log_config.get("handler") == "file":
        log_path_str = log_config.get("path", "assets/logs/app.log")
        log_path = PROJECT_ROOT / log_path_str
        log_path.parent.mkdir(parents=True, exist_ok=True)

        fh = RotatingFileHandler(log_path, maxBytes=5 * 1024 * 1024, backupCount=5)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


# Create a base logger instance
base_logger = setup_base_logger("base")
