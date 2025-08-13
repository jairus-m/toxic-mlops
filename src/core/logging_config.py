import logging
import json
from logging.handlers import RotatingFileHandler
import os
from .load_config import config
from .base_logger import setup_base_logger, PROJECT_ROOT


class JsonFormatter(logging.Formatter):
    """
    Formats log records as JSON strings.
    """

    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "pathname": record.pathname,
        }
        if isinstance(record.msg, dict):
            log_record.update(record.msg)
        else:
            log_record["message"] = record.getMessage()

        return json.dumps(log_record)


class S3FileHandler(logging.Handler):
    """
    A logging handler that appends logs to a file in S3.
    """

    def __init__(self, bucket: str, key: str):
        super().__init__()
        self.bucket = bucket
        self.key = key
        self.local_temp_path = (
            PROJECT_ROOT / "assets" / "logs" / "temp_prediction_logs.json"
        )
        self.local_temp_path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, record):
        from .aws import (
            upload_to_s3,
            download_from_s3,
        )  # Import here to avoid circular dependency

        log_entry = self.format(record)

        # Download the current log file from S3, if it exists
        download_from_s3(self.bucket, self.key, self.local_temp_path)

        # Append the new log entry
        with open(self.local_temp_path, "a") as f:
            f.write(log_entry + "\n")

        # Upload the updated log file back to S3
        upload_to_s3(self.local_temp_path, self.key)


def setup_prediction_logger(config: dict) -> logging.Logger:
    """
    Sets up a logger for predictions based on the provided configuration.
    Args:
        config (dict): The application configuration.
    Returns:
        logging.Logger: The configured logger instance.
    """
    logger = logging.getLogger("prediction_logger")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    log_config = config.get("prediction_logging", {})
    handler_type = log_config.get("handler")
    env = config.get("env")

    if env == "production" and handler_type == "s3":
        bucket_name = os.getenv("S3_BUCKET_NAME")
        s3_key = log_config.get("key")
        if bucket_name and s3_key:
            handler = S3FileHandler(bucket=bucket_name, key=s3_key)
            handler.setFormatter(JsonFormatter())
            logger.addHandler(handler)
        else:
            logger.error("S3_BUCKET_NAME or S3 key not configured for production.")
            logger.addHandler(logging.NullHandler())
    elif handler_type == "file":
        log_path_str = log_config.get("path", "assets/logs/prediction_logs.json")
        log_path = PROJECT_ROOT / log_path_str
        log_path.parent.mkdir(parents=True, exist_ok=True)

        fh = RotatingFileHandler(log_path, maxBytes=5 * 1024 * 1024, backupCount=5)
        fh.setFormatter(JsonFormatter())
        logger.addHandler(fh)
    else:
        logger.error(
            f"Invalid prediction_logging handler for env '{env}': {handler_type}"
        )
        logger.addHandler(logging.NullHandler())

    return logger


# Create the loggers using base configuration
logger = setup_base_logger("main", config.get("main_logging", {}))
prediction_logger = setup_prediction_logger(config)
