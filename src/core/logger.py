"""
Consolidated logging module for the toxic comment classification project.

This module provides all logging functionality including:
- Basic console and file loggers
- JSON formatters for structured logging
- S3 and DynamoDB handlers for cloud logging
- Pre-configured logger instances
"""

import logging
import json
import uuid
import sys
import os
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler

from .load_config import config


class JsonFormatter(logging.Formatter):
    """Formats log records as JSON strings."""

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
    """A logging handler that appends logs to a file in S3."""

    def __init__(self, bucket: str, key: str, config: dict):
        super().__init__()
        self.bucket = bucket
        self.key = key

        # Use config-based path
        if "log_directories" in config:
            temp_path = config["log_directories"].get(
                "temp_prediction", "assets/logs/temp_prediction_logs.json"
            )
            self.local_temp_path = config["project_root"] / temp_path
        else:
            self.local_temp_path = (
                config["project_root"] / "assets" / "logs" / "temp_prediction_logs.json"
            )

        self.local_temp_path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, record):
        from .aws_utils import upload_to_s3, download_from_s3

        log_entry = self.format(record)

        download_from_s3(
            self.bucket,
            self.key,
            self.local_temp_path,
            logger=logging.getLogger(__name__),
        )

        with open(self.local_temp_path, "a") as f:
            f.write(log_entry + "\n")

        upload_to_s3(self.local_temp_path, self.key, logger=logging.getLogger(__name__))


class DynamoDBHandler(logging.Handler):
    """A logging handler that writes logs to DynamoDB with error handling and fallback."""

    def __init__(self, table_name: str, config: dict):
        super().__init__()
        self.table_name = table_name
        self.config = config
        self._dynamodb_client = None
        self._fallback_handler = None
        self._setup_fallback()

    def _setup_fallback(self):
        """Setup fallback file handler for when DynamoDB is unavailable."""
        if "log_directories" in self.config:
            fallback_path_str = self.config["log_directories"].get(
                "fallback_prediction", "assets/logs/prediction_logs_fallback.json"
            )
            fallback_path = self.config["project_root"] / fallback_path_str
        else:
            fallback_path = (
                self.config["project_root"]
                / "assets"
                / "logs"
                / "prediction_logs_fallback.json"
            )

        fallback_path.parent.mkdir(parents=True, exist_ok=True)
        self._fallback_handler = RotatingFileHandler(
            fallback_path, maxBytes=5 * 1024 * 1024, backupCount=5
        )
        self._fallback_handler.setFormatter(JsonFormatter())

    @property
    def dynamodb_client(self):
        """Lazy initialization of DynamoDB client with connection pooling."""
        if self._dynamodb_client is None:
            try:
                import boto3
                from botocore.config import Config

                # Use connection pooling for better performance
                config = Config(
                    region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
                    retries={"max_attempts": 3, "mode": "adaptive"},
                    max_pool_connections=50,
                )
                self._dynamodb_client = boto3.client("dynamodb", config=config)
            except ImportError:
                raise ImportError("boto3 is required for DynamoDB logging")
            except Exception as e:
                # Log error but don't fail - will use fallback
                print(f"Failed to initialize DynamoDB client: {e}")

        return self._dynamodb_client

    def _create_dynamodb_item(self, record):
        """Create DynamoDB item with optimized partition key strategy."""
        # Create composite partition key for better distribution
        log_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        timestamp = datetime.now(timezone.utc).isoformat()
        partition_key = f"{log_date}#{str(uuid.uuid4())[:8]}"
        sort_key = f"{timestamp}#{str(uuid.uuid4())[:8]}"

        # Parse the log record
        if isinstance(record.msg, dict):
            log_data = record.msg.copy()
        else:
            log_data = {"message": record.getMessage()}

        log_data.update(
            {
                "log_level": record.levelname,
                "pathname": record.pathname,
                "timestamp": timestamp,
                "log_date": log_date,
            }
        )

        # Convert to DynamoDB format
        item = {
            "partition_key": {"S": partition_key},
            "sort_key": {"S": sort_key},
            "timestamp": {"S": timestamp},
            "log_date": {"S": log_date},
            "log_level": {"S": record.levelname},
            "data": {"S": json.dumps(log_data)},
        }

        return item

    def emit(self, record):
        """Emit log record to DynamoDB with fallback to file logging."""
        try:
            if self.dynamodb_client is None:
                raise Exception("DynamoDB client not available")

            item = self._create_dynamodb_item(record)
            self.dynamodb_client.put_item(TableName=self.table_name, Item=item)

        except Exception as e:
            print(f"DynamoDB logging failed, using fallback: {e}")
            if self._fallback_handler:
                self._fallback_handler.emit(record)


def setup_logger(
    name: str, config: dict, log_config_key: str = "main_logging"
) -> logging.Logger:
    """
    Sets up a logger with console and optional file/cloud output.

    Args:
        name (str): Logger name
        config (dict): Application configuration
        log_config_key (str): Key in config for logging configuration

    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    # Basic formatter for console and simple file logging
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Always add console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Add file handler if configured
    log_config = config.get(log_config_key, {})
    if log_config.get("handler") == "file":
        log_path_str = log_config.get("path", "assets/logs/app.log")
        log_path = config["project_root"] / log_path_str
        log_path.parent.mkdir(parents=True, exist_ok=True)

        fh = RotatingFileHandler(log_path, maxBytes=5 * 1024 * 1024, backupCount=5)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def setup_prediction_logger(config: dict) -> logging.Logger:
    """
    Sets up a specialized logger for predictions with cloud storage options.

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
            handler = S3FileHandler(bucket=bucket_name, key=s3_key, config=config)
            handler.setFormatter(JsonFormatter())
            logger.addHandler(handler)
        else:
            logger.error("S3_BUCKET_NAME or S3 key not configured for production.")
            logger.addHandler(logging.NullHandler())

    elif env == "production" and handler_type == "dynamodb":
        table_name = os.getenv("DYNAMODB_TABLE_NAME") or log_config.get("table_name")
        if table_name:
            try:
                handler = DynamoDBHandler(table_name=table_name, config=config)
                handler.setFormatter(JsonFormatter())
                logger.addHandler(handler)
            except ImportError:
                logger.error(
                    "boto3 not available for DynamoDB logging. Using file fallback."
                )
                _add_fallback_file_handler(logger, config)
            except Exception as e:
                logger.error(
                    f"Failed to setup DynamoDB logging: {e}. Using file fallback."
                )
                _add_fallback_file_handler(logger, config)
        else:
            logger.error("DYNAMODB_TABLE_NAME not configured for production.")
            logger.addHandler(logging.NullHandler())

    elif handler_type == "file":
        log_path_str = log_config.get("path", "assets/logs/prediction_logs.json")
        log_path = config["project_root"] / log_path_str
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


def _add_fallback_file_handler(logger: logging.Logger, config: dict):
    """Helper function to add fallback file handler."""
    if "log_directories" in config:
        fallback_path_str = config["log_directories"].get(
            "fallback_prediction", "assets/logs/prediction_logs_fallback.json"
        )
        log_path = config["project_root"] / fallback_path_str
    else:
        log_path = (
            config["project_root"] / "assets" / "logs" / "prediction_logs_fallback.json"
        )

    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = RotatingFileHandler(log_path, maxBytes=5 * 1024 * 1024, backupCount=5)
    fh.setFormatter(JsonFormatter())
    logger.addHandler(fh)


logger = setup_logger("main", config)
prediction_logger = setup_prediction_logger(config)
