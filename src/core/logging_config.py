import logging
import json
import uuid
from datetime import datetime, timezone
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


class DynamoDBHandler(logging.Handler):
    """
    A logging handler that writes logs to DynamoDB with error handling and fallback.
    """

    def __init__(self, table_name: str):
        super().__init__()
        self.table_name = table_name
        self._dynamodb_client = None
        self._fallback_handler = None
        self._setup_fallback()

    def _setup_fallback(self):
        """Setup fallback file handler for when DynamoDB is unavailable."""
        fallback_path = (
            PROJECT_ROOT / "assets" / "logs" / "prediction_logs_fallback.json"
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

        # Add metadata
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

            # Write to DynamoDB
            self.dynamodb_client.put_item(TableName=self.table_name, Item=item)

        except Exception as e:
            # Fallback to file logging
            print(f"DynamoDB logging failed, using fallback: {e}")
            if self._fallback_handler:
                self._fallback_handler.emit(record)


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
    elif env == "production" and handler_type == "dynamodb":
        table_name = os.getenv("DYNAMODB_TABLE_NAME") or log_config.get("table_name")
        if table_name:
            try:
                handler = DynamoDBHandler(table_name=table_name)
                handler.setFormatter(JsonFormatter())
                logger.addHandler(handler)
            except ImportError:
                logger.error(
                    "boto3 not available for DynamoDB logging. Using file fallback."
                )
                # Fallback to file logging
                log_path = (
                    PROJECT_ROOT / "assets" / "logs" / "prediction_logs_fallback.json"
                )
                log_path.parent.mkdir(parents=True, exist_ok=True)
                fh = RotatingFileHandler(
                    log_path, maxBytes=5 * 1024 * 1024, backupCount=5
                )
                fh.setFormatter(JsonFormatter())
                logger.addHandler(fh)
            except Exception as e:
                logger.error(
                    f"Failed to setup DynamoDB logging: {e}. Using file fallback."
                )
                # Fallback to file logging
                log_path = (
                    PROJECT_ROOT / "assets" / "logs" / "prediction_logs_fallback.json"
                )
                log_path.parent.mkdir(parents=True, exist_ok=True)
                fh = RotatingFileHandler(
                    log_path, maxBytes=5 * 1024 * 1024, backupCount=5
                )
                fh.setFormatter(JsonFormatter())
                logger.addHandler(fh)
        else:
            logger.error("DYNAMODB_TABLE_NAME not configured for production.")
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
