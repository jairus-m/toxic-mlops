import pandas as pd
import json
import os
from pathlib import Path
from src.core import get_asset_path, config, logger
from src.core.aws import download_from_s3


def _load_logs_from_dynamodb() -> list:
    """
    Loads all logs from DynamoDB with pagination and error handling.
    Returns:
        list: A list of all logs from DynamoDB.
    """
    logs = []
    table_name = os.getenv("DYNAMODB_TABLE_NAME")

    if not table_name:
        logger.error("DYNAMODB_TABLE_NAME environment variable not set.")
        return logs

    try:
        import boto3
        from botocore.config import Config
        from botocore.exceptions import ClientError

        # Configure DynamoDB client with retries
        config_boto = Config(
            region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
            retries={"max_attempts": 3, "mode": "adaptive"},
        )
        dynamodb = boto3.client("dynamodb", config=config_boto)

        # Scan the table with pagination
        scan_kwargs = {"TableName": table_name, "Select": "ALL_ATTRIBUTES"}

        items_processed = 0

        while True:
            try:
                response = dynamodb.scan(**scan_kwargs)

                # Process items
                for item in response.get("Items", []):
                    try:
                        # Extract the JSON data from the 'data' field
                        if "data" in item and "S" in item["data"]:
                            log_data = json.loads(item["data"]["S"])
                            logs.append(log_data)
                            items_processed += 1
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning(f"Failed to parse log item: {e}")
                        continue

                # Check for more data
                if "LastEvaluatedKey" not in response:
                    break

                scan_kwargs["ExclusiveStartKey"] = response["LastEvaluatedKey"]

                # Add safety limit to prevent runaway scans
                if items_processed > 10000:
                    logger.warning(
                        "Reached safety limit of 10,000 items. Stopping scan."
                    )
                    break

            except ClientError as e:
                error_code = e.response["Error"]["Code"]
                if error_code == "ProvisionedThroughputExceededException":
                    logger.warning(
                        "DynamoDB throughput exceeded. Retrying after delay..."
                    )
                    import time

                    time.sleep(2)
                    continue
                else:
                    logger.error(f"DynamoDB scan failed with error: {e}")
                    break

        logger.info(f"Loaded {len(logs)} logs from DynamoDB table '{table_name}'.")

    except ImportError:
        logger.error("boto3 is required for DynamoDB access but not available.")
    except Exception as e:
        logger.error(f"Error loading logs from DynamoDB: {e}")
        # Try to load from fallback file
        fallback_path = (
            Path(config["project_root"])
            / "assets"
            / "logs"
            / "prediction_logs_fallback.json"
        )
        if fallback_path.exists():
            try:
                with open(fallback_path, "r") as f:
                    logs = [json.loads(line) for line in f]
                logger.info(f"Loaded {len(logs)} logs from fallback file.")
            except Exception as fallback_error:
                logger.error(f"Failed to load from fallback file: {fallback_error}")

    return logs


def load_dataset() -> pd.DataFrame:
    """
    Loads the dataset.
    Uses get_asset_path to be environment-aware (local vs. S3).
    Returns:
        pd.DataFrame: The loaded dataset.
    """
    try:
        logger.info("Attempting to load dataset...")
        data_path = get_asset_path("data")
        df = pd.read_csv(data_path)
        logger.info("Dataset loaded successfully.")
        return df
    except Exception as e:
        logger.error(f"Failed to load dataset. Error: {e}")
        return pd.DataFrame()


def load_all_logs() -> list:
    """
    Loads all logs from the prediction log file.
    Returns:
        list: A list of all logs.
    """
    logs = []
    env = config.get("env")

    if env == "development":
        log_config = config.get("prediction_logging", {})
        log_path_str = log_config.get("path")
        if not log_path_str:
            logger.warning("Prediction log path not configured.")
            return logs

        log_file = os.path.join(config["project_root"], log_path_str)

        if os.path.exists(log_file):
            try:
                with open(log_file, "r") as f:
                    logs = [json.loads(line) for line in f]
                logger.info(f"Loaded {len(logs)} logs from {log_file}.")
            except Exception as e:
                logger.error(f"Error reading or parsing log file {log_file}: {e}")
        else:
            logger.warning(f"Log file not found at {log_file}.")

    elif env == "production":
        log_config = config.get("prediction_logging", {})
        handler_type = log_config.get(
            "handler", "s3"
        )  # Default to S3 for backward compatibility

        if handler_type == "dynamodb":
            logs = _load_logs_from_dynamodb()
        else:
            # S3 fallback
            s3_key = log_config.get("key")
            bucket_name = os.getenv("S3_BUCKET_NAME")

            if not s3_key or not bucket_name:
                logger.error("S3 bucket name or key not configured for production.")
                return logs

            # Define a local path to download the S3 file
            local_log_path = (
                Path(config["project_root"])
                / "assets"
                / "logs"
                / "prediction_logs_s3.json"
            )
            local_log_path.parent.mkdir(parents=True, exist_ok=True)

            if download_from_s3(
                bucket_name, s3_key, local_log_path, needs_full_download=True
            ):
                if local_log_path.exists():
                    try:
                        with open(local_log_path, "r") as f:
                            logs = [json.loads(line) for line in f]
                        logger.info(f"Loaded {len(logs)} logs from S3.")
                    except Exception as e:
                        logger.error(f"Error reading or parsing log file from S3: {e}")
            else:
                logger.warning(
                    "Could not download logs from S3. File might not exist yet."
                )

    return logs


def load_feedback_logs() -> list:
    """
    Filters all logs to return only those with feedback.
    Returns:
        list: A list of feedback logs.
    """
    all_logs = load_all_logs()
    feedback_logs = [
        log
        for log in all_logs
        if log.get("endpoint") == "/feedback" and "is_prediction_correct" in log
    ]
    logger.info(f"Found {len(feedback_logs)} feedback logs.")
    return feedback_logs
