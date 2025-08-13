import pandas as pd
import json
import os
from pathlib import Path
from src.core import get_asset_path, config, logger
from src.core.aws import download_from_s3


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
        s3_key = log_config.get("key")
        bucket_name = os.getenv("S3_BUCKET_NAME")

        if not s3_key or not bucket_name:
            logger.error("S3 bucket name or key not configured for production.")
            return logs

        # Define a local path to download the S3 file
        local_log_path = (
            Path(config["project_root"]) / "assets" / "logs" / "prediction_logs_s3.json"
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
            logger.warning("Could not download logs from S3. File might not exist yet.")

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
