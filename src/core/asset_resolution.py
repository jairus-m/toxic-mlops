from pathlib import Path
import os
import sys
from .load_config import PROJECT_ROOT, config
from .aws import download_from_s3
from .logging_config import logger


def get_asset_path(asset_key: str) -> Path:
    """
    Returns the local filesystem path for a given asset key (e.g., 'model', 'data').

    In 'production' mode, it downloads the asset from S3 to a temporary local
    directory (assets) and returns the path to the local copy.
    In 'development' mode, it returns the direct local path from the config.

    Args:
        asset_key (str): The key for the asset, as defined in config.yaml.

    Returns:
        Path: The local, ready-to-use path for the asset.
    """
    path_info = config["paths"][asset_key]

    if config["env"] == "production":
        bucket = os.getenv("S3_BUCKET_NAME")
        s3_key = path_info
        if not bucket or not s3_key:
            logger.critical("S3 bucket name or key is not configured in environment.")
            sys.exit(1)

        local_path = PROJECT_ROOT / "assets" / Path(s3_key).name
        if not download_from_s3(bucket, s3_key, local_path):
            logger.critical(f"Failed to retrieve required asset {s3_key} from S3.")
            sys.exit(1)
        return local_path
    else:
        dev_path = PROJECT_ROOT / path_info
        if not dev_path.exists():
            logger.critical(f"Asset '{asset_key}' not found at local path: {dev_path}")
            sys.exit(1)
        return dev_path
