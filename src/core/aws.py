from botocore.exceptions import ClientError
import os
from pathlib import Path
import boto3
from .base_logger import setup_base_logger

# Create AWS-specific logger
logger = setup_base_logger("aws")


def upload_to_s3(local_path: Path, s3_key: str) -> bool:
    """
    Uploads a local file to an S3 bucket.

    Args:
        local_path (Path): The path to the local file to upload.
        s3_key (str): The destination key (path) in the S3 bucket.

    Returns:
        bool: True if upload was successful, False otherwise.
    """
    bucket = os.getenv("S3_BUCKET_NAME")
    if not bucket:
        logger.critical("S3_BUCKET_NAME is not configured. Cannot upload.")
        return False

    try:
        s3 = boto3.client("s3")
        logger.info(f"Uploading {local_path.name} to s3://{bucket}/{s3_key}...")
        s3.upload_file(str(local_path), bucket, s3_key)
        logger.info("Upload to S3 successful!")
        return True
    except ClientError as e:
        logger.error(f"Failed to upload to S3: {e}")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred during S3 upload: {e}")
        return False


def download_from_s3(
    bucket: str, key: str, local_path: Path, needs_full_download: bool = False
) -> bool:
    """
    Downloads a file from an S3 bucket to a local path.

    Args:
        bucket (str): The S3 bucket name.
        key (str): The key (path) of the object in the bucket.
        local_path (Path): The local destination path.
        needs_full_download (bool): If True, the file will be downloaded even if it already exists locally.

    Returns:
        bool: True if download was successful or file already exists, False otherwise.
    """
    if local_path.exists() and needs_full_download is False:
        logger.info(f"File {local_path} already exists locally. Skipping S3 download.")
        return True
    try:
        s3 = boto3.client("s3")
        logger.info(f"Downloading s3://{bucket}/{key} to {local_path}...")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, key, str(local_path))
        logger.info("Download complete.")
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            logger.error(f"S3 object not found: s3://{bucket}/{key}")
        else:
            logger.error(f"Error downloading from S3: {e}")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred during S3 download: {e}")
        return False
