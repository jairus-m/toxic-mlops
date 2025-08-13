"""
Module for downloading the training dataset and uploading it to S3.
"""

import shutil
from pathlib import Path
import kagglehub
from src.core import logger, config, PROJECT_ROOT, upload_to_s3


def download_kaggle_dataset() -> Path:
    """
    Downloads the dataset from Kaggle using kagglehub.

    In a production environment, it also uploads the downloaded dataset to S3.

    Returns:
        Path: The local path to the downloaded dataset file.
    """
    try:
        dataset_path = config["kaggle"]["dataset_path"]
        dataset_name = config["kaggle"]["dataset_name"]
        env = config["env"]

        # Determine the local path for the data
        if env == "production":
            # In prod, always use a temporary directory for local storage
            local_dir = PROJECT_ROOT / "assets" / "data"
        else:
            # In dev, use the path from config
            local_path_str = config["paths"]["data"]
            local_dir = (PROJECT_ROOT / local_path_str).parent

        local_dir.mkdir(parents=True, exist_ok=True)
        destination_file = local_dir / dataset_name

        # Download from Kaggle
        logger.info(f"Downloading dataset from Kaggle: {dataset_path}")
        path = kagglehub.dataset_download(dataset_path)
        downloaded_path = Path(path[0] if isinstance(path, list) else path)

        # Find and copy the CSV file
        csv_file = downloaded_path / dataset_name
        if csv_file.exists():
            shutil.copy(csv_file, destination_file)
            logger.info(f"Dataset '{dataset_name}' saved locally to {destination_file}")
        else:
            raise FileNotFoundError(
                f"Specified dataset file '{dataset_name}' not found in download."
            )

        # If in production, upload the file to S3
        if env == "production":
            s3_key = config["paths"]["data"]
            upload_to_s3(destination_file, s3_key)

        return destination_file

    except Exception as e:
        logger.error(f"Error in dataset acquisition: {str(e)}")
        raise
