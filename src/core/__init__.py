from .load_config import config, PROJECT_ROOT
from .aws import download_from_s3, upload_to_s3
from .logging_config import logger, prediction_logger
from .asset_resolution import get_asset_path

logger.info(f"Configuration loaded for '{config['env']}' environment.")

__all__ = [
    "config",
    "logger",
    "prediction_logger",
    "get_asset_path",
    "upload_to_s3",
    "download_from_s3",
    "PROJECT_ROOT",
]
