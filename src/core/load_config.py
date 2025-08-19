import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any
import yaml
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

# Define project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config() -> Dict[str, Any]:
    """
    Loads configuration from config.yaml based on the APP_ENV environment variable.

    Merges default settings with environment-specific settings.
    Exits the application if the configuration cannot be loaded.

    Returns:
        Dict[str, Any]: The fully loaded configuration dictionary.
    """
    env = os.getenv("APP_ENV", "development")
    config_path = PROJECT_ROOT / "config.yaml"

    try:
        with open(config_path, "r") as f:
            full_config = yaml.safe_load(f)

        # Start with default config
        config = full_config.get("default", {})
        # Merge in environment-specific config
        env_config = full_config.get(env, {})

        # Deep merge for nested dictionaries like 'paths'
        for key, value in env_config.items():
            if isinstance(value, dict) and isinstance(config.get(key), dict):
                config[key].update(value)
            else:
                config[key] = value

        config["env"] = env
        config["project_root"] = PROJECT_ROOT

        # Replace placeholders with environment variables in production
        if env == "production":
            mlflow_server_ip = os.getenv("MLFLOW_SERVER_IP")
            s3_bucket_name = os.getenv("S3_BUCKET_NAME")

            if mlflow_server_ip and "mlflow" in config:
                config["mlflow"]["tracking_uri"] = config["mlflow"][
                    "tracking_uri"
                ].replace("MLFLOW_SERVER_IP", mlflow_server_ip)

            if s3_bucket_name and "mlflow" in config:
                config["mlflow"]["artifact_root"] = config["mlflow"][
                    "artifact_root"
                ].replace("BUCKET_NAME", s3_bucket_name)

        return config
    except FileNotFoundError:
        logging.critical(f"Config file not found at {config_path}")
        sys.exit(1)
    except Exception as e:
        logging.critical(f"Error loading configuration: {e}")
        sys.exit(1)


config = load_config()
