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
        return config
    except FileNotFoundError:
        logging.critical(f"Config file not found at {config_path}")
        sys.exit(1)
    except Exception as e:
        logging.critical(f"Error loading configuration: {e}")
        sys.exit(1)


config = load_config()
