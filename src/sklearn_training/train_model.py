"""
Module for training the toxic comment classification model and saving it.

This script supports two operational modes based on the `TRAIN_MODEL` env var:
1. `TRAIN_MODEL=true`: Runs the full training pipeline.
2. `TRAIN_MODEL=false`: Deploys a pre-existing local model.
"""

import os
import json
from pathlib import Path
import joblib
import psutil

from src.core import (
    logger,
    config,
    upload_to_s3,
    download_from_s3,
    PROJECT_ROOT,
)
from src.sklearn_training.utils.data_loader import download_kaggle_dataset
from src.sklearn_training.utils.experiment_tracking import (
    ExperimentTracker,
    get_model_configurations,
    train_single_model,
)
from src.sklearn_training.utils.preprocess import load_and_preprocess_data
from src.sklearn_training.utils.validation import validate_inputs
from src.sklearn_training.utils.feature_engineering import prepare_features
from src.sklearn_training.utils.model_evaluation import analyze_feature_importance
from src.sklearn_training.utils.memory import log_memory_usage


def create_and_train_model_pipeline(X, y, metadata: dict) -> tuple[dict, dict]:
    """
    Create and train multiple toxic comment classification models with MLflow tracking.

    Args:
        X (np.ndarray): Features (cleaned comments).
        y (np.ndarray): Labels (toxicity indicators).
        metadata (dict): Dataset metadata.

    Returns:
        tuple: Dictionary of trained models and their metrics
    """
    logger.info("Creating and training multiple model pipelines...")

    validate_inputs(X, y)

    # Log initial memory usage for ec2 monitoring
    process = psutil.Process()
    initial_memory = log_memory_usage(process, "Initial memory usage")

    X_train_transformed, X_test, y_train, y_test, vectorizer, X_train = (
        prepare_features(X, y)
    )

    model_configs = get_model_configurations()
    trained_models = {}
    model_metrics = {}

    # Train each model with MLflow tracking
    for model_name, model_config in model_configs.items():
        pipeline, metrics = train_single_model(
            model_name,
            model_config,
            X_train_transformed,
            y_train,
            X_test,
            y_test,
            vectorizer,
            X_train,
        )

        if pipeline is not None and metrics is not None:
            trained_models[model_name] = pipeline
            model_metrics[model_name] = metrics

    log_memory_usage(process, "Final memory usage", initial_memory)

    return trained_models, model_metrics


def run_training_pipeline():
    """
    Main entry point for the hybrid MLflow toxic comment classification training process.
    """
    logger.info(
        "Starting Hybrid MLflow Multi-Model Toxic Comment Classification Training..."
    )

    env = config.get("env", "development")
    db_s3_key = config.get("mlflow", {}).get("db_s3_key")
    local_db_path = Path("mlflow.db")
    s3_bucket = config.get("s3", {}).get("bucket_name")

    if env == "production" and db_s3_key and s3_bucket:
        logger.info(
            f"Attempting to download existing MLflow DB from s3://{s3_bucket}/{db_s3_key}"
        )
        download_from_s3(s3_bucket, db_s3_key, local_db_path)

    try:
        tracker = ExperimentTracker()
        experiment_id = tracker.setup_tracking()
        logger.info(f"MLflow experiment ID: {experiment_id}")

        local_data_path = download_kaggle_dataset()
        X, y, metadata = load_and_preprocess_data(local_data_path)

        trained_models, model_metrics = create_and_train_model_pipeline(X, y, metadata)

        if not trained_models:
            raise ValueError("No models were trained successfully")

        best_model_name, best_metrics = tracker.identify_best_model(model_metrics)

        feature_importance = analyze_feature_importance(trained_models[best_model_name])
        best_metrics["feature_importance"] = feature_importance

        tracker.promote_model(best_model_name, trained_models, best_metrics, metadata)

        logger.info("Hybrid MLflow training process completed successfully!")
        logger.info(f"Best model: {best_model_name}")
        logger.info(
            f"Best model performance - Mean ROC-AUC: {best_metrics['test_mean_roc_auc']:.4f}"
        )
        logger.info(f"Total models trained: {len(trained_models)}")

        tracker.log_training_summary(trained_models, best_metrics)

    except Exception as e:
        logger.error(f"An unexpected error occurred during training: {e}")
        raise
    finally:
        if env == "production" and db_s3_key and local_db_path.exists():
            logger.info(f"Uploading MLflow DB to s3://{s3_bucket}/{db_s3_key}")
            upload_to_s3(local_db_path, db_s3_key)
            logger.info("MLflow DB upload complete.")


def deploy_local_model():
    """
    Deploys a pre-trained model from a local path.
    """
    logger.info("Starting local model deployment process...")

    model_path = PROJECT_ROOT / "assets/models/toxic_model.pkl"
    metadata_path = PROJECT_ROOT / "assets/models/toxic_model_metadata.json"

    if not model_path.exists() or not metadata_path.exists():
        logger.error(
            f"Local model or metadata not found. Searched paths:\n- {model_path}\n- {metadata_path}"
        )
        raise FileNotFoundError("Required model and metadata files not found.")

    logger.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)

    logger.info(f"Loading metadata from {metadata_path}")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # Download fresh dataset and upload to S3 if in production
    logger.info("Downloading fresh training dataset...")
    try:
        local_data_path = download_kaggle_dataset()
        env = config.get("env", "development")
        if env == "production":
            logger.info(
                f"Dataset staged successfully to {local_data_path} and uploaded to S3"
            )
        else:
            logger.info(f"Dataset staged successfully to {local_data_path}")
    except Exception as e:
        logger.warning(f"Failed to download dataset: {e}")
        logger.info("Continuing with local model deployment without fresh data...")

    # ExperimentTracker class expects a specific structure. We need to create it.
    best_model_name = "local_model"
    trained_models = {best_model_name: model}

    # Metadata file should contain dataset_metadata and training_metrics
    dataset_metadata = metadata.get("dataset_metadata", {})
    best_metrics = metadata.get("training_metrics", {})

    tracker = ExperimentTracker()
    tracker.setup_tracking()

    logger.info("Promoting local model...")
    tracker.promote_model(
        best_model_name=best_model_name,
        trained_models=trained_models,
        best_metrics=best_metrics,
        metadata=dataset_metadata,
    )
    logger.info("Local model deployment completed successfully!")


def main():
    """
    Main entry point for the script.
    Reads the TRAIN_MODEL environment variable to determine the execution path.
    """
    # Default to 'true' if the variable is not set
    train_model_flag = os.getenv("TRAIN_MODEL", "true").lower()
    logger.info(f"TRAIN_MODEL: {train_model_flag}")

    if train_model_flag == "true":
        logger.info("Running training pipeline...")
        run_training_pipeline()
    elif train_model_flag == "false":
        logger.info("Deploying local model...")
        deploy_local_model()
    else:
        logger.warning(
            f"Invalid value for TRAIN_MODEL: '{train_model_flag}'. Defaulting to training."
        )
        run_training_pipeline()


if __name__ == "__main__":
    main()
