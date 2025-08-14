"""
MLflow experiment tracking and model management utilities.

Handles MLflow setup, model comparison, promotion, and registry operations
for the toxic comment classification training pipeline.
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Any
import shutil

import mlflow
import mlflow.sklearn
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.core import logger, config, PROJECT_ROOT, upload_to_s3


TARGET_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def get_model_configurations() -> Dict[str, Dict[str, Any]]:
    """
    Define different model configurations for multi-model training.

    Returns:
        dict: Dictionary of model configurations with instantiated models and parameters
    """
    return {
        "logistic_regression": {
            "model": LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight="balanced",
                C=1.0,
                verbose=1,
            ),
            "params": {
                "algorithm": "LogisticRegression",
                "C": 1.0,
                "class_weight": "balanced",
                "max_iter": 1000,
                "random_state": 42,
            },
        },
        "random_forest": {
            "model": RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight="balanced",
                max_depth=10,
                min_samples_split=5,
                n_jobs=1,
                verbose=1,
            ),
            "params": {
                "algorithm": "RandomForestClassifier",
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 5,
                "class_weight": "balanced",
                "random_state": 42,
            },
        },
        "xgboost": {
            "model": XGBClassifier(
                n_estimators=100,
                random_state=42,
                learning_rate=0.1,
                max_depth=6,
                verbosity=1,
            ),
            "params": {
                "algorithm": "XGBClassifier",
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 6,
                "random_state": 42,
            },
        },
    }


class ExperimentTracker:
    """
    Manages MLflow experiment tracking and model operations for toxic comment classification.

    Provides methods for setting up MLflow tracking, comparing models, promoting winners,
    and future model registry operations.
    """

    def __init__(self):
        """Initialize the experiment tracker."""
        self.experiment_id = None
        self.tracking_uri = None
        self.experiment_name = None

    def setup_tracking(self) -> str:
        """
        Initialize MLflow tracking based on environment configuration.

        Returns:
            str: The experiment ID
        """
        mlflow_config = config["mlflow"]

        self.tracking_uri = mlflow_config["tracking_uri"]
        mlflow.set_tracking_uri(self.tracking_uri)
        logger.info(f"MLflow tracking URI set to: {self.tracking_uri}")

        self.experiment_name = mlflow_config["experiment_name"]
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                self.experiment_id = mlflow.create_experiment(self.experiment_name)
                logger.info(
                    f"Created new MLflow experiment: {self.experiment_name} (ID: {self.experiment_id})"
                )
            else:
                self.experiment_id = experiment.experiment_id
                logger.info(
                    f"Using existing MLflow experiment: {self.experiment_name} (ID: {self.experiment_id})"
                )
        except Exception as e:
            logger.warning(
                f"Error setting up experiment: {e}. Using default experiment."
            )
            self.experiment_id = "0"

        mlflow.set_experiment(self.experiment_name)
        return self.experiment_id

    def identify_best_model(
        self, model_metrics: Dict[str, Dict[str, Any]]
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Identify the best performing model based on mean ROC-AUC score.

        Args:
            model_metrics: Dictionary of model metrics

        Returns:
            tuple: Best model name and its metrics
        """
        if not model_metrics:
            raise ValueError("No models were trained successfully")

        best_model_name = None
        best_score = -1
        best_metrics = None

        logger.info("Comparing model performances:")
        for model_name, metrics in model_metrics.items():
            mean_auc = metrics["test_mean_roc_auc"]
            logger.info(f"  {model_name}: Mean ROC-AUC = {mean_auc:.4f}")

            if mean_auc > best_score:
                best_score = mean_auc
                best_model_name = model_name
                best_metrics = metrics

        logger.info(
            f"Best performing model: {best_model_name} (ROC-AUC: {best_score:.4f})"
        )
        return best_model_name, best_metrics

    def promote_model(
        self,
        best_model_name: str,
        trained_models: Dict[str, Any],
        best_metrics: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> None:
        """
        Promote the best model to the stable deployment path based on environment.

        Args:
            best_model_name: Name of the best model
            trained_models: Dictionary of trained models
            best_metrics: Metrics of the best model
            metadata: Dataset metadata
        """
        env = config["env"]
        model_path = config["paths"]["model"]
        model_metadata_path = config["paths"]["model_metadata"]

        best_pipeline = trained_models[best_model_name]

        complete_metadata = {
            "model_info": {
                "model_type": "toxic_comment_classifier",
                "sklearn_pipeline": True,
                "multi_label": True,
                "created_at": datetime.now().isoformat(),
                "target_columns": TARGET_COLS,
                "best_model_algorithm": best_metrics["algorithm"],
                "promoted_from_experiment": True,
            },
            "dataset_metadata": metadata,
            "training_metrics": best_metrics,
            "mlflow_info": {
                "experiment_name": self.experiment_name,
                "tracking_uri": self.tracking_uri,
                "environment": env,
            },
        }

        logger.info(f"Promoting {best_model_name} to stable deployment path...")

        if env == "production":
            temp_dir = PROJECT_ROOT / "assets/models"
            temp_dir.mkdir(exist_ok=True)

            model_local_path = temp_dir / Path(model_path).name
            logger.info(
                f"Saving best model temporarily to {model_local_path} for S3 upload..."
            )
            joblib.dump(best_pipeline, model_local_path)

            metadata_local_path = temp_dir / Path(model_metadata_path).name
            logger.info(
                f"Saving metadata temporarily to {metadata_local_path} for S3 upload..."
            )
            with open(metadata_local_path, "w") as f:
                json.dump(complete_metadata, f, indent=2)

            # Upload to S3
            model_s3_key = model_path
            if upload_to_s3(model_local_path, model_s3_key):
                logger.info(f"Removing temporary model file: {model_local_path}")
                os.remove(model_local_path)

            # Upload metadata to S3
            metadata_s3_key = model_metadata_path
            if upload_to_s3(metadata_local_path, metadata_s3_key):
                logger.info(f"Removing temporary metadata file: {metadata_local_path}")
                os.remove(metadata_local_path)

            # Upload mlruns to S3 - this is behaviour specific to production
            # need to refactor this
            mlruns_path = config["mlflow"]["mlruns"]
            temp_mlflow_dir = PROJECT_ROOT / "assets"
            temp_mlflow_dir.mkdir(exist_ok=True)
            mlruns_local_path = temp_mlflow_dir / Path(mlruns_path).name

            # Copy mlruns directory to temporary location
            mlflow_source = PROJECT_ROOT / mlruns_path
            if mlflow_source.exists():
                logger.info(
                    f"Copying MLflow runs from {mlflow_source} to {mlruns_local_path}"
                )
                shutil.copytree(mlflow_source, mlruns_local_path, dirs_exist_ok=True)

                # Upload each file in the mlruns directory
                for root, _, files in os.walk(mlruns_local_path):
                    for file in files:
                        file_path = Path(root) / file
                        relative_path = file_path.relative_to(temp_mlflow_dir)
                        s3_file_key = str(relative_path)

                        if upload_to_s3(file_path, s3_file_key):
                            logger.debug(f"Successfully uploaded {file_path} to S3")
                        else:
                            logger.error(f"Failed to upload {file_path} to S3")

                # Clean up temporary directory
                logger.info(f"Removing temporary mlruns directory: {mlruns_local_path}")
                shutil.rmtree(mlruns_local_path)
            else:
                logger.warning(f"MLflow runs directory not found at {mlflow_source}")

        else:
            # In development, save directly to local paths
            model_local_path = PROJECT_ROOT / model_path
            model_local_path.parent.mkdir(parents=True, exist_ok=True)

            logger.info(f"Saving best model locally to {model_local_path}...")
            joblib.dump(best_pipeline, model_local_path)
            model_file_size = model_local_path.stat().st_size / (1024 * 1024)
            logger.info(
                f"Best model saved successfully! File size: {model_file_size:.2f} MB"
            )

            # Save metadata
            metadata_local_path = PROJECT_ROOT / model_metadata_path
            logger.info(f"Saving metadata locally to {metadata_local_path}...")
            with open(metadata_local_path, "w") as f:
                json.dump(complete_metadata, f, indent=2)
            logger.info("Metadata saved successfully!")

    def log_training_summary(
        self, trained_models: Dict[str, Any], best_metrics: Dict[str, Any]
    ) -> None:
        """
        Log final training summary to MLflow.

        Args:
            trained_models: Dictionary of all trained models
            best_metrics: Metrics of the best performing model
        """
        with mlflow.start_run(run_name="training_summary"):
            mlflow.log_param("total_models_trained", len(trained_models))
            mlflow.log_param("best_model", best_metrics["model_name"])
            mlflow.log_param("best_algorithm", best_metrics["algorithm"])
            mlflow.log_metric("best_model_auc", best_metrics["test_mean_roc_auc"])
            mlflow.log_metric(
                "best_model_accuracy", best_metrics["test_exact_match_accuracy"]
            )

    def register_model(
        self, model_name: str, run_id: str, model_metrics: Dict[str, Any]
    ) -> str:
        """
        Register the best model to MLflow Model Registry.

        Args:
            model_name: Name to register the model under
            run_id: MLflow run ID containing the model
            model_metrics: Model performance metrics

        Returns:
            str: Model version number
        """
        registry_model_name = "toxic-comment-classifier"
        model_uri = f"runs:/{run_id}/model_{model_name}"

        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=registry_model_name,
            description=f"Best performing model: {model_name} (AUC: {model_metrics['test_mean_roc_auc']:.4f})",
        )

        client = mlflow.tracking.MlflowClient()
        client.set_model_version_tag(
            name=registry_model_name,
            version=model_version.version,
            key="algorithm",
            value=model_metrics["algorithm"],
        )
        client.set_model_version_tag(
            name=registry_model_name,
            version=model_version.version,
            key="training_date",
            value=datetime.now().isoformat(),
        )

        logger.info(
            f"Model {registry_model_name} v{model_version.version} registered successfully"
        )
        return model_version.version

    def promote_model_to_production_stage(self, model_name: str, version: str) -> None:
        """
        Transition registered model to Production stage

        Args:
            model_name: Name of the registered model
            version: Version to promote
        """
        client = mlflow.tracking.MlflowClient()

        # Transition to Production stage
        client.transition_model_version_stage(
            name=model_name, version=version, stage="Production"
        )

        logger.info(f"Model {model_name} v{version} promoted to Production stage")
