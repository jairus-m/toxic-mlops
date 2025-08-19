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

import mlflow
import mlflow.sklearn
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier
from mlflow.models.signature import infer_signature

from src.core import logger, config, PROJECT_ROOT, upload_to_s3


TARGET_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def get_display_uri(tracking_uri: str) -> str:
    """
    Convert internal Docker URI to localhost for display purposes.
    Args:
        tracking_uri: Internal MLflow tracking URI (e.g., http://mlflow-server:5000)

    Returns:
        str: Display-friendly URI (e.g., http://localhost:5000)
    """
    env = config.get("env", "development")
    if env == "development":
        return tracking_uri.replace("mlflow-server:5000", "localhost:5000")
    return tracking_uri


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
    and model registry methods.
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
        logger.info(f"MLflow tracking URI: {self.tracking_uri}")

        display_uri = get_display_uri(self.tracking_uri)
        logger.info(f"View MLflow UI at: {display_uri}")

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

            experiment_url = f"{self.tracking_uri}/#/experiments/{self.experiment_id}"
            display_experiment_url = get_display_uri(experiment_url)
            logger.info(f"View experiment at: {display_experiment_url}")
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

            # MLflow artifacts are now managed by the remote MLflow server
            logger.info("MLflow experiment data is managed by remote MLflow server")

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

        # Register the best model in MLflow Model Registry
        if "run_id" in best_metrics:
            self._register_best_model(
                best_metrics["run_id"], best_model_name, best_metrics
            )

    def _register_best_model(
        self, run_id: str, model_name: str, metrics: Dict[str, Any]
    ) -> None:
        """
        Register the best model in MLflow Model Registry.

        Args:
            run_id: MLflow run ID containing the model
            model_name: Name of the best model
            metrics: Model performance metrics
        """
        try:
            model_uri = f"runs:/{run_id}/model"
            registry_model_name = "toxic-comment-classifier"

            logger.info(f"Registering best model {model_name} from run {run_id}")
            model_version = mlflow.register_model(
                model_uri=model_uri,
                name=registry_model_name,
                tags={
                    "algorithm": model_name,
                    "training_run": run_id,
                    "mean_auc": str(metrics.get("test_mean_roc_auc", 0)),
                },
            )
            logger.info(
                f"✅ Best model registered as {registry_model_name} version {model_version.version}"
            )
        except Exception as e:
            logger.warning(f"Model registration failed: {e}")

    def log_training_summary(
        self, trained_models: Dict[str, Any], best_metrics: Dict[str, Any]
    ) -> None:
        """
        Log final training summary to MLflow.

        Args:
            trained_models: Dictionary of all trained models
            best_metrics: Metrics of the best performing model
        """
        with mlflow.start_run(run_name="training_summary") as run:
            mlflow.log_param("total_models_trained", len(trained_models))
            mlflow.log_param("best_model", best_metrics["model_name"])
            mlflow.log_param("best_algorithm", best_metrics["algorithm"])
            mlflow.log_metric("best_model_auc", best_metrics["test_mean_roc_auc"])
            mlflow.log_metric(
                "best_model_accuracy", best_metrics["test_exact_match_accuracy"]
            )

            # Log user-friendly URLs
            run_url = f"{self.tracking_uri}/#/experiments/{self.experiment_id}/runs/{run.info.run_id}"
            experiment_url = f"{self.tracking_uri}/#/experiments/{self.experiment_id}"

            user_friendly_run_url = get_display_uri(run_url)
            user_friendly_experiment_url = get_display_uri(experiment_url)

            logger.info("🎯 Training Summary Complete!")
            logger.info(f"📊 View training summary run: {user_friendly_run_url}")
            logger.info(f"🧪 View all experiments: {user_friendly_experiment_url}")
            logger.info(
                f"🏆 Best model: {best_metrics['model_name']} (AUC: {best_metrics['test_mean_roc_auc']:.4f})"
            )
            logger.info(
                "ℹ️  Note: MLflow may also log internal Docker URLs (mlflow-server:5000) - use the localhost URLs above for browser access"
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


def register_model_in_mlflow(pipeline: Pipeline, model_name: str) -> str:
    """
    Register model in MLflow registry.

    Args:
        pipeline: Trained model pipeline
        model_name (str): Name of the model

    Returns:
        str: MLflow run ID
    """
    # Use standard MLflow artifact path convention
    model_info = mlflow.sklearn.log_model(
        pipeline,
        "model",  # Standard MLflow artifact path
        await_registration_for=60,  # Wait up to 60 seconds for artifact logging
    )

    # Register model using the returned model URI
    registry_model_name = "toxic-comment-classifier"
    run_id = mlflow.active_run().info.run_id
    logger.info(f"Registering model {model_name} from run {run_id}")
    logger.info(f"Model URI: {model_info.model_uri}")

    try:
        model_version = mlflow.register_model(
            model_uri=model_info.model_uri,
            name=registry_model_name,
            tags={"algorithm": model_name, "training_run": run_id},
        )
        logger.info(
            f"Model registered successfully as {registry_model_name} version {model_version.version}"
        )
    except Exception as e:
        logger.warning(
            f"Model registration failed: {e}. Model artifacts are still logged."
        )

    return run_id


def train_single_model(
    model_name: str,
    model_config: dict,
    X_train_transformed,
    y_train,
    X_test,
    y_test,
    vectorizer,
    X_train,
) -> tuple:
    """
    Train a single model with MLflow tracking.

    Args:
        model_name (str): Name of the model
        model_config (dict): Model configuration
        X_train_transformed: Transformed training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        vectorizer: TF-IDF vectorizer
        X_train: Original training features

    Returns:
        tuple: (pipeline, metrics_dict) or (None, None) if training failed
    """
    from .model_evaluation import evaluate_model  # Import here to avoid circular import

    logger.info(f"Training {model_name}...")

    with mlflow.start_run(run_name=f"toxic_model_{model_name}"):
        # Log initial parameters
        mlflow.log_params(model_config["params"])
        mlflow.log_params(
            {
                "tfidf_max_features": 10000,  # TFIDF_MAX_FEATURES constant
                "tfidf_ngram_range": str((1, 2)),  # TFIDF_NGRAM_RANGE constant
                "train_size": X_train.shape[0],
                "test_size": X_test.shape[0],
                "vocabulary_size": len(vectorizer.vocabulary_),
            }
        )

        classifier = MultiOutputClassifier(model_config["model"], n_jobs=1)

        try:
            classifier.fit(X_train_transformed, y_train)
            logger.info(f"{model_name} training completed successfully")
        except Exception as e:
            logger.error(f"{model_name} training failed: {str(e)}")
            return None, None

        pipeline = Pipeline([("tfidf", vectorizer), ("classifier", classifier)])

        logger.info(f"Evaluating {model_name} performance...")
        metrics = evaluate_model(
            classifier, X_test, y_test, vectorizer, X_train_transformed, y_train
        )

        # Log metrics to MLflow
        for col in TARGET_COLS:
            mlflow.log_metric(
                f"{col}_accuracy", metrics["per_label_metrics"][col]["accuracy"]
            )
            mlflow.log_metric(
                f"{col}_auc", metrics["per_label_metrics"][col]["roc_auc"]
            )

        mlflow.log_metric("exact_match_accuracy", metrics["test_exact_match_accuracy"])
        mlflow.log_metric("mean_auc", metrics["test_mean_roc_auc"])
        mlflow.log_metric("training_accuracy", metrics["training_accuracy"])

        # Create model signature and input example for proper artifact metadata
        input_example = X_train[:5]  # X_train contains original text strings
        predictions = pipeline.predict(input_example)

        signature = infer_signature(input_example, predictions)

        logger.info(
            f"Starting to log model {model_name} with signature and input example..."
        )
        logger.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

        display_uri = get_display_uri(mlflow.get_tracking_uri())
        logger.info(f"View MLflow UI at: {display_uri}")
        logger.info(f"Active run artifact URI: {mlflow.get_artifact_uri()}")

        try:
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
                signature=signature,
                input_example=input_example,
                await_registration_for=60,
            )
            run_id = mlflow.active_run().info.run_id
            logger.info(
                f"✅ Model {model_name} successfully logged to MLflow run {run_id}"
            )
            logger.info(
                f"Expected artifact location: {mlflow.get_artifact_uri()}/model"
            )

            run_url = f"{mlflow.get_tracking_uri()}/#/experiments/{mlflow.active_run().info.experiment_id}/runs/{run_id}"
            display_run_url = get_display_uri(run_url)
            logger.info(f"View run at: {display_run_url}")
        except Exception as e:
            logger.error(f"❌ Failed to log model {model_name}: {e}")
            import traceback

            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise

        logger.info(f"{model_name} results:")
        logger.info(f"  Training accuracy (mean): {metrics['training_accuracy']:.4f}")
        logger.info(
            f"  Test exact match accuracy: {metrics['test_exact_match_accuracy']:.4f}"
        )
        logger.info(f"  Test mean ROC-AUC: {metrics['test_mean_roc_auc']:.4f}")

        model_metrics = {
            "training_accuracy": metrics["training_accuracy"],
            "test_exact_match_accuracy": metrics["test_exact_match_accuracy"],
            "test_mean_roc_auc": metrics["test_mean_roc_auc"],
            "per_label_metrics": metrics["per_label_metrics"],
            "model_name": model_name,
            "algorithm": model_config["params"]["algorithm"],
            "run_id": run_id,
        }

        return pipeline, model_metrics
