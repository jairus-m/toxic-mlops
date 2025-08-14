"""
Module for training the toxic comment classification model and saving it.

In a 'development' environment, the model is saved locally.
In a 'production' environment, the model is uploaded to an S3 bucket.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import re
import psutil
from scipy import sparse
import mlflow
import mlflow.sklearn

from src.core import (
    logger,
    config,
    upload_to_s3,
    download_from_s3,
)
from src.sklearn_training.utils.data_loader import download_kaggle_dataset
from src.sklearn_training.utils.experiment_tracking import (
    ExperimentTracker,
    get_model_configurations,
    TARGET_COLS,
)

pd.set_option("future.no_silent_downcasting", True)


def clean_text(text):
    """
    Basic text cleaning function for comment preprocessing.

    Args:
        text (str): Raw comment text

    Returns:
        str: Cleaned comment text
    """
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = " ".join(text.split())

    return text


def load_and_preprocess_data(data_path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Load and preprocess the Jigsaw toxic comment dataset.

    Args:
        data_path (Path): Path to the toxic comment dataset file.

    Returns:
        tuple: Features (X), labels (y), and metadata dict
    """
    logger.info(f"Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    logger.info(f"Dataset loaded successfully! Shape: {df.shape}")
    logger.info(f"Columns: {df.columns.tolist()}")

    missing_values = df.isnull().sum()
    logger.info(f"Missing values:\n{missing_values}")

    missing_targets = [col for col in TARGET_COLS if col not in df.columns]
    if missing_targets:
        raise ValueError(f"Missing target columns: {missing_targets}")

    logger.info("Preprocessing comment text...")
    df["comment_text_clean"] = df["comment_text"].apply(clean_text)

    label_stats = {}
    logger.info("Label distribution:")
    for col in TARGET_COLS:
        positive_rate = df[col].mean()
        label_stats[col] = {
            "positive_rate": float(positive_rate),
            "positive_count": int(df[col].sum()),
            "total_count": int(len(df)),
        }
        logger.info(
            f"  {col}: {positive_rate:.4f} ({positive_rate * 100:.2f}% positive)"
        )

    X = df["comment_text_clean"].values
    y = df[TARGET_COLS].values

    metadata = {
        "dataset_shape": df.shape,
        "feature_column": "comment_text_clean",
        "target_columns": TARGET_COLS,
        "label_statistics": label_stats,
        "missing_values": missing_values.to_dict(),
        "preprocessing_steps": [
            "lowercase_conversion",
            "special_character_removal",
            "whitespace_normalization",
        ],
    }

    return X, y, metadata


def process_in_batches(texts, batch_size=2000):
    """Process text data in batches to manage memory."""
    for i in range(0, len(texts), batch_size):
        yield texts[i : i + batch_size]


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

    # Log initial memory usage
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024
    logger.info(f"Initial memory usage: {initial_memory:.2f} MB")

    logger.info("Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y[:, 0],  # Stratify on 'toxic' column
    )

    logger.info(f"Training set size: {X_train.shape[0]:,}")
    logger.info(f"Test set size: {X_test.shape[0]:,}")

    logger.info("Creating TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        stop_words="english",
        lowercase=True,
        strip_accents="ascii",
    )

    logger.info("Building TF-IDF vocabulary from training data...")
    try:
        vectorizer.fit(X_train)
        logger.info(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    except Exception as e:
        logger.error(f"Vocabulary building failed: {str(e)}")
        raise

    logger.info("Transforming training data in batches...")
    transformed_batches = []
    BATCH_SIZE = 2000

    for batch_idx, batch_texts in enumerate(process_in_batches(X_train, BATCH_SIZE)):
        batch_transformed = vectorizer.transform(batch_texts)
        transformed_batches.append(batch_transformed)

        if (batch_idx + 1) % 10 == 0:
            current_memory = process.memory_info().rss / 1024 / 1024
            logger.info(
                f"Processed {(batch_idx + 1) * BATCH_SIZE} samples. Memory: {current_memory:.2f} MB"
            )

    X_train_transformed = sparse.vstack(transformed_batches)
    del transformed_batches
    import gc

    gc.collect()

    model_configs = get_model_configurations()
    trained_models = {}
    model_metrics = {}

    # Train each model with MLflow tracking
    for model_name, model_config in model_configs.items():
        logger.info(f"Training {model_name}...")

        with mlflow.start_run(run_name=f"toxic_model_{model_name}"):
            mlflow.log_params(model_config["params"])
            mlflow.log_params(
                {
                    "tfidf_max_features": 10000,
                    "tfidf_ngram_range": "(1, 2)",
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
                continue

            pipeline = Pipeline([("tfidf", vectorizer), ("classifier", classifier)])

            logger.info(f"Evaluating {model_name} performance...")
            test_predictions = []
            test_probas = []

            for batch_texts in process_in_batches(X_test, BATCH_SIZE):
                batch_transformed = vectorizer.transform(batch_texts)
                batch_pred = classifier.predict(batch_transformed)
                batch_proba = classifier.predict_proba(batch_transformed)

                test_predictions.append(batch_pred)
                test_probas.append([prob[:, 1] for prob in batch_proba])

                del batch_transformed
                gc.collect()

            test_predictions = np.vstack(test_predictions)
            test_proba_array = np.vstack(
                [np.column_stack(batch) for batch in test_probas]
            )

            train_score = classifier.score(X_train_transformed, y_train)

            # Calculate per-label metrics
            per_label_metrics = {}
            roc_auc_scores = []

            for i, col in enumerate(TARGET_COLS):
                accuracy = accuracy_score(y_test[:, i], test_predictions[:, i])
                try:
                    auc = roc_auc_score(y_test[:, i], test_proba_array[:, i])
                    roc_auc_scores.append(auc)
                except ValueError:
                    auc = 0.0
                    roc_auc_scores.append(0.0)

                per_label_metrics[col] = {
                    "accuracy": float(accuracy),
                    "roc_auc": float(auc),
                }

                # Log individual metrics to MLflow
                mlflow.log_metric(f"{col}_accuracy", accuracy)
                mlflow.log_metric(f"{col}_auc", auc)

            exact_match_accuracy = accuracy_score(y_test, test_predictions)
            mean_roc_auc = np.mean(roc_auc_scores)

            # Log summary metrics
            mlflow.log_metric("exact_match_accuracy", exact_match_accuracy)
            mlflow.log_metric("mean_auc", mean_roc_auc)
            mlflow.log_metric("training_accuracy", train_score)

            # Log model and wait for it to complete
            model_info = mlflow.sklearn.log_model(
                pipeline,
                "model",
                await_registration_for=60,  # Wait up to 60 seconds for artifact logging
            )

            # Register model using the returned model URI
            registry_model_name = "toxic-comment-classifier"
            logger.info(
                f"Registering model {model_name} from run {mlflow.active_run().info.run_id}"
            )
            mlflow.register_model(
                model_uri=model_info.model_uri, name=registry_model_name
            )
            logger.info(f"Model registered successfully as {registry_model_name}")

            logger.info(f"{model_name} results:")
            logger.info(f"  Training accuracy (mean): {train_score:.4f}")
            logger.info(f"  Test exact match accuracy: {exact_match_accuracy:.4f}")
            logger.info(f"  Test mean ROC-AUC: {mean_roc_auc:.4f}")

            # Store results
            trained_models[model_name] = pipeline
            model_metrics[model_name] = {
                "training_accuracy": float(train_score),
                "test_exact_match_accuracy": float(exact_match_accuracy),
                "test_mean_roc_auc": float(mean_roc_auc),
                "per_label_metrics": per_label_metrics,
                "model_name": model_name,
                "algorithm": model_config["params"]["algorithm"],
                "run_id": mlflow.active_run().info.run_id,
            }

    final_memory = process.memory_info().rss / 1024 / 1024
    logger.info(f"Final memory usage: {final_memory:.2f} MB")
    logger.info(f"Peak memory increase: {final_memory - initial_memory:.2f} MB")

    return trained_models, model_metrics


def analyze_feature_importance(pipeline: Pipeline) -> dict:
    """
    Analyze feature importance for the toxic classification model.

    Args:
        pipeline: Trained pipeline

    Returns:
        dict: Feature importance analysis
    """
    logger.info("Analyzing feature importance...")

    tfidf = pipeline.named_steps["tfidf"]
    feature_names = tfidf.get_feature_names_out()

    toxic_classifier = pipeline.named_steps["classifier"].estimators_[0]
    toxic_coef = toxic_classifier.coef_[0]

    top_toxic_indices = toxic_coef.argsort()[-20:][::-1]
    top_non_toxic_indices = toxic_coef.argsort()[:20]

    feature_importance = {
        "top_toxic_features": [
            {"feature": feature_names[idx], "coefficient": float(toxic_coef[idx])}
            for idx in top_toxic_indices
        ],
        "top_non_toxic_features": [
            {"feature": feature_names[idx], "coefficient": float(toxic_coef[idx])}
            for idx in top_non_toxic_indices
        ],
    }

    logger.info("Top 10 toxic indicators:")
    for item in feature_importance["top_toxic_features"][:10]:
        logger.info(f"  {item['feature']}: {item['coefficient']:.4f}")

    return feature_importance


def run_training():
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


if __name__ == "__main__":
    run_training()
