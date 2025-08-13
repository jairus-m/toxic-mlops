"""
Module for training the toxic comment classification model and saving it.

In a 'development' environment, the model is saved locally.
In a 'production' environment, the model is uploaded to an S3 bucket.
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib
import re
import json
import psutil
from datetime import datetime
from scipy import sparse

from src.core import (
    logger,
    config,
    PROJECT_ROOT,
    upload_to_s3,
)
from src.sklearn_training.utils.data_loader import download_kaggle_dataset

pd.set_option("future.no_silent_downcasting", True)

TARGET_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


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


def create_and_train_model_pipeline(X, y, metadata: dict) -> tuple[Pipeline, dict]:
    """
    Create and train the toxic comment classification pipeline.

    Args:
        X (np.ndarray): Features (cleaned comments).
        y (np.ndarray): Labels (toxicity indicators).
        metadata (dict): Dataset metadata.

    Returns:
        tuple: The trained pipeline and training metrics
    """
    logger.info("Creating and training the model pipeline...")

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

    # Log memory after split
    split_memory = process.memory_info().rss / 1024 / 1024
    logger.info(f"Memory usage after split: {split_memory:.2f} MB")

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

    # First pass: Build vocabulary
    logger.info("Building TF-IDF vocabulary from full dataset...")
    try:
        vectorizer.fit(X_train)
        logger.info("Vocabulary building completed")
        logger.info(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    except Exception as e:
        logger.error(f"Vocabulary building failed with error: {str(e)}")
        raise

    # Second pass: Transform in batches
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

    logger.info("Combining transformed batches...")
    X_train_transformed = sparse.vstack(transformed_batches)
    logger.info(f"Combined shape: {X_train_transformed.shape}")

    # Free memory
    del transformed_batches
    import gc

    gc.collect()

    logger.info("Training classifier...")
    classifier = MultiOutputClassifier(
        LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight="balanced",
            C=1.0,
            verbose=1,
        ),
        n_jobs=1,
    )

    try:
        classifier.fit(X_train_transformed, y_train)
        logger.info("Classifier training completed successfully")
    except Exception as e:
        logger.error(f"Classifier training failed with error: {str(e)}")
        raise

    pipeline = Pipeline([("tfidf", vectorizer), ("classifier", classifier)])

    logger.info("Evaluating model performance...")
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
    test_proba_array = np.vstack([np.column_stack(batch) for batch in test_probas])

    train_score = classifier.score(X_train_transformed, y_train)

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

        per_label_metrics[col] = {"accuracy": float(accuracy), "roc_auc": float(auc)}
        logger.info(f"  {col} - Accuracy: {accuracy:.4f}, ROC-AUC: {auc:.4f}")

    exact_match_accuracy = accuracy_score(y_test, test_predictions)
    mean_roc_auc = np.mean(roc_auc_scores)

    logger.info("Model training completed!")
    logger.info(f"Training accuracy (mean): {train_score:.4f}")
    logger.info(f"Test exact match accuracy: {exact_match_accuracy:.4f}")
    logger.info(f"Test mean ROC-AUC: {mean_roc_auc:.4f}")

    final_memory = process.memory_info().rss / 1024 / 1024
    logger.info(f"Final memory usage: {final_memory:.2f} MB")
    logger.info(f"Peak memory increase: {final_memory - initial_memory:.2f} MB")

    training_metrics = {
        "training_accuracy": float(train_score),
        "test_exact_match_accuracy": float(exact_match_accuracy),
        "test_mean_roc_auc": float(mean_roc_auc),
        "per_label_metrics": per_label_metrics,
        "train_size": int(X_train.shape[0]),
        "test_size": int(X_test.shape[0]),
        "model_parameters": {
            "tfidf_max_features": 10000,
            "tfidf_ngram_range": (1, 2),
            "classifier": "LogisticRegression",
            "class_weight": "balanced",
            "multi_output": True,
            "vocabulary_size": len(vectorizer.vocabulary_),
        },
    }

    return pipeline, training_metrics


def save_model_and_metadata(
    pipeline: Pipeline, metadata: dict, training_metrics: dict
) -> None:
    """
    Saves the trained model pipeline and associated metadata.

    - In 'development', saves to local file paths defined in config.
    - In 'production', saves to temporary local files, uploads to S3,
      and then deletes the temporary files.
    """
    env = config["env"]
    model_path = config["paths"]["model"]
    model_metadata_path = config["paths"]["model_metadata"]

    complete_metadata = {
        "model_info": {
            "model_type": "toxic_comment_classifier",
            "sklearn_pipeline": True,
            "multi_label": True,
            "created_at": datetime.now().isoformat(),
            "target_columns": TARGET_COLS,
        },
        "dataset_metadata": metadata,
        "training_metrics": training_metrics,
    }

    if env == "production":
        # Save to temporary local files first for uploading
        temp_dir = PROJECT_ROOT / "assets/models"
        temp_dir.mkdir(exist_ok=True)

        model_local_path = temp_dir / Path(model_path).name
        logger.info(f"Saving model temporarily to {model_local_path} for S3 upload...")
        joblib.dump(pipeline, model_local_path)

        metadata_local_path = temp_dir / Path(model_metadata_path).name
        logger.info(
            f"Saving metadata temporarily to {metadata_local_path} for S3 upload..."
        )
        with open(metadata_local_path, "w") as f:
            json.dump(complete_metadata, f, indent=2)

        model_s3_key = model_path
        if upload_to_s3(model_local_path, model_s3_key):
            logger.info(f"Removing temporary model file: {model_local_path}")
            os.remove(model_local_path)

        metadata_s3_key = model_metadata_path
        if upload_to_s3(metadata_local_path, metadata_s3_key):
            logger.info(f"Removing temporary metadata file: {metadata_local_path}")
            os.remove(metadata_local_path)

    else:
        # In development, save directly to local paths
        model_local_path = PROJECT_ROOT / model_path
        model_local_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving model locally to {model_local_path}...")
        joblib.dump(pipeline, model_local_path)
        model_file_size = model_local_path.stat().st_size / (1024 * 1024)
        logger.info(f"Model saved successfully! File size: {model_file_size:.2f} MB")

        # Save metadata
        metadata_local_path = PROJECT_ROOT / model_metadata_path
        logger.info(f"Saving metadata locally to {metadata_local_path}...")
        with open(metadata_local_path, "w") as f:
            json.dump(complete_metadata, f, indent=2)
        logger.info("Metadata saved successfully!")


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
    Main entry point for the toxic comment classification training process.
    """
    logger.info("Starting Jigsaw Toxic Comment Classification Model Training...")

    try:
        local_data_path = download_kaggle_dataset()

        X, y, metadata = load_and_preprocess_data(local_data_path)

        pipeline, training_metrics = create_and_train_model_pipeline(X, y, metadata)

        feature_importance = analyze_feature_importance(pipeline)
        training_metrics["feature_importance"] = feature_importance

        save_model_and_metadata(pipeline, metadata, training_metrics)

        logger.info("Training process completed successfully!")
        logger.info(
            f"Final model performance - Mean ROC-AUC: {training_metrics['test_mean_roc_auc']:.4f}"
        )

    except Exception as e:
        logger.error(f"An unexpected error occurred during training: {e}")
        raise


if __name__ == "__main__":
    run_training()
