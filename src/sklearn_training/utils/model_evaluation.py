"""
Model evaluation utilities for toxic comment classification.

This module contains functions for evaluating trained models,
calculating performance metrics, and analyzing feature importance.
"""

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline

from src.core import logger
from sklearn_training.utils.constants import TARGET_COLS, BATCH_SIZE
from src.sklearn_training.utils.preprocess import process_in_batches
from src.sklearn_training.utils.memory import clean_memory


def evaluate_model(
    classifier, X_test, y_test, vectorizer, X_train_transformed, y_train
) -> dict:
    """
    Evaluate a trained model and calculate performance metrics.

    Args:
        classifier: Trained classifier
        X_test (np.ndarray): Test features
        y_test (np.ndarray): Test labels
        vectorizer: Fitted TF-IDF vectorizer
        X_train_transformed: Transformed training features
        y_train (np.ndarray): Training labels

    Returns:
        dict: Performance metrics
    """
    test_predictions = []
    test_probas = []

    for batch_texts in process_in_batches(X_test, BATCH_SIZE):
        batch_transformed = vectorizer.transform(batch_texts)
        batch_pred = classifier.predict(batch_transformed)
        batch_proba = classifier.predict_proba(batch_transformed)

        test_predictions.append(batch_pred)
        test_probas.append([prob[:, 1] for prob in batch_proba])

        del batch_transformed
        clean_memory()

    test_predictions = np.vstack(test_predictions)
    test_proba_array = np.vstack([np.column_stack(batch) for batch in test_probas])

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

    exact_match_accuracy = accuracy_score(y_test, test_predictions)
    mean_roc_auc = np.mean(roc_auc_scores)

    return {
        "training_accuracy": float(train_score),
        "test_exact_match_accuracy": float(exact_match_accuracy),
        "test_mean_roc_auc": float(mean_roc_auc),
        "per_label_metrics": per_label_metrics,
        "roc_auc_scores": roc_auc_scores,
        "test_predictions": test_predictions,
        "test_proba_array": test_proba_array,
    }


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
