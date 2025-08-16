"""
Feature engineering utilities for toxic comment classification.

This module contains functions for TF-IDF vectorization,
data splitting, and feature preparation.
"""

import psutil
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from scipy import sparse

from src.core import logger
from src.sklearn_training.utils.preprocess import process_in_batches
from src.sklearn_training.utils.memory import clean_memory

# Training config constants for TF-IDF vectorizer
BATCH_SIZE = 2000
TFIDF_MAX_FEATURES = 10000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.95
TEST_SIZE = 0.2
RANDOM_STATE = 42


def prepare_features(X, y) -> tuple:
    """
    Prepare features and split data for training.

    Args:
        X (np.ndarray): Features (cleaned comments)
        y (np.ndarray): Labels (toxicity indicators)

    Returns:
        tuple: (X_train_transformed, X_test, y_train, y_test, vectorizer, X_train)
    """
    logger.info("Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y[:, 0],  # Stratify on 'toxic' column
    )

    logger.info(f"Training set size: {X_train.shape[0]:,}")
    logger.info(f"Test set size: {X_test.shape[0]:,}")

    logger.info("Creating TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        min_df=TFIDF_MIN_DF,
        max_df=TFIDF_MAX_DF,
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
    process = psutil.Process()

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
    clean_memory()

    return X_train_transformed, X_test, y_train, y_test, vectorizer, X_train
