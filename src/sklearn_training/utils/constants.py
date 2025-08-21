"""
Constants for the sklearn training module.

This module contains shared constants to avoid circular imports.
"""

# Features of dataset
TARGET_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

TFIDF_MAX_FEATURES = 10000
TFIDF_NGRAM_RANGE = (1, 2)
BATCH_SIZE = 2000
