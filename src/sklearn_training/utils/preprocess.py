"""
Data preprocessing utilities for toxic comment classification.

This module contains functions for loading, cleaning, and preprocessing
the Jigsaw toxic comment dataset.
"""

import re
from pathlib import Path
import pandas as pd
import numpy as np

from src.core import logger
from src.sklearn_training.utils.experiment_tracking import TARGET_COLS

pd.set_option("future.no_silent_downcasting", True)


def clean_text(text: str) -> str:
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
    """
    Process text data in batches to manage memory.
    Args:
        texts (list): List of text strings
        batch_size (int): Size of each batch
    Returns:
        Generator: Yields batches of text strings
    """
    for i in range(0, len(texts), batch_size):
        yield texts[i : i + batch_size]
