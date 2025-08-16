"""
Input validation utilities for training pipeline.

This module contains functions for validating input data
and parameters before model training.
"""

from src.core import logger
from src.sklearn_training.utils.experiment_tracking import TARGET_COLS


def validate_inputs(X, y):
    """
    Validate input data for training.

    Args:
        X (np.ndarray): Features
        y (np.ndarray): Labels

    Raises:
        ValueError: If inputs are invalid
    """
    if X is None or y is None:
        raise ValueError("Input features and labels cannot be None")

    if len(X) == 0 or len(y) == 0:
        raise ValueError("Input features and labels cannot be empty")

    if len(X) != len(y):
        raise ValueError(
            f"Features and labels must have same length: {len(X)} vs {len(y)}"
        )

    if y.shape[1] != len(TARGET_COLS):
        raise ValueError(
            f"Labels must have {len(TARGET_COLS)} columns, got {y.shape[1]}"
        )

    logger.info(
        f"Input validation passed: {len(X)} samples, {y.shape[1]} target columns"
    )
