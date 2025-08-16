"""
Memory management utilities for training pipeline.

This module contains functions for monitoring memory usage and
performing garbage collection during model training.
"""

from src.core import logger


def log_memory_usage(process, stage: str, initial_memory: float = None):
    """
    Log current memory usage for monitoring.
    Args:
        process (psutil.Process): Process object
        stage (str): Stage name
        initial_memory (float): Initial memory usage
    """
    current_memory = process.memory_info().rss / 1024 / 1024
    if initial_memory is not None:
        logger.info(
            f"{stage} - Memory: {current_memory:.2f} MB (increase: {current_memory - initial_memory:.2f} MB)"
        )
    else:
        logger.info(f"{stage} - Memory: {current_memory:.2f} MB")
    return current_memory


def clean_memory():
    """Perform garbage collection to free memory."""
    import gc

    gc.collect()
