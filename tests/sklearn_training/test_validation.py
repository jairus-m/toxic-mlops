"""
Simple unit tests for validation module.
"""

import pytest
import numpy as np
from unittest.mock import patch

from src.sklearn_training.utils.validation import validate_inputs


class TestValidateInputs:
    """Test input validation function."""

    @patch("src.sklearn_training.utils.validation.logger")
    def test_validate_inputs_success(
        self, mock_logger, sample_text_data, sample_labels
    ):
        """Test successful validation."""
        X = np.array(sample_text_data)
        y = sample_labels

        # Should not raise exception
        validate_inputs(X, y)
        mock_logger.info.assert_called_once()

    def test_validate_inputs_none_inputs(self):
        """Test validation with None inputs."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_inputs(None, None)

    def test_validate_inputs_length_mismatch(self):
        """Test validation with mismatched lengths."""
        X = np.array(["comment1", "comment2", "comment3"])
        y = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])

        with pytest.raises(ValueError, match="same length"):
            validate_inputs(X, y)

    def test_validate_inputs_wrong_columns(self):
        """Test validation with wrong number of target columns."""
        X = np.array(["comment1", "comment2"])
        y = np.array([[1, 0, 0], [0, 1, 0]])  # Only 3 columns instead of 6

        with pytest.raises(ValueError, match="must have 6 columns"):
            validate_inputs(X, y)
