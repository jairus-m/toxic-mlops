"""
Simple unit tests for preprocess module.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from src.sklearn_training.utils.preprocess import clean_text, load_and_preprocess_data


class TestCleanText:
    """Test text cleaning function."""

    def test_clean_text_basic(self):
        """Test basic text cleaning."""
        assert clean_text("Hello World!") == "hello world"
        assert clean_text("Test@#$123") == "test"
        assert clean_text("") == ""
        assert clean_text(None) == ""


class TestLoadAndPreprocessData:
    """Test data loading function."""

    @patch("src.sklearn_training.utils.preprocess.logger")
    def test_load_and_preprocess_data_success(self, mock_logger, sample_csv_file):
        """Test successful data loading."""
        X, y, metadata = load_and_preprocess_data(sample_csv_file)

        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert isinstance(metadata, dict)
        assert len(X) == len(y)
        assert "dataset_shape" in metadata

    @patch("src.sklearn_training.utils.preprocess.logger")
    def test_load_and_preprocess_data_missing_columns(self, mock_logger, tmp_path):
        """Test error when target columns missing."""
        df = pd.DataFrame({"comment_text": ["test"], "toxic": [1]})
        csv_path = tmp_path / "incomplete.csv"
        df.to_csv(csv_path, index=False)

        with pytest.raises(ValueError, match="Missing target columns"):
            load_and_preprocess_data(csv_path)
