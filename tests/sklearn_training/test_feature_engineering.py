"""
Simple unit tests for feature engineering module.
"""

import numpy as np
from unittest.mock import patch, Mock

from src.sklearn_training.utils.feature_engineering import prepare_features


class TestPrepareFeatures:
    """Test feature preparation function."""

    @patch("src.sklearn_training.utils.feature_engineering.train_test_split")
    @patch("src.sklearn_training.utils.feature_engineering.TfidfVectorizer")
    @patch("src.sklearn_training.utils.feature_engineering.process_in_batches")
    @patch("src.sklearn_training.utils.feature_engineering.clean_memory")
    @patch("psutil.Process")
    def test_prepare_features_success(
        self,
        mock_process,
        mock_clean_memory,
        mock_process_in_batches,
        mock_vectorizer_class,
        mock_train_test_split,
        sample_text_data,
        sample_labels,
    ):
        """Test successful feature preparation."""
        X = np.array(sample_text_data)
        y = sample_labels

        # Mock train_test_split
        X_train, X_test = X[:4], X[4:]
        y_train, y_test = y[:4], y[4:]
        mock_train_test_split.return_value = (X_train, X_test, y_train, y_test)

        # Mock TF-IDF vectorizer
        mock_vectorizer = Mock()
        mock_vectorizer.vocabulary_ = {"test": 0, "word": 1}
        mock_vectorizer_class.return_value = mock_vectorizer

        # Mock process and batch processing
        mock_process.return_value = Mock()
        mock_process_in_batches.return_value = [X_train]

        with patch("scipy.sparse.vstack") as mock_vstack:
            mock_sparse_matrix = Mock()
            mock_sparse_matrix.shape = (4, 100)
            mock_vstack.return_value = mock_sparse_matrix

            result = prepare_features(X, y)

            # Should return 6-tuple
            assert len(result) == 6
            (
                X_train_transformed,
                X_test_out,
                y_train_out,
                y_test_out,
                vectorizer,
                X_train_out,
            ) = result

            # Check basic structure
            assert len(X_train_out) == len(X_train)
            assert len(X_test_out) == len(X_test)

    def test_prepare_features_constants(self):
        """Test that constants have expected values."""
        from src.sklearn_training.utils.feature_engineering import (
            BATCH_SIZE,
            TFIDF_MAX_FEATURES,
            TEST_SIZE,
            RANDOM_STATE,
        )

        assert BATCH_SIZE == 2000
        assert TFIDF_MAX_FEATURES == 10000
        assert TEST_SIZE == 0.2
        assert RANDOM_STATE == 42
