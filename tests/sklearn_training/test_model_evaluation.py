"""
Simple unit tests for model evaluation module.
"""

import numpy as np
from unittest.mock import patch, Mock

from src.sklearn_training.utils.model_evaluation import (
    evaluate_model,
    analyze_feature_importance,
)


class TestEvaluateModel:
    """Test model evaluation function."""

    @patch("src.sklearn_training.utils.model_evaluation.process_in_batches")
    @patch("src.sklearn_training.utils.model_evaluation.clean_memory")
    def test_evaluate_model_success(self, mock_clean_memory, mock_process_in_batches):
        """Test successful model evaluation."""
        # Setup test data
        X_test = np.array(["test comment 1", "test comment 2"])
        y_test = np.array([[1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1]])
        X_train_transformed = Mock()
        y_train = np.array([[1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0]])

        # Mock batch processing
        mock_process_in_batches.return_value = [X_test]

        # Mock classifier
        mock_classifier = Mock()
        mock_classifier.predict.return_value = y_test
        mock_classifier.predict_proba.return_value = [
            np.array([[0.2, 0.8], [0.9, 0.1]]) for _ in range(6)
        ]
        mock_classifier.score.return_value = 0.85

        # Mock vectorizer
        mock_vectorizer = Mock()

        # Run evaluation
        metrics = evaluate_model(
            mock_classifier,
            X_test,
            y_test,
            mock_vectorizer,
            X_train_transformed,
            y_train,
        )

        # Check return structure
        assert isinstance(metrics, dict)
        assert "training_accuracy" in metrics
        assert "test_exact_match_accuracy" in metrics
        assert "test_mean_roc_auc" in metrics
        assert "per_label_metrics" in metrics


class TestAnalyzeFeatureImportance:
    """Test feature importance analysis."""

    @patch("src.sklearn_training.utils.model_evaluation.logger")
    def test_analyze_feature_importance_success(self, mock_logger):
        """Test successful feature importance analysis."""
        # Mock pipeline
        mock_pipeline = Mock()
        mock_tfidf = Mock()
        mock_tfidf.get_feature_names_out.return_value = np.array(
            ["good", "bad", "hate"]
        )

        mock_classifier = Mock()
        mock_classifier.coef_ = [np.array([-0.5, 2.5, 3.0])]

        mock_pipeline.named_steps = {
            "tfidf": mock_tfidf,
            "classifier": Mock(estimators_=[mock_classifier]),
        }

        result = analyze_feature_importance(mock_pipeline)

        # Check return structure
        assert isinstance(result, dict)
        assert "top_toxic_features" in result
        assert "top_non_toxic_features" in result
