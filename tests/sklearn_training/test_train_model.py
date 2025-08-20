"""
Simple unit tests for train_model module.
"""

import numpy as np
import os
from unittest.mock import patch, Mock

from src.sklearn_training.train_model import create_and_train_model_pipeline, main


class TestCreateAndTrainModelPipeline:
    """Test training pipeline creation."""

    @patch("src.sklearn_training.train_model.psutil.Process")
    @patch("src.sklearn_training.train_model.log_memory_usage")
    @patch("src.sklearn_training.train_model.validate_inputs")
    @patch("src.sklearn_training.train_model.prepare_features")
    @patch("src.sklearn_training.train_model.get_model_configurations")
    @patch("src.sklearn_training.train_model.train_single_model")
    def test_create_and_train_model_pipeline_success(
        self,
        mock_train_single,
        mock_get_configs,
        mock_prepare_features,
        mock_validate,
        mock_log_memory,
        mock_process,
        sample_text_data,
        sample_labels,
    ):
        """Test successful training pipeline creation."""
        X = np.array(sample_text_data)
        y = sample_labels
        metadata = {"dataset_shape": (6, 2)}

        # Mock setup
        mock_process.return_value = Mock()
        mock_prepare_features.return_value = (
            Mock(),
            X[4:],
            y[:4],
            y[4:],
            Mock(),
            X[:4],
        )
        mock_get_configs.return_value = {"model_a": {"params": {"algorithm": "ModelA"}}}

        # Mock successful training
        mock_pipeline = Mock()
        mock_metrics = {"test_mean_roc_auc": 0.85, "model_name": "model_a"}
        mock_train_single.return_value = (mock_pipeline, mock_metrics)

        # Run pipeline
        trained_models, model_metrics = create_and_train_model_pipeline(X, y, metadata)

        # Check results
        assert len(trained_models) == 1
        assert "model_a" in trained_models
        assert trained_models["model_a"] == mock_pipeline


class TestMain:
    """Test main function."""

    @patch("src.sklearn_training.train_model.run_training_pipeline")
    @patch("src.sklearn_training.train_model.logger")
    def test_main_train_model_true(self, mock_logger, mock_run_training):
        """Test main function with TRAIN_MODEL=true."""
        with patch.dict(os.environ, {"TRAIN_MODEL": "true"}):
            main()

        mock_run_training.assert_called_once()

    @patch("src.sklearn_training.train_model.deploy_local_model")
    @patch("src.sklearn_training.train_model.logger")
    def test_main_train_model_false(self, mock_logger, mock_deploy):
        """Test main function with TRAIN_MODEL=false."""
        with patch.dict(os.environ, {"TRAIN_MODEL": "false"}):
            main()

        mock_deploy.assert_called_once()
