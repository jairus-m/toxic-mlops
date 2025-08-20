"""
Simple unit tests for experiment tracking module.
"""

from unittest.mock import patch

from src.sklearn_training.utils.experiment_tracking import (
    get_model_configurations,
    ExperimentTracker,
)


class TestGetModelConfigurations:
    """Test model configuration function."""

    def test_get_model_configurations_structure(self):
        """Test that model configurations have correct structure."""
        configs = get_model_configurations()

        # Check expected models are present
        expected_models = ["logistic_regression", "random_forest", "xgboost"]
        for model_name in expected_models:
            assert model_name in configs
            assert "model" in configs[model_name]
            assert "params" in configs[model_name]
            assert "algorithm" in configs[model_name]["params"]


class TestExperimentTracker:
    """Test experiment tracker class."""

    def test_experiment_tracker_init(self):
        """Test tracker initialization."""
        tracker = ExperimentTracker()
        assert tracker.experiment_id is None
        assert tracker.tracking_uri is None
        assert tracker.experiment_name is None

    @patch("src.sklearn_training.utils.experiment_tracking.mlflow")
    def test_setup_tracking_new_experiment(self, mock_mlflow, mock_config):
        """Test setting up tracking with new experiment."""
        with patch(
            "src.sklearn_training.utils.experiment_tracking.config", mock_config
        ):
            mock_mlflow.get_experiment_by_name.return_value = None
            mock_mlflow.create_experiment.return_value = "new_experiment_id"

            tracker = ExperimentTracker()
            experiment_id = tracker.setup_tracking()

            assert tracker.experiment_id == "new_experiment_id"
            assert experiment_id == "new_experiment_id"

    def test_identify_best_model_success(self):
        """Test identifying best model from metrics."""
        tracker = ExperimentTracker()

        model_metrics = {
            "model_a": {"test_mean_roc_auc": 0.85},
            "model_b": {"test_mean_roc_auc": 0.92},
            "model_c": {"test_mean_roc_auc": 0.88},
        }

        with patch("src.sklearn_training.utils.experiment_tracking.logger"):
            best_name, best_metrics = tracker.identify_best_model(model_metrics)

        assert best_name == "model_b"
        assert best_metrics["test_mean_roc_auc"] == 0.92
