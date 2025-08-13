import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from src.sklearn_training import train_model


@patch("src.sklearn_training.train_model.download_kaggle_dataset")
@patch("src.sklearn_training.train_model.load_and_preprocess_data")
@patch("src.sklearn_training.train_model.create_and_train_model_pipeline")
@patch("src.sklearn_training.train_model.analyze_feature_importance")
@patch("src.sklearn_training.train_model.save_model_and_metadata")
def test_run_training_smoke(
    mock_save_model_and_metadata,
    mock_analyze_feature_importance,
    mock_create_and_train_model_pipeline,
    mock_load_and_preprocess_data,
    mock_download_kaggle_dataset,
):
    """
    Smoke test for the main training function to ensure it runs without errors.
    """
    # Mock return values to simulate the training process
    mock_download_kaggle_dataset.return_value = "dummy_path.csv"

    # Mock data, metadata, pipeline, and metrics
    mock_X = np.array(["a", "b", "c"])
    mock_y = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]])
    mock_metadata = {"dataset_shape": (3, 2)}
    mock_pipeline = MagicMock()
    mock_metrics = {"test_mean_roc_auc": 0.95}

    mock_load_and_preprocess_data.return_value = (mock_X, mock_y, mock_metadata)
    mock_create_and_train_model_pipeline.return_value = (mock_pipeline, mock_metrics)
    mock_analyze_feature_importance.return_value = {"top_toxic_features": []}

    try:
        # Run the main training function
        train_model.run_training()
    except Exception as e:
        pytest.fail(f"run_training() raised an exception: {e}")

    # Assert that the key functions in the training pipeline were called
    mock_download_kaggle_dataset.assert_called_once()
    mock_load_and_preprocess_data.assert_called_once_with("dummy_path.csv")
    mock_create_and_train_model_pipeline.assert_called_once_with(
        mock_X, mock_y, mock_metadata
    )
    mock_analyze_feature_importance.assert_called_once_with(mock_pipeline)
    mock_save_model_and_metadata.assert_called_once()

    # Check that the final metrics dictionary passed to save_model_and_metadata is correct
    final_call_args = mock_save_model_and_metadata.call_args[0]
    assert final_call_args[0] == mock_pipeline
    assert final_call_args[1] == mock_metadata
    assert "feature_importance" in final_call_args[2]
    assert final_call_args[2]["test_mean_roc_auc"] == 0.95
