import pytest
from unittest.mock import patch
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from src.sklearn_training import train_model


@patch("src.sklearn_training.train_model.download_kaggle_dataset")
@patch("src.sklearn_training.train_model.load_and_preprocess_data")
@patch("src.sklearn_training.train_model.create_and_train_model_pipeline")
@patch("src.sklearn_training.train_model.analyze_feature_importance")
@patch("src.sklearn_training.utils.experiment_tracking.ExperimentTracker.promote_model")
def test_main_smoke(
    mock_promote_model,
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

    # Mock data and metadata
    mock_X = np.array(["a", "b", "c"])
    mock_y = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]])
    mock_metadata = {"dataset_shape": (3, 2)}

    # Create a real, picklable pipeline for the test
    vectorizer = TfidfVectorizer()
    dummy_estimator = LogisticRegression()
    classifier = MultiOutputClassifier(estimator=dummy_estimator)

    # Fit with dummy data that has multiple classes to avoid ValueError
    dummy_X = vectorizer.fit_transform(["good comment", "bad comment"])
    dummy_y = np.array([[0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1]])
    classifier.fit(dummy_X, dummy_y)

    mock_pipeline = Pipeline([("tfidf", vectorizer), ("classifier", classifier)])

    mock_metrics = {
        "test_mean_roc_auc": 0.95,
        "test_exact_match_accuracy": 0.85,
        "model_name": "mock_model",
        "algorithm": "mock_algorithm",
    }
    mock_trained_models = {"mock_model": mock_pipeline}
    mock_model_metrics = {"mock_model": mock_metrics}

    mock_load_and_preprocess_data.return_value = (mock_X, mock_y, mock_metadata)
    mock_create_and_train_model_pipeline.return_value = (
        mock_trained_models,
        mock_model_metrics,
    )
    mock_analyze_feature_importance.return_value = {"top_toxic_features": []}

    try:
        # Run the main training function
        train_model.main()
    except Exception as e:
        pytest.fail(f"main() raised an exception: {e}")

    # Assert that the key functions in the training pipeline were called
    mock_download_kaggle_dataset.assert_called_once()
    mock_load_and_preprocess_data.assert_called_once_with("dummy_path.csv")
    mock_create_and_train_model_pipeline.assert_called_once_with(
        mock_X, mock_y, mock_metadata
    )
    mock_analyze_feature_importance.assert_called_once_with(mock_pipeline)
    mock_promote_model.assert_called_once()

    # Check that the final metrics dictionary passed to promote_model is correct
    final_call_args = mock_promote_model.call_args[0]
    assert final_call_args[0] == "mock_model"
    assert final_call_args[1] == mock_trained_models
    assert "feature_importance" in final_call_args[2]
    assert final_call_args[2]["test_mean_roc_auc"] == 0.95
