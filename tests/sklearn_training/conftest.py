"""
Simple test fixtures for sklearn_training tests.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch

# Test data constants
SAMPLE_COMMENTS = [
    "This is a good comment",
    "You are so stupid and ugly",
    "I hate you so much",
    "Great job on this project",
    "What a terrible day",
    "Love this community",
]

SAMPLE_LABELS = np.array(
    [
        [0, 0, 0, 0, 0, 0],  # good comment
        [1, 0, 1, 0, 1, 0],  # toxic, obscene, insult
        [1, 1, 0, 1, 0, 1],  # toxic, severe_toxic, threat, identity_hate
        [0, 0, 0, 0, 0, 0],  # good comment
        [0, 0, 0, 0, 0, 0],  # neutral comment
        [0, 0, 0, 0, 0, 0],  # good comment
    ]
)


@pytest.fixture
def sample_text_data():
    """Sample text data for testing."""
    return SAMPLE_COMMENTS


@pytest.fixture
def sample_labels():
    """Sample labels for testing."""
    return SAMPLE_LABELS


@pytest.fixture
def sample_dataframe():
    """Sample dataframe matching expected dataset structure."""
    return pd.DataFrame(
        {
            "comment_text": SAMPLE_COMMENTS,
            "toxic": SAMPLE_LABELS[:, 0],
            "severe_toxic": SAMPLE_LABELS[:, 1],
            "obscene": SAMPLE_LABELS[:, 2],
            "threat": SAMPLE_LABELS[:, 3],
            "insult": SAMPLE_LABELS[:, 4],
            "identity_hate": SAMPLE_LABELS[:, 5],
        }
    )


@pytest.fixture
def sample_csv_file(tmp_path, sample_dataframe):
    """Create a temporary CSV file with sample data."""
    csv_path = tmp_path / "test_data.csv"
    sample_dataframe.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    return {
        "env": "development",
        "mlflow": {
            "tracking_uri": "http://localhost:5000",
            "experiment_name": "test_experiment",
        },
        "paths": {
            "model": "assets/models/toxic_model.pkl",
            "model_metadata": "assets/models/toxic_model_metadata.json",
            "data": "assets/data/train.csv",
        },
    }


@pytest.fixture(autouse=True)
def mock_external_dependencies():
    """Mock external dependencies like file I/O and S3."""
    with (
        patch("src.core.upload_to_s3", return_value=True),
        patch("joblib.dump"),
        patch("joblib.load"),
    ):
        yield
