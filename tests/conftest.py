"""
This file contains reusable mock fixtures for the tests.
"""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from contextlib import ExitStack
import asyncio
import numpy as np


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
def mock_get_asset_path():
    """
    Fixture to mock get_asset_path function to avoid file system access during tests.
    """
    with patch("src.core.asset_resolution.get_asset_path") as mock:
        # Return a dummy path. The path doesn't have to exist because
        # the functions that use it (like loading a model or data)
        # should also be mocked in the tests where they are used
        mock.return_value = Path("/tmp/dummy_asset")
        yield mock


@pytest.fixture(autouse=True)
def mock_streamlit():
    """
    Fixture to mock all Streamlit functions to prevent actual UI rendering.
    """
    streamlit_patches = [
        patch(
            "streamlit.runtime.scriptrunner.get_script_run_ctx",
            return_value=MagicMock(),
        ),
        patch("streamlit.runtime.state.get_session_state", return_value={}),
        patch("streamlit.set_page_config"),
        patch("streamlit.title"),
        patch("streamlit.markdown"),
        patch("streamlit.subheader"),
        patch("streamlit.button", return_value=False),
        patch("streamlit.text_area", return_value=""),
        patch("streamlit.spinner"),
        patch("streamlit.success"),
        patch("streamlit.error"),
        patch("streamlit.warning"),
        patch("streamlit.write"),
        patch("streamlit.sidebar.title"),
        patch("streamlit.sidebar.radio"),
        patch("streamlit.dataframe"),
        patch("streamlit.plotly_chart"),
        patch("streamlit.altair_chart"),
        patch("streamlit.metric"),
        # Use a function to return a new generator each time, preventing exhaustion
        patch("streamlit.metric"),
        patch.dict("os.environ", {"FASTAPI_BACKEND_URL": "http://mock-backend:8000"}),
    ]

    with ExitStack() as stack:
        for p in streamlit_patches:
            stack.enter_context(p)
        yield


@pytest.fixture
def mock_model():
    """
    Fixture to mock the toxic comment classification model.
    """
    mock = MagicMock()
    # Simulate a prediction for a comment that is 'toxic' and 'insult'
    mock.predict.return_value = np.array([[1, 0, 0, 0, 1, 0]])

    # Simulate probabilities for each of the 6 labels
    mock.predict_proba.return_value = [
        np.array([[0.2, 0.8]]),  # toxic
        np.array([[0.9, 0.1]]),  # severe_toxic
        np.array([[0.95, 0.05]]),  # obscene
        np.array([[0.98, 0.02]]),  # threat
        np.array([[0.3, 0.7]]),  # insult
        np.array([[0.99, 0.01]]),  # identity_hate
    ]
    return mock


@pytest.fixture
def mock_prediction_logger():
    """
    Fixture to mock the prediction logger.
    """
    with patch("src.fastapi_backend.main.prediction_logger") as mock_logger:
        yield mock_logger


@pytest.fixture
def app(mock_model):
    """
    Fixture to create FastAPI app with mocked model.
    Args:
        mock_model: Mock model to use in the app
    """
    with patch(
        "src.fastapi_backend.utils.model_loader.load_model", return_value=mock_model
    ):
        from src.fastapi_backend.main import app

        return app


@pytest.fixture
def client(app):
    """
    Fixture to create test client with mocked app.
    Args:
        app: FastAPI app with mocked model
    """
    from fastapi.testclient import TestClient

    return TestClient(app)
