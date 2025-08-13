import pytest
from unittest.mock import patch, MagicMock


def test_streamlit_frontend_import():
    """
    Test that the Streamlit frontend app can be imported without errors.
    """
    with patch("streamlit.columns", return_value=(MagicMock(), MagicMock())):
        try:
            from src.streamlit_frontend import app

            assert app is not None
        except Exception as e:
            pytest.fail(f"Streamlit frontend app failed to import: {e}")
