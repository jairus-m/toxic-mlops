from unittest.mock import patch, MagicMock
import pandas as pd
import pytest


def test_streamlit_monitoring_import():
    """
    Test that the Streamlit monitoring app can be imported without errors.
    This test ensures that the app initializes correctly by mocking all
    external data dependencies.
    """
    with patch(
        "streamlit.columns", return_value=(MagicMock(), MagicMock(), MagicMock())
    ):
        with patch(
            "src.streamlit_monitoring.utils.data_loader.load_all_logs", return_value=[]
        ):
            with patch(
                "src.streamlit_monitoring.utils.data_loader.load_feedback_logs",
                return_value=[],
            ):
                # Return an empty DataFrame, which is serializable, instead of a MagicMock
                with patch(
                    "src.streamlit_monitoring.utils.data_loader.load_dataset",
                    return_value=pd.DataFrame(),
                ):
                    try:
                        from src.streamlit_monitoring import app

                        assert app is not None
                    except Exception as e:
                        pytest.fail(f"Streamlit monitoring app failed to import: {e}")
