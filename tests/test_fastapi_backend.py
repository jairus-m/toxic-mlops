from unittest.mock import patch
import pandas as pd

TARGET_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def test_health_check(client):
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    json_response = response.json()
    assert json_response["status"] == "healthy"
    assert json_response["model_ready"] is True
    assert json_response["toxicity_labels"] == TARGET_COLS
    assert (
        "timestamp" in json_response
    )  # Check timestamp exists but don't compare value


def test_predict(client, mock_prediction_logger):
    """Test prediction endpoint"""
    response = client.post("/predict", json={"text": "This is an insult."})
    assert response.status_code == 200
    expected_response = {
        "is_toxic": True,
        "predictions": {
            "toxic": True,
            "insult": True,
            "severe_toxic": False,
            "obscene": False,
            "threat": False,
            "identity_hate": False,
        },
    }
    assert response.json() == expected_response
    mock_prediction_logger.info.assert_called_once()


def test_predict_proba(client, mock_prediction_logger):
    """Test probability prediction endpoint"""
    response = client.post("/predict_proba", json={"text": "This is an insult."})
    assert response.status_code == 200
    expected_response = {
        "is_toxic": True,
        "max_toxicity_probability": 0.8,
        "predictions": {
            "toxic": True,
            "insult": True,
            "severe_toxic": False,
            "obscene": False,
            "threat": False,
            "identity_hate": False,
        },
        "probabilities": {
            "toxic": 0.8,
            "severe_toxic": 0.1,
            "obscene": 0.05,
            "threat": 0.02,
            "insult": 0.7,
            "identity_hate": 0.01,
        },
    }
    assert response.json() == expected_response
    mock_prediction_logger.info.assert_called_once()


def test_example(client):
    """Test example endpoint"""
    mock_df = pd.DataFrame(
        {
            "comment_text": ["A random non-toxic comment."],
            "toxic": [0],
            "severe_toxic": [0],
            "obscene": [0],
            "threat": [0],
            "insult": [0],
            "identity_hate": [0],
        }
    )
    with (
        patch("src.fastapi_backend.main.pd.read_csv", return_value=mock_df),
        patch(
            "src.fastapi_backend.main.get_asset_path",
            return_value="/mock/path/examples.csv",
        ),
    ):
        response = client.get("/example")
        assert response.status_code == 200
        expected_response = {
            "comment": "A random non-toxic comment.",
            "actual_labels": {
                "toxic": False,
                "severe_toxic": False,
                "obscene": False,
                "threat": False,
                "insult": False,
                "identity_hate": False,
            },
        }
        assert response.json() == expected_response


def test_feedback(client, mock_prediction_logger):
    """Test feedback endpoint"""
    feedback_data = {
        "request_text": "You are such an idiot!",
        "predicted_labels": {
            "toxic": True,
            "severe_toxic": False,
            "obscene": False,
            "threat": False,
            "insult": True,
            "identity_hate": False,
        },
        "predicted_probabilities": {
            "toxic": 0.8234,
            "severe_toxic": 0.1245,
            "obscene": 0.0892,
            "threat": 0.0156,
            "insult": 0.7123,
            "identity_hate": 0.0234,
        },
        "true_labels": {
            "toxic": True,
            "severe_toxic": False,
            "obscene": False,
            "threat": False,
            "insult": True,
            "identity_hate": False,
        },
        "is_prediction_correct": True,
        "user_comments": "The prediction was accurate.",
    }
    response = client.post("/feedback", json=feedback_data)
    assert response.status_code == 200
    assert "message" in response.json()
    assert "feedback_id" in response.json()
    mock_prediction_logger.info.assert_called_once()
