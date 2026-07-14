import pytest
from fastapi.testclient import TestClient

from app.main import app

SAMPLE_REQUEST = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
    "Dependents": "No", "tenure": 1, "PhoneService": "No",
    "MultipleLines": "No phone service", "InternetService": "DSL",
    "OnlineSecurity": "No", "OnlineBackup": "Yes",
    "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "No", "StreamingMovies": "No",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85, "TotalCharges": 29.85,
}


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is True


def test_predict_returns_valid_response(client):
    response = client.post("/predict", json=SAMPLE_REQUEST)
    assert response.status_code == 200
    body = response.json()
    assert body["churn_prediction"] in ("Yes", "No")
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert "model_version" in body


def test_predict_rejects_invalid_payload(client):
    bad_request = dict(SAMPLE_REQUEST)
    bad_request["gender"] = "Unknown"  # not in the allowed Literal set
    response = client.post("/predict", json=bad_request)
    assert response.status_code == 422


def test_predict_rejects_negative_tenure(client):
    bad_request = dict(SAMPLE_REQUEST)
    bad_request["tenure"] = -5
    response = client.post("/predict", json=bad_request)
    assert response.status_code == 422
