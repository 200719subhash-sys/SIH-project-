from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def valid_profile():
    return {
        "state": "Delhi",
        "age": 30,
        "social_category": "SC",
        "annual_income": 300000,
        "sector": "Services",
        "business_stage": "Idea",
        "loan_required": 100000,
        "keywords": [],
    }


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_schemes_endpoint():
    response = client.get("/api/schemes")
    assert response.status_code == 200
    assert len(response.json()) == 6


def test_valid_match_request():
    response = client.post("/api/match", json=valid_profile())
    assert response.status_code == 200
    body = response.json()
    assert body["count"] > 0
    assert "score_components" in body["results"][0]


def test_invalid_profile_returns_validation_error():
    payload = valid_profile()
    payload["age"] = 17
    response = client.post("/api/match", json=payload)
    assert response.status_code == 422


def test_missing_required_profile_field_returns_validation_error():
    payload = valid_profile()
    del payload["sector"]
    response = client.post("/api/match", json=payload)
    assert response.status_code == 422