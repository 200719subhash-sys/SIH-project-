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
    result = body["results"][0]
    assert result["status"] == "eligible"
    assert result["why_match"]
    assert result["passed_rules"]
    assert result["failed_rules"] == []
    assert result["missing_information"] == []
    assert result["verification"]["status"] == "unverified"
    assert "probability" not in response.text.lower()


def test_match_returns_not_eligible_explanations():
    payload = valid_profile()
    payload["annual_income"] = 600000
    response = client.post("/api/match", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["not_eligible"]
    income_failures = [item for item in body["not_eligible"] if item["scheme"]["id"] == "sc-entrepreneur-finance"]
    assert income_failures[0]["why_not_eligible"]
    assert income_failures[0]["failed_rules"][0]["rule"] == "annual_income"


def test_match_returns_missing_information_explanations():
    payload = valid_profile()
    payload["annual_income"] = None
    response = client.post("/api/match", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["needs_information"]
    income_missing = [item for item in body["needs_information"] if item["scheme"]["id"] == "sc-entrepreneur-finance"]
    assert income_missing[0]["missing_information"][0]["field"] == "annual_income"


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


def test_profile_extract_endpoint():
    response = client.post("/api/profile/extract", json={"text": "I am a 27 year old SC woman from Tamil Nadu and need 5 lakh for tailoring."})
    assert response.status_code == 200
    body = response.json()
    assert body["profile"]["age"] == 27
    assert body["profile"]["social_category"] == "SC"
    assert body["profile"]["loan_required"] == 500000
    assert body["profile"]["sector"] == "Services"


def test_profile_extract_rejects_empty_and_large_input():
    assert client.post("/api/profile/extract", json={"text": ""}).status_code == 400
    assert client.post("/api/profile/extract", json={"text": "x" * 5001}).status_code == 400


def test_text_match_reuses_deterministic_matching():
    response = client.post(
        "/api/match/text",
        json={"text": "I am a 27 year old SC woman from Tamil Nadu. My family income is 3.5 lakh. I want to start a tailoring business and need 5 lakh."},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["profile"]["annual_income"] == 350000
    assert body["match"]["results"]
    assert body["match"]["results"][0]["status"] in {"eligible", "needs_information"}


def test_text_match_without_sector_does_not_invent_one():
    response = client.post("/api/match/text", json={"text": "I am an entrepreneur in Karnataka."})
    assert response.status_code == 200
    body = response.json()
    assert body["profile"]["sector"] is None
    assert body["match"] is None
    assert body["profile"]["age"] is None
    assert body["profile"]["annual_income"] is None