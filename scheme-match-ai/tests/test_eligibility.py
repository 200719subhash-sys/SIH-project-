from app.eligibility import evaluate_eligibility
from app.models import Profile, Scheme


def profile(**overrides):
    values = {
        "state": "Delhi",
        "age": 30,
        "social_category": "SC",
        "annual_income": 500000,
        "sector": "Services",
    }
    values.update(overrides)
    return Profile(**values)


def scheme(**eligibility):
    return Scheme(
        id="test-scheme",
        name="Test Scheme",
        ministry="Test Ministry",
        description="Test description",
        benefit="Test benefit",
        max_assistance=1000000,
        official_url="https://example.com/scheme",
        eligibility=eligibility,
        search_text="test scheme service entrepreneur",
        source={
            "source_name": "Test fixture",
            "source_type": "other",
            "data_version": "test-1",
            "verification_status": "unverified",
        },
    )


def test_income_boundaries():
    current = scheme(income={"max": 500000})
    assert evaluate_eligibility(current, profile(annual_income=499999)).status == "eligible"
    assert evaluate_eligibility(current, profile(annual_income=500000)).status == "eligible"
    assert evaluate_eligibility(current, profile(annual_income=500001)).status == "not_eligible"


def test_category_match_and_mismatch():
    current = scheme(categories=["SC"])
    assert evaluate_eligibility(current, profile()).status == "eligible"
    assert evaluate_eligibility(current, profile(social_category="ST")).status == "not_eligible"


def test_age_boundaries():
    current = scheme(min_age=21, max_age=60)
    assert evaluate_eligibility(current, profile(age=21)).status == "eligible"
    assert evaluate_eligibility(current, profile(age=60)).status == "eligible"
    assert evaluate_eligibility(current, profile(age=20)).status == "not_eligible"
    assert evaluate_eligibility(current, profile(age=61)).status == "not_eligible"


def test_boolean_requirements():
    assert evaluate_eligibility(scheme(entrepreneur=True), profile(entrepreneur=False)).status == "not_eligible"
    assert evaluate_eligibility(scheme(student=True), profile(student=False)).status == "not_eligible"
    assert evaluate_eligibility(scheme(disability=True), profile(disability=False)).status == "not_eligible"
    assert evaluate_eligibility(scheme(rural=True), profile(rural=False)).status == "not_eligible"


def test_business_stage():
    current = scheme(business_stages=["Idea"])
    assert evaluate_eligibility(current, profile(business_stage="Idea")).status == "eligible"
    assert evaluate_eligibility(current, profile(business_stage="Existing")).status == "not_eligible"


def test_unknown_state_needs_information():
    result = evaluate_eligibility(scheme(states=["Delhi"]), profile(state=None))
    assert result.status == "needs_information"
    assert result.missing_information == ["state"]
    assert result.rules[0].status == "unknown"


def test_multiple_unknown_eligibility_fields_are_needs_information():
    current = scheme(categories=["SC"], income={"max": 500000}, states=["Delhi"])
    result = evaluate_eligibility(current, profile(social_category=None, annual_income=None, state=None))
    assert result.status == "needs_information"
    assert set(result.missing_information) == {"social_category", "annual_income", "state"}


def test_missing_category_is_not_ineligible():
    result = evaluate_eligibility(scheme(categories=["SC"]), profile(social_category=None))
    assert result.status == "needs_information"
    assert result.failed_rules == []
    assert result.missing_information == ["social_category"]