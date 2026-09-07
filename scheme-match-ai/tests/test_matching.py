from datetime import date

from app.matching import match_schemes, score_scheme
from app.models import Profile, Scheme


def profile(**overrides):
    values = {
        "state": "Delhi",
        "age": 30,
        "social_category": "SC",
        "annual_income": 300000,
        "sector": "Services",
        "business_stage": "Idea",
        "loan_required": 100000,
    }
    values.update(overrides)
    return Profile(**values)


def scheme(identifier, name, categories=None):
    return Scheme(
        id=identifier,
        name=name,
        ministry="Test Ministry",
        description="Test description",
        benefit="Test benefit",
        max_assistance=1000000,
        official_url="https://example.com/scheme",
        eligibility={
            "categories": categories or ["SC"],
            "income": {"max": 500000},
            "entrepreneur": True,
            "business_stages": ["Idea"],
            "sectors": ["Services"],
        },
        search_text="SC entrepreneur services business idea",
        source={
            "source_name": "Test fixture",
            "source_type": "other",
            "data_version": "test-1",
            "verification_status": "unverified",
        },
    )


def test_score_components_are_explicit_and_not_probability():
    result = score_scheme(scheme("one", "One"), profile())
    assert result.status == "eligible"
    assert result.score_components.category == 30
    assert result.score_components.income == 20
    assert result.score_components.sector == 15
    assert result.score_components.business_stage == 10
    assert result.score_components.loan_amount == 5
    assert "probability" not in result.model_dump_json().lower()


def test_matching_ranks_eligible_results_and_separates_no_match():
    high = scheme("high", "High")
    low = scheme("low", "Low", categories=["ST"])
    results, needs_information = match_schemes([low, high], profile())
    assert [item.scheme["id"] for item in results] == ["high"]
    assert needs_information == []


def test_unknown_rule_is_not_a_no_match():
    current = scheme("state", "State Scheme")
    current.eligibility.states = ["Delhi"]
    results, needs_information = match_schemes([current], profile(state=None))
    assert results == []
    assert needs_information[0].status == "needs_information"


def test_explanation_separates_eligibility_from_ranking():
    result = score_scheme(scheme("explain", "Explain"), profile())
    assert result.status == "eligible"
    assert result.why_match
    assert result.passed_rules
    assert result.failed_rules == []
    assert result.missing_information == []
    assert result.score_breakdown.model_dump() == result.score_components.model_dump()
    assert any("match score" in factor for factor in result.ranking_factors)
    assert "probability" not in result.model_dump_json().lower()


def test_ineligible_explanation_contains_failed_rule():
    result = score_scheme(scheme("income", "Income", categories=["SC"]), profile(annual_income=600000))
    assert result.status == "not_eligible"
    assert result.why_not_eligible
    assert result.failed_rules[0].rule == "annual_income"
    assert "600,000" in result.failed_rules[0].message
    assert "500,000" in result.failed_rules[0].message


def test_missing_information_explanation_does_not_fail_rule():
    current = scheme("missing", "Missing")
    current.eligibility.income = type(current.eligibility.income)(max=500000)
    result = score_scheme(current, profile(annual_income=None))
    assert result.status == "needs_information"
    assert result.failed_rules == []
    assert result.missing_information[0].field == "annual_income"
    assert result.confidence == "low"


def test_verified_metadata_has_no_unverified_warning():
    current = scheme("verified", "Verified")
    current.source.verification_status = "verified"
    current.source.verified_at = date(2026, 1, 1)
    result = score_scheme(current, profile())
    assert result.verification.status == "verified"
    assert "not yet been verified" not in result.verification.message
    assert result.confidence == "high"