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