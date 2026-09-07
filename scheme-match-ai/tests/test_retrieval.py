from app.catalogue import load_catalogue
from app.main import DATA
from app.models import Profile, Scheme
from app.retrieval import profile_query, retrieve_schemes, scheme_search_text


class FakeEncoder:
    def encode(self, texts):
        return [[1.0, 0.0] if index == 0 else [0.0, 1.0] for index, _ in enumerate(texts)]


def catalogue():
    return load_catalogue(DATA)


def profile(**overrides):
    values = {
        "state": "Delhi",
        "age": 30,
        "social_category": "SC",
        "annual_income": 300000,
        "sector": "Services",
        "business_stage": "Idea",
        "keywords": ["tailoring"],
    }
    values.update(overrides)
    return Profile(**values)


def test_search_text_uses_only_catalogue_fields():
    current = catalogue().schemes[0]
    text = scheme_search_text(current)
    assert current.name in text
    assert current.description in text
    assert "invented" not in text


def test_lexical_retrieval_is_deterministic_and_profile_aware():
    first, method = retrieve_schemes(catalogue(), query="small business finance", profile=profile(state="Delhi", keywords=["entrepreneur"]), top_k=3)
    second, second_method = retrieve_schemes(catalogue(), query="small business finance", profile=profile(state="Delhi", keywords=["entrepreneur"]), top_k=3)
    assert method == "lexical"
    assert second_method == "lexical"
    assert [item.scheme["id"] for item in first] == [item.scheme["id"] for item in second]
    assert any("keyword relevance" in item.retrieval_factors for item in first)
    assert "Delhi" in profile_query(profile(state="Delhi"))


def test_semantic_encoder_is_optional_and_hybrid_score_is_bounded():
    candidates, method = retrieve_schemes(catalogue(), query="business finance", top_k=2, encoder=FakeEncoder())
    assert method == "hybrid"
    assert len(candidates) == 2
    assert all(0 <= item.hybrid_score <= 1 for item in candidates)
    assert all(item.semantic_score >= 0 for item in candidates)


def test_missing_profile_facts_do_not_add_assumptions():
    candidates, _ = retrieve_schemes(catalogue(), query="", profile=Profile(sector=None), top_k=6)
    assert all("sector relevance" not in item.retrieval_factors for item in candidates)


def test_retrieval_does_not_produce_eligibility_status():
    candidates, _ = retrieve_schemes(catalogue(), query="SC business", top_k=3)
    assert candidates
    assert not hasattr(candidates[0], "status")


def test_empty_query_and_top_k_validation():
    candidates, _ = retrieve_schemes(catalogue(), query="", top_k=1)
    assert len(candidates) == 1
    try:
        retrieve_schemes(catalogue(), top_k=0)
    except ValueError as error:
        assert "top_k" in str(error)
    else:
        raise AssertionError("top_k=0 should be rejected")
