from functools import lru_cache
from typing import Any, Protocol

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .models import Profile, RetrievalCandidate, Scheme, SchemeCatalogue


class SemanticEncoder(Protocol):
    def encode(self, texts: list[str]) -> list[list[float]]:
        ...


def scheme_search_text(scheme: Scheme) -> str:
    parts = [scheme.name, scheme.ministry, scheme.description, scheme.benefit, scheme.search_text]
    eligibility = scheme.eligibility
    for values in (eligibility.categories, eligibility.states, eligibility.districts, eligibility.business_stages, eligibility.sectors, eligibility.business_types, eligibility.education_levels):
        if values:
            parts.extend(values)
    if eligibility.income:
        if eligibility.income.min is not None:
            parts.append(f"income minimum {eligibility.income.min}")
        if eligibility.income.max is not None:
            parts.append(f"income maximum {eligibility.income.max}")
    if scheme.coverage:
        parts.extend(scheme.coverage.states or [])
        parts.extend(scheme.coverage.districts or [])
    if scheme.implementing_agency:
        parts.append(scheme.implementing_agency)
    return " ".join(parts)


def profile_query(profile: Profile | None) -> str:
    if not profile:
        return ""
    parts = []
    for value in (profile.state, profile.district, profile.gender, profile.social_category, profile.sector, profile.business_stage, profile.business_type, profile.education_level):
        if value:
            parts.append(value)
    if profile.age is not None:
        parts.append(f"age {profile.age}")
    if profile.annual_income is not None:
        parts.append(f"income {profile.annual_income}")
    if profile.loan_required is not None:
        parts.append(f"loan {profile.loan_required}")
    parts.extend(profile.keywords)
    return " ".join(parts)


def _lexical_scores(query: str, schemes: list[Scheme]) -> list[float]:
    documents = [query or "profile"] + [scheme_search_text(scheme) for scheme in schemes]
    matrix = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit_transform(documents)
    return [float(value) for value in cosine_similarity(matrix[0:1], matrix[1:]).ravel()]


def _semantic_scores(query: str, schemes: list[Scheme], encoder: SemanticEncoder | None) -> list[float] | None:
    if encoder is None:
        return None
    try:
        vectors = encoder.encode([query] + [scheme_search_text(scheme) for scheme in schemes])
        query_vector = vectors[0]
        scores = []
        query_norm = sum(value * value for value in query_vector) ** 0.5
        for vector in vectors[1:]:
            norm = sum(value * value for value in vector) ** 0.5
            scores.append(sum(left * right for left, right in zip(query_vector, vector)) / (query_norm * norm) if query_norm and norm else 0.0)
        return scores
    except Exception:
        return None


def retrieve_schemes(catalogue: SchemeCatalogue, query: str = "", profile: Profile | None = None, top_k: int = 5, encoder: SemanticEncoder | None = None) -> tuple[list[RetrievalCandidate], str]:
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    schemes = catalogue.schemes
    combined_query = " ".join(part for part in (query.strip(), profile_query(profile)) if part).strip()
    lexical = _lexical_scores(combined_query, schemes)
    semantic = _semantic_scores(combined_query, schemes, encoder)
    method = "hybrid" if semantic is not None else "lexical"
    candidates = []
    for scheme, lexical_score, index in zip(schemes, lexical, range(len(schemes))):
        semantic_score = semantic[index] if semantic is not None else 0.0
        hybrid_score = 0.7 * lexical_score + 0.3 * semantic_score if semantic is not None else lexical_score
        factors = []
        text = combined_query.lower()
        if profile and profile.state and profile.state.lower() in scheme_search_text(scheme).lower():
            factors.append("location relevance")
        if profile and profile.sector and profile.sector.lower() in scheme_search_text(scheme).lower():
            factors.append("sector relevance")
        if any(keyword.lower() in scheme_search_text(scheme).lower() for keyword in (profile.keywords if profile else [])):
            factors.append("keyword relevance")
        if text and lexical_score > 0:
            factors.append("text relevance")
        candidates.append(RetrievalCandidate(scheme=scheme.model_dump(mode="json"), lexical_score=round(lexical_score, 6), semantic_score=round(semantic_score, 6), hybrid_score=round(hybrid_score, 6), retrieval_factors=factors))
    candidates.sort(key=lambda item: (-item.hybrid_score, item.scheme["id"]))
    return candidates[:top_k], method