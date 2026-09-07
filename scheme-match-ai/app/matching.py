from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .eligibility import evaluate_eligibility
from .explanations import build_explanation
from .models import MatchResult, Profile, Scheme, ScoreComponents


def build_profile_text(profile: Profile) -> str:
    return " ".join(
        [
            profile.social_category or "",
            profile.state or "",
            profile.sector or "",
            profile.business_stage or "",
            profile.business_type or "",
            profile.education_level or "",
            "entrepreneur" if profile.entrepreneur else "aspiring entrepreneur",
            "student" if profile.student else "",
            "rural" if profile.rural else "",
            "disability" if profile.disability else "",
            " ".join(profile.keywords),
        ]
    )


def _text_similarity(profile: Profile, scheme: Scheme) -> float:
    corpus = [build_profile_text(profile), scheme.search_text]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(corpus)
    return float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0])


def score_scheme(scheme: Scheme | dict[str, Any], profile: Profile) -> MatchResult:
    normalized = scheme if isinstance(scheme, Scheme) else Scheme.model_validate(scheme)
    eligibility = evaluate_eligibility(normalized, profile)
    components = ScoreComponents()

    if eligibility.status != "not_eligible":
        criteria = normalized.eligibility
        if profile.social_category and profile.social_category in (criteria.categories or []):
            components.category = 30
        if criteria.income is not None and criteria.income.max is not None:
            components.income = 20
        if profile.state and profile.state in (criteria.states or []):
            components.state = 10
        if profile.sector and profile.sector.lower() in [item.lower() for item in (criteria.sectors or [])]:
            components.sector = 15
        if profile.business_stage in (criteria.business_stages or []):
            components.business_stage = 10
        if profile.loan_required and normalized.max_assistance >= profile.loan_required:
            components.loan_amount = 5
        if profile.rural and criteria.rural is True:
            components.rural = 5
        components.text_similarity = _text_similarity(profile, normalized) * 5

    score = min(99.0, round(components.total, 1))
    result = MatchResult(
        scheme=normalized.model_dump(mode="json"),
        status=eligibility.status,
        match_score=score,
        rules=eligibility.rules,
        reasons=eligibility.reasons[:5],
        passed_rules=[],
        failed_rules=[],
        missing_information=[],
        score_components=components,
        score_breakdown=components,
        confidence="low",
        verification={"status": normalized.source.verification_status, "message": ""},
    )
    return build_explanation(normalized, eligibility, components, result)


def match_schemes(schemes: list[Scheme], profile: Profile) -> tuple[list[MatchResult], list[MatchResult]]:
    results: list[MatchResult] = []
    needs_information: list[MatchResult] = []
    for scheme in schemes:
        result = score_scheme(scheme, profile)
        if result.status == "eligible":
            results.append(result)
        elif result.status == "needs_information":
            needs_information.append(result)
    results.sort(key=lambda item: item.match_score, reverse=True)
    needs_information.sort(key=lambda item: item.match_score, reverse=True)
    return results, needs_information