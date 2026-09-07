from .models import (
    EligibilityResult,
    MatchResult,
    MissingInformation,
    RuleExplanation,
    Scheme,
    ScoreComponents,
    VerificationExplanation,
)


def _ranking_factors(score: ScoreComponents) -> list[str]:
    factors = []
    labels = {
        "category": "Social category relevance contributed to the match score.",
        "income": "Having a stated income ceiling contributed to the match score.",
        "state": "State coverage contributed to the match score.",
        "sector": "Sector relevance contributed to the match score; it is not treated as a hard rule unless the eligibility data says so.",
        "business_stage": "Business-stage relevance contributed to the match score.",
        "loan_amount": "Requested assistance fitting within the stated maximum contributed to the match score.",
        "rural": "Rural-coverage relevance contributed to the match score.",
        "text_similarity": "Text similarity between the profile and scheme description contributed to the match score.",
    }
    for field, value in score.model_dump().items():
        if value:
            factors.append(labels[field])
    return factors


def _confidence(result: EligibilityResult, scheme: Scheme) -> str:
    if result.missing_information:
        return "low"
    if scheme.source.verification_status != "verified":
        return "medium"
    return "high"


def _verification(scheme: Scheme) -> VerificationExplanation:
    status = scheme.source.verification_status
    if status == "verified":
        message = "Scheme information has verification metadata from the recorded source."
    elif status == "needs_review":
        message = "Scheme information is marked for review and should not be treated as confirmed."
    else:
        message = "Scheme information has not yet been verified against an official source."
    return VerificationExplanation(status=status, message=message)


def build_explanation(
    scheme: Scheme,
    eligibility: EligibilityResult,
    score: ScoreComponents,
    result: MatchResult,
) -> MatchResult:
    passed = [RuleExplanation(rule=item.rule, message=item.message) for item in eligibility.rules if item.status == "passed"]
    failed = [RuleExplanation(rule=item.rule, message=item.message) for item in eligibility.rules if item.status == "failed"]
    missing = [MissingInformation(field=item.rule, message=item.message) for item in eligibility.rules if item.status == "unknown"]
    ranking = _ranking_factors(score)
    why_match = [item.message for item in passed]
    if eligibility.status == "eligible":
        why_match.extend(ranking)
    return result.model_copy(
        update={
            "why_match": why_match,
            "why_not_eligible": [item.message for item in failed],
            "passed_rules": passed,
            "failed_rules": failed,
            "missing_information": missing,
            "score_breakdown": score,
            "ranking_factors": ranking,
            "confidence": _confidence(eligibility, scheme),
            "verification": _verification(scheme),
        }
    )
