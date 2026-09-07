from typing import Any

from .models import EligibilityResult, Profile, RuleResult, Scheme


def _known(value: Any) -> bool:
    return value is not None and value != ""


def _rule(rule: str, status: str, message: str) -> RuleResult:
    return RuleResult(rule=rule, status=status, message=message)


def evaluate_eligibility(scheme: Scheme, profile: Profile) -> EligibilityResult:
    criteria = scheme.eligibility
    rules: list[RuleResult] = []

    if criteria.categories:
        if not _known(profile.social_category):
            rules.append(_rule("social_category", "unknown", "Social category is required"))
        elif profile.social_category in criteria.categories:
            rules.append(_rule("social_category", "passed", "Social category is covered"))
        else:
            rules.append(_rule("social_category", "failed", "Social category does not match"))

    if criteria.income is not None:
        income = profile.annual_income
        if not _known(income):
            rules.append(_rule("annual_income", "unknown", "Annual family income is required"))
        elif criteria.income.max is not None and income > criteria.income.max:
            rules.append(_rule("annual_income", "failed", "Annual family income is above the scheme limit"))
        elif criteria.income.min is not None and income < criteria.income.min:
            rules.append(_rule("annual_income", "failed", "Annual family income is below the scheme limit"))
        else:
            rules.append(_rule("annual_income", "passed", "Annual family income is within the scheme limit"))

    if criteria.states:
        if not _known(profile.state):
            rules.append(_rule("state", "unknown", "State is required"))
        elif profile.state in criteria.states:
            rules.append(_rule("state", "passed", "State is covered"))
        else:
            rules.append(_rule("state", "failed", "State is outside the listed coverage"))

    if criteria.min_age is not None:
        if profile.age < criteria.min_age:
            rules.append(_rule("min_age", "failed", "Applicant is below the minimum age"))
        else:
            rules.append(_rule("min_age", "passed", "Applicant meets the minimum age"))

    if criteria.max_age is not None:
        if profile.age > criteria.max_age:
            rules.append(_rule("max_age", "failed", "Applicant is above the maximum age"))
        else:
            rules.append(_rule("max_age", "passed", "Applicant meets the maximum age"))

    boolean_rules = (
        ("entrepreneur", criteria.entrepreneur, profile.entrepreneur, "Scheme is intended for entrepreneurs", "Entrepreneur status is covered"),
        ("student", criteria.student, profile.student, "Student status is required", "Student status is covered"),
        ("disability", criteria.disability, profile.disability, "Disability status is required", "Disability status is covered"),
        ("rural", criteria.rural, profile.rural, "Rural eligibility is required", "Rural eligibility is covered"),
    )
    for name, required, value, failure_message, pass_message in boolean_rules:
        if required is True:
            rules.append(_rule(name, "passed" if value else "failed", pass_message if value else failure_message))

    if criteria.business_stages:
        if not _known(profile.business_stage):
            rules.append(_rule("business_stage", "unknown", "Business stage is required"))
        elif profile.business_stage in criteria.business_stages:
            rules.append(_rule("business_stage", "passed", "Business stage is supported"))
        else:
            rules.append(_rule("business_stage", "failed", "Business stage does not match"))

    passed = [item.rule for item in rules if item.status == "passed"]
    failed = [item.rule for item in rules if item.status == "failed"]
    unknown = [item.rule for item in rules if item.status == "unknown"]
    status = "not_eligible" if failed else "needs_information" if unknown else "eligible"
    reasons = [item.message for item in rules if item.status == "passed"]
    reasons.extend(item.message for item in rules if item.status == "failed")

    return EligibilityResult(
        status=status,
        rules=rules,
        passed_rules=passed,
        failed_rules=failed,
        missing_information=unknown,
        reasons=reasons,
    )


def hard_eligibility(scheme: Scheme | dict[str, Any], profile: Profile) -> tuple[bool, str]:
    """Compatibility helper for callers of the original boolean API."""
    normalized = scheme if isinstance(scheme, Scheme) else Scheme.model_validate(scheme)
    result = evaluate_eligibility(normalized, profile)
    return result.status == "eligible", result.reasons[0] if result.reasons else "Eligible on stated profile constraints"