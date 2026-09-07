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
            rules.append(_rule("social_category", "unknown", "Your social category is needed to determine eligibility."))
        elif profile.social_category in criteria.categories:
            rules.append(_rule("social_category", "passed", "Your social category matches the scheme requirement."))
        else:
            rules.append(_rule("social_category", "failed", "Your social category is not included in the scheme requirement."))

    if criteria.income is not None:
        income = profile.annual_income
        if not _known(income):
            rules.append(_rule("annual_income", "unknown", "Your annual family income is needed to determine eligibility."))
        elif criteria.income.max is not None and income > criteria.income.max:
            rules.append(_rule("annual_income", "failed", f"Your annual family income is ₹{income:,.0f}, while this scheme requires income of ₹{criteria.income.max:,.0f} or below."))
        elif criteria.income.min is not None and income < criteria.income.min:
            rules.append(_rule("annual_income", "failed", f"Your annual family income is ₹{income:,.0f}, while this scheme requires income of at least ₹{criteria.income.min:,.0f}."))
        else:
            rules.append(_rule("annual_income", "passed", "Your annual family income is within the scheme's stated income limit."))

    if criteria.states:
        if not _known(profile.state):
            rules.append(_rule("state", "unknown", "Your state is needed to determine eligibility."))
        elif profile.state in criteria.states:
            rules.append(_rule("state", "passed", "Your state is included in the scheme's stated coverage."))
        else:
            rules.append(_rule("state", "failed", "Your state is outside the scheme's stated coverage."))

    if criteria.min_age is not None:
        if profile.age is None:
            rules.append(_rule("age", "unknown", "Your age is needed to determine eligibility."))
        elif profile.age < criteria.min_age:
            rules.append(_rule("age", "failed", f"Your age is below the scheme's minimum age of {criteria.min_age}."))
        else:
            rules.append(_rule("age", "passed", "Your age meets the scheme's minimum age requirement."))

    if criteria.max_age is not None:
        if profile.age is None:
            if not any(item.rule == "age" for item in rules):
                rules.append(_rule("age", "unknown", "Your age is needed to determine eligibility."))
        elif profile.age > criteria.max_age:
            rules.append(_rule("age", "failed", f"Your age is above the scheme's maximum age of {criteria.max_age}."))
        else:
            rules.append(_rule("age", "passed", "Your age falls within the scheme's stated age range."))

    boolean_rules = (
        ("entrepreneur", criteria.entrepreneur, profile.entrepreneur, "The scheme requires entrepreneur status.", "The scheme is available to entrepreneurs matching your profile."),
        ("student", criteria.student, profile.student, "The scheme requires student status.", "The scheme's student requirement matches your profile."),
        ("disability", criteria.disability, profile.disability, "The scheme requires disability status.", "The scheme's disability requirement matches your profile."),
        ("rural", criteria.rural, profile.rural, "The scheme requires rural eligibility.", "The scheme's rural requirement matches your profile."),
    )
    for name, required, value, failure_message, pass_message in boolean_rules:
        if required is True:
            if value is None:
                rules.append(_rule(name, "unknown", f"Your {name} status is needed to determine eligibility."))
            else:
                rules.append(_rule(name, "passed" if value else "failed", pass_message if value else failure_message))

    if criteria.business_stages:
        if not _known(profile.business_stage):
            rules.append(_rule("business_stage", "unknown", "Your business stage is needed to determine eligibility."))
        elif profile.business_stage in criteria.business_stages:
            rules.append(_rule("business_stage", "passed", "Your business stage is supported by the scheme."))
        else:
            rules.append(_rule("business_stage", "failed", "Your business stage is not included in the scheme requirement."))

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