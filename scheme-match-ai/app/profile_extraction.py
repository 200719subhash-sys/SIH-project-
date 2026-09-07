import re
from typing import Any

from .llm.provider import ProfileExtractionProvider, configured_profile_extraction_provider
from .models import ExtractedProfile, Profile, ProfileExtractionResult

MAX_INPUT_LENGTH = 5000
IMPORTANT_FIELDS = ("age", "state", "social_category", "annual_income", "sector")
STATES = {
    "andhra pradesh": "Andhra Pradesh",
    "delhi": "Delhi",
    "gujarat": "Gujarat",
    "karnataka": "Karnataka",
    "maharashtra": "Maharashtra",
    "tamil nadu": "Tamil Nadu",
    "telangana": "Telangana",
    "uttar pradesh": "Uttar Pradesh",
    "west bengal": "West Bengal",
}


def normalize_currency(value: str | float | int | None) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (float, int)):
        return float(value)
    match = re.fullmatch(r"\s*[₹,]?\s*(\d+(?:\.\d+)?)\s*(lakh|lac|lakhs|crore|crores)?\s*", value.lower())
    if not match:
        return None
    amount = float(match.group(1).replace(",", ""))
    unit = match.group(2)
    multiplier = 1
    if unit in {"lakh", "lac", "lakhs"}:
        multiplier = 100000
    elif unit in {"crore", "crores"}:
        multiplier = 10000000
    return amount * multiplier


def normalize_gender(value: str | None) -> str | None:
    if not value:
        return None
    normalized = value.strip().lower()
    if normalized in {"woman", "female"}:
        return "female"
    if normalized in {"man", "male"}:
        return "male"
    return value.strip() if normalized in {"non-binary", "nonbinary"} else None


def normalize_category(value: str | None) -> str | None:
    if not value:
        return None
    normalized = value.strip().lower().replace("-", " ")
    aliases = {
        "sc": "SC", "scheduled caste": "SC",
        "st": "ST", "scheduled tribe": "ST",
        "obc": "OBC", "other backward class": "OBC",
        "general": "General", "gen": "General",
        "ews": "EWS",
    }
    return aliases.get(normalized)


def normalize_state(value: str | None) -> str | None:
    if not value:
        return None
    return STATES.get(value.strip().lower())


def _amount_near(text: str, keywords: tuple[str, ...]) -> float | None:
    amount_pattern = re.compile(r"(?:₹\s*)?(\d+(?:\.\d+)?)\s*(lakh|lac|lakhs|crore|crores)\b", re.IGNORECASE)
    for match in amount_pattern.finditer(text):
        prefix = text[max(0, match.start() - 70):match.start()].lower()
        if any(keyword in prefix for keyword in keywords):
            return normalize_currency(match.group(0))
    return None


def _first_state(text: str) -> str | None:
    lowered = text.lower()
    for state, canonical in STATES.items():
        if re.search(rf"\b{re.escape(state)}\b", lowered):
            return canonical
    return None


def extract_local_facts(text: str) -> ExtractedProfile:
    """Extract only explicit facts with conservative deterministic patterns."""
    facts: dict[str, Any] = {}
    age = re.search(r"\b(?:i(?:'m| am)|age(?: is)?|aged)\s*(?:actually\s*)?(\d{1,3})\b|\b(\d{1,3})\s*(?:years?\s*old|year-old|yo)\b", text, re.IGNORECASE)
    if age:
        facts["age"] = int(age.group(1) or age.group(2))

    state = _first_state(text)
    if state:
        facts["state"] = state

    category = re.search(r"\b(SC|ST|OBC|EWS|General|scheduled caste|scheduled tribe)\b", text, re.IGNORECASE)
    if category:
        facts["social_category"] = normalize_category(category.group(1))

    gender = re.search(r"\b(woman|female|man|male)\b", text, re.IGNORECASE)
    if gender:
        facts["gender"] = normalize_gender(gender.group(1))

    income = _amount_near(text, ("income", "earning", "earn", "family"))
    if income is not None:
        facts["annual_income"] = income

    loan = _amount_near(text, ("need", "loan", "borrow", "capital", "require"))
    if loan is not None:
        facts["loan_required"] = loan

    lower = text.lower()
    business_type = None
    for candidate in ("tailoring", "retail", "manufacturing", "agriculture", "farming", "technology"):
        if re.search(rf"\b{candidate}\b", lower):
            business_type = candidate.title()
            break
    if business_type:
        facts["business_type"] = business_type
        facts["keywords"] = [business_type.lower()]
        facts["sector"] = "Services" if business_type == "Tailoring" else business_type

    starting = bool(re.search(r"\b(start|starting|launch|new business|set up|setup)\b", lower))
    running = bool(re.search(r"\b(run|running|operate|operating|existing business)\b", lower))
    if starting or running or "entrepreneur" in lower:
        facts["entrepreneur"] = True
    if starting:
        facts["business_stage"] = "Idea"
    elif running:
        facts["business_stage"] = "Existing"

    return ExtractedProfile.model_validate(facts)


def normalize_extracted_profile(extracted: ExtractedProfile | dict[str, Any]) -> ExtractedProfile:
    data = extracted.model_dump() if isinstance(extracted, ExtractedProfile) else dict(extracted)
    data["gender"] = normalize_gender(data.get("gender"))
    data["social_category"] = normalize_category(data.get("social_category"))
    data["state"] = normalize_state(data.get("state")) or data.get("state")
    if isinstance(data.get("annual_income"), str):
        data["annual_income"] = normalize_currency(data["annual_income"])
    if isinstance(data.get("loan_required"), str):
        data["loan_required"] = normalize_currency(data["loan_required"])
    return ExtractedProfile.model_validate(data)


def _missing_fields(profile: ExtractedProfile) -> list[str]:
    values = profile.model_dump()
    return [field for field in IMPORTANT_FIELDS if values.get(field) is None]


def _conflicting_fields(text: str) -> list[str]:
    ages = re.findall(r"\b(?:i(?:'m| am)|age(?: is)?|aged)\s*(?:actually\s*)?(\d{1,3})\b|\b(\d{1,3})\s*(?:years?\s*old|year-old|yo)\b", text, re.IGNORECASE)
    age_values = {int(first or second) for first, second in ages}
    amount_matches = re.finditer(r"(?:₹\s*)?(\d+(?:\.\d+)?)\s*(lakh|lac|lakhs|crore|crores)\b", text, re.IGNORECASE)
    income_amounts = set()
    for match in amount_matches:
        prefix = text[max(0, match.start() - 70):match.start()].lower()
        income_position = max(prefix.rfind(keyword) for keyword in ("income", "earning", "earn", "family"))
        loan_position = max(prefix.rfind(keyword) for keyword in ("need", "loan", "borrow", "capital", "require"))
        if income_position >= loan_position and income_position >= 0:
            income_amounts.add(normalize_currency(match.group(0)))
    conflicts = ["age"] if len(age_values) > 1 else []
    if len(income_amounts) > 1:
        conflicts.append("annual_income")
    return conflicts


def extract_profile(text: str, provider: ProfileExtractionProvider | None = None) -> ProfileExtractionResult:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must not be empty")
    if len(text) > MAX_INPUT_LENGTH:
        raise ValueError(f"text must be {MAX_INPUT_LENGTH} characters or fewer")
    provider = provider or configured_profile_extraction_provider()
    extracted = provider.extract(text)
    raw_values = extracted if isinstance(extracted, dict) else extracted.model_dump()
    normalized = normalize_extracted_profile(extracted)
    missing = _missing_fields(normalized)
    uncertain = [
        field for field, raw_value in raw_values.items()
        if raw_value not in (None, "", []) and normalized.model_dump().get(field) is None
    ]
    for field in _conflicting_fields(text):
        if field not in uncertain:
            uncertain.append(field)
        normalized = normalized.model_copy(update={field: None})
    missing = _missing_fields(normalized)
    return ProfileExtractionResult(
        profile=normalized,
        extracted_fields=[field for field, value in normalized.model_dump().items() if value not in (None, "", [])],
        missing_fields=missing,
        uncertain_fields=uncertain,
        needs_clarification=bool(missing or uncertain),
    )


def extracted_to_profile(extracted: ExtractedProfile) -> Profile:
    values = extracted.model_dump()
    return Profile.model_validate(values)
