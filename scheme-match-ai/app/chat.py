import re
from typing import Any, Callable

from .catalogue import load_catalogue
from .eligibility import evaluate_eligibility
from .matching import match_schemes, score_scheme
from .models import (
    ChatRequest,
    ChatResponse,
    Intent,
    IntentResult,
    MatchProfile,
    Profile,
    ProfileExtractionResult,
    Scheme,
)
from .profile_extraction import extract_profile, extracted_to_profile
from .rag import DEFAULT_RAG_CORPUS
from .retrieval import retrieve_schemes


TOOL_NAMES = {
    "match_schemes",
    "explain_eligibility",
    "get_scheme",
    "compare_schemes",
    "get_benefit_information",
    "get_documents",
    "get_application_information",
    "extract_profile",
    "retrieve_evidence",
}


def detect_intent(message: str, context: dict[str, Any] | None = None) -> IntentResult:
    text = re.sub(r"[^a-z0-9\s]", "", message.lower()).strip()
    if any(word in text for word in ("ignore previous", "system prompt", "modify the database", "execute tool")):
        return IntentResult(intent=Intent.unknown, confidence="high")
    if any(word in text for word in ("document", "paperwork", "what do i need")):
        return IntentResult(intent=Intent.documents, confidence="high")
    if any(word in text for word in ("how do i apply", "how to apply", "application process", "apply for")):
        return IntentResult(intent=Intent.application_process, confidence="high")
    if any(word in text for word in ("compare", "comparison", "compare the")):
        return IntentResult(intent=Intent.scheme_comparison, confidence="high")
    if any(word in text for word in ("why am i eligible", "why eligible", "why did this match", "why didn't i", "why did i not", "why am i not eligible", "why not eligible", "why was i rejected", "qualify")):
        return IntentResult(intent=Intent.eligibility_explanation, confidence="high")
    if any(word in text for word in ("how much", "benefit", "amount", "loan amount", "get from")):
        return IntentResult(intent=Intent.benefit_information, confidence="high")
    if any(word in text for word in ("find schemes", "which schemes", "what schemes", "schemes can i", "recommend", "scheme for me")):
        return IntentResult(intent=Intent.scheme_recommendation, confidence="high")
    if "what is a government scheme" in text or "what are government schemes" in text:
        return IntentResult(intent=Intent.general_scheme_question, confidence="high")
    try:
        extraction = extract_profile(message)
        if extraction.extracted_fields or extraction.uncertain_fields:
            return IntentResult(intent=Intent.profile_update, confidence="medium")
    except Exception:
        pass
    if text == "why" and context and context.get("selected_scheme_id"):
        return IntentResult(intent=Intent.eligibility_explanation, confidence="medium")
    if text == "what documents" and context and context.get("selected_scheme_id"):
        return IntentResult(intent=Intent.documents, confidence="medium")
    if text in {"what next", "how"} and context:
        return IntentResult(intent=Intent.clarification, confidence="medium")
    return IntentResult(intent=Intent.unknown, confidence="low")


def _merge_profile(current: Profile | None, extraction: ProfileExtractionResult) -> tuple[Profile, dict[str, Any]]:
    base = current.model_dump() if current else Profile().model_dump()
    updates = {key: value for key, value in extraction.profile.model_dump().items() if value not in (None, "", [])}
    base.update(updates)
    return Profile.model_validate(base), updates


def _scheme_id_from_context(request: ChatRequest, schemes: list[Scheme]) -> str | None:
    if request.selected_scheme_id:
        return request.selected_scheme_id
    context_id = request.conversation_context.get("selected_scheme_id")
    if context_id:
        return context_id
    text = request.message.lower()
    for scheme in schemes:
        if scheme.id.lower() in text or scheme.name.lower() in text:
            return scheme.id
    recent = request.conversation_context.get("recent_scheme_ids", [])
    return recent[0] if len(recent) == 1 else None


def _get_scheme(schemes: list[Scheme], scheme_id: str | None) -> Scheme | None:
    if not scheme_id:
        return None
    return next((scheme for scheme in schemes if scheme.id == scheme_id), None)


def _match_tool(profile: Profile) -> dict[str, Any]:
    catalogue = load_catalogue_from_app()
    retrieved, retrieval_method = retrieve_schemes(catalogue, profile=profile, top_k=len(catalogue.schemes))
    scheme_by_id = {scheme.id: scheme for scheme in catalogue.schemes}
    schemes = [scheme_by_id[item.scheme["id"]] for item in retrieved]
    results, needs = match_schemes(schemes, profile)
    not_eligible = [score_scheme(scheme, profile) for scheme in schemes]
    not_eligible = [item for item in not_eligible if item.status == "not_eligible"]
    return {
        "results": [item.model_dump(mode="json") for item in results],
        "needs_information": [item.model_dump(mode="json") for item in needs],
        "not_eligible": [item.model_dump(mode="json") for item in not_eligible],
        "retrieval": {"method": retrieval_method, "candidates": [item.model_dump(mode="json") for item in retrieved]},
    }


def load_catalogue_from_app():
    from .main import DATA

    return load_catalogue(DATA)


def _tool_registry() -> dict[str, Callable[..., Any]]:
    return {
        "match_schemes": _match_tool,
        "get_scheme": lambda scheme_id: _get_scheme(load_catalogue_from_app().schemes, scheme_id),
        "explain_eligibility": lambda profile, scheme_id: score_scheme(_get_scheme(load_catalogue_from_app().schemes, scheme_id), profile),
        "compare_schemes": _match_tool,
        "get_benefit_information": lambda scheme_id: _get_scheme(load_catalogue_from_app().schemes, scheme_id),
        "get_documents": lambda scheme_id: _get_scheme(load_catalogue_from_app().schemes, scheme_id),
        "get_application_information": lambda scheme_id: _get_scheme(load_catalogue_from_app().schemes, scheme_id),
        "extract_profile": extract_profile,
        "retrieve_evidence": DEFAULT_RAG_CORPUS.retrieve_evidence,
    }


def invoke_tool(name: str, *args: Any, **kwargs: Any) -> Any:
    tool = _tool_registry().get(name)
    if tool is None:
        raise ValueError("requested tool is not allowed")
    return tool(*args, **kwargs)


def _comparison(match: dict[str, Any]) -> str:
    rows = []
    for item in (match.get("results") or [])[:3]:
        scheme = item["scheme"]
        max_assistance = scheme.get("max_assistance")
        amount = f"₹{max_assistance:,.0f}" if max_assistance is not None else "Not available in current catalogue"
        rows.append(f"{scheme['name']}: match score {item['match_score']}, maximum assistance {amount}, verification {item['verification']['status']}.")
    return "Here are the top available matches:\n" + "\n".join(rows) if rows else "I do not have enough profile information to compare schemes yet."


def _scheme_answer(intent: Intent, scheme: Scheme | None, match: dict[str, Any] | None) -> tuple[str, str | None, list[str], bool, list[str]]:
    if scheme is None:
        return "Which scheme would you like me to check?", None, [], True, []
    if intent == Intent.documents:
        if not scheme.documents:
            return "The current catalogue does not contain verified document information for this scheme.", "get_documents", [scheme.id], False, []
        return "Documents listed in the current catalogue: " + "; ".join(document.name for document in scheme.documents), "get_documents", [scheme.id], False, []
    if intent == Intent.application_process:
        if not scheme.application_steps and not scheme.application_url and not scheme.implementing_agency:
            return "The current catalogue does not contain verified application instructions for this scheme.", "get_application_information", [scheme.id], False, []
        details = [step.name for step in scheme.application_steps or []]
        if scheme.application_url:
            details.append(f"Application URL: {scheme.application_url}")
        if scheme.implementing_agency:
            details.append(f"Implementing agency: {scheme.implementing_agency}")
        return "Application information in the current catalogue: " + "; ".join(details), "get_application_information", [scheme.id], False, []
    if intent == Intent.benefit_information:
        amount = f"₹{scheme.max_assistance:,.0f}" if scheme.max_assistance is not None else "Not available in current catalogue"
        qualifier = " This comes from the current unverified demo catalogue." if scheme.source.verification_status != "verified" else ""
        return f"Benefit: {scheme.benefit} Maximum assistance recorded in the catalogue: {amount}.{qualifier}", "get_benefit_information", [scheme.id], False, []
    item = next((x for x in (match or {}).get("results", []) + (match or {}).get("not_eligible", []) + (match or {}).get("needs_information", []) if x["scheme"]["id"] == scheme.id), None)
    if not item:
        return "I need a complete profile before I can explain this scheme.", "explain_eligibility", [scheme.id], True, ["age", "social_category", "annual_income", "sector"]
    if item["status"] == "eligible":
        return "You match this scheme on these deterministic checks: " + " ".join(item["why_match"][:5]), "explain_eligibility", [scheme.id], False, []
    if item["status"] == "needs_information":
        return "I need more information to determine this scheme: " + " ".join(value["message"] for value in item["missing_information"]), "explain_eligibility", [scheme.id], True, [value["field"] for value in item["missing_information"]]
    return "You are not eligible under the currently recorded rules because: " + " ".join(item["why_not_eligible"]), "explain_eligibility", [scheme.id], False, []


def generate_response(intent: Intent, result: dict[str, Any]) -> str:
    """Grounded response boundary: only formats supplied deterministic facts."""
    return result["answer"]


def orchestrate_chat(request: ChatRequest) -> ChatResponse:
    schemes = load_catalogue_from_app().schemes
    intent_result = detect_intent(request.message, request.conversation_context)
    extraction = extract_profile(request.message)
    profile, updates = _merge_profile(request.profile, extraction)
    context = dict(request.conversation_context)
    context["last_intent"] = intent_result.intent.value
    tool_used = None
    scheme_ids: list[str] = []
    missing: list[str] = []
    needs_clarification = False
    selected_id = _scheme_id_from_context(request, schemes)
    match: dict[str, Any] | None = None
    evidence = []

    if intent_result.intent == Intent.profile_update:
        answer = "I updated your profile with the facts you explicitly provided."
        if extraction.missing_fields:
            answer += " To narrow down schemes, the most useful missing fields are: " + ", ".join(extraction.missing_fields) + "."
        needs_clarification = extraction.needs_clarification
        missing = list(dict.fromkeys(extraction.missing_fields + extraction.uncertain_fields))
        tool_used = "extract_profile"
    elif intent_result.intent == Intent.scheme_recommendation:
        if profile.sector is None:
            answer = "To find relevant schemes, what type of business or sector are you planning?"
            needs_clarification, missing, tool_used = True, ["sector"], "match_schemes"
        else:
            match = invoke_tool("match_schemes", profile)
            tool_used = "match_schemes"
            scheme_ids = [item["scheme"]["id"] for item in match["results"][:3]]
            answer = _comparison(match)
    elif intent_result.intent == Intent.scheme_comparison:
        match = invoke_tool("match_schemes", profile) if profile.sector else None
        tool_used = "compare_schemes"
        answer = _comparison(match or {})
        if not match:
            needs_clarification, missing = True, ["sector"]
    elif intent_result.intent in {Intent.eligibility_explanation, Intent.documents, Intent.application_process, Intent.benefit_information}:
        scheme = _get_scheme(schemes, selected_id)
        if scheme and profile.sector:
            match = invoke_tool("match_schemes", profile)
        answer, tool_used, scheme_ids, needs_clarification, missing = _scheme_answer(intent_result.intent, scheme, match)
    elif intent_result.intent == Intent.general_scheme_question:
        answer = "A government scheme is a public programme that provides defined support to eligible people or businesses under recorded rules."
    elif intent_result.intent == Intent.unknown:
        answer = "I can help find schemes, explain eligibility, compare schemes, describe benefits, list catalogue documents, or explain application information."
        needs_clarification = True
    else:
        answer = "Which scheme would you like me to check?"
        needs_clarification = True

    if intent_result.intent in {Intent.benefit_information, Intent.documents, Intent.application_process, Intent.general_scheme_question, Intent.eligibility_explanation}:
        evidence = DEFAULT_RAG_CORPUS.retrieve_evidence(request.message, scheme_id=selected_id, top_k=3)
        if evidence:
            answer += " Supporting verified source: " + "; ".join(item.title for item in evidence) + "."

    if scheme_ids:
        selected_id = selected_id or scheme_ids[0]
    context["selected_scheme_id"] = selected_id
    context["recent_scheme_ids"] = scheme_ids or context.get("recent_scheme_ids", [])
    return ChatResponse(
        reply=generate_response(intent_result.intent, {"answer": answer}),
        intent=intent_result.intent,
        tool_used=tool_used,
        scheme_ids=scheme_ids,
        profile_updates=updates,
        needs_clarification=needs_clarification,
        missing_information=missing,
        selected_scheme_id=selected_id,
        conversation_context=context,
        evidence=evidence,
    )
