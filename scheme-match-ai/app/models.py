from datetime import date
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, HttpUrl, model_validator


class Profile(BaseModel):
    name: str | None = ""
    state: str | None = None
    district: str | None = ""
    age: int | None = Field(default=None, ge=18, le=100)
    gender: str | None = "Any"
    social_category: str | None = None
    annual_income: float | None = Field(default=None, ge=0)
    entrepreneur: bool | None = True
    business_stage: str | None = "Idea"
    sector: str | None = None
    business_type: str | None = "Proprietorship"
    disability: bool | None = False
    student: bool | None = False
    veteran: bool | None = False
    rural: bool | None = False
    loan_required: float | None = Field(default=0, ge=0)
    education_level: str | None = "Graduate"
    keywords: list[str] = Field(default_factory=list)


class ChatRequest(BaseModel):
    message: str
    profile: Profile | None = None
    selected_scheme_id: str | None = None
    conversation_context: dict[str, Any] = Field(default_factory=dict)


class Intent(str, Enum):
    scheme_recommendation = "scheme_recommendation"
    eligibility_explanation = "eligibility_explanation"
    benefit_information = "benefit_information"
    documents = "documents"
    application_process = "application_process"
    scheme_comparison = "scheme_comparison"
    profile_update = "profile_update"
    general_scheme_question = "general_scheme_question"
    clarification = "clarification"
    unknown = "unknown"


class IntentResult(BaseModel):
    intent: Intent
    confidence: Literal["high", "medium", "low"]


class ChatResponse(BaseModel):
    reply: str
    intent: Intent
    tool_used: str | None = None
    scheme_ids: list[str] = Field(default_factory=list)
    profile_updates: dict[str, Any] = Field(default_factory=dict)
    needs_clarification: bool = False
    missing_information: list[str] = Field(default_factory=list)
    selected_scheme_id: str | None = None
    conversation_context: dict[str, Any] = Field(default_factory=dict)
    evidence: list["Evidence"] = Field(default_factory=list)


class MatchProfile(Profile):
    sector: str


class ProfileExtractionRequest(BaseModel):
    text: str


class ExtractedProfile(BaseModel):
    name: str | None = None
    state: str | None = None
    district: str | None = None
    age: int | None = Field(default=None, ge=18, le=100)
    gender: str | None = None
    social_category: str | None = None
    annual_income: float | None = Field(default=None, ge=0)
    entrepreneur: bool | None = None
    business_stage: str | None = None
    sector: str | None = None
    business_type: str | None = None
    disability: bool | None = None
    student: bool | None = None
    veteran: bool | None = None
    rural: bool | None = None
    loan_required: float | None = Field(default=None, ge=0)
    education_level: str | None = None
    keywords: list[str] = Field(default_factory=list)


class ProfileExtractionResult(BaseModel):
    profile: ExtractedProfile
    extracted_fields: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    uncertain_fields: list[str] = Field(default_factory=list)
    needs_clarification: bool = False


class TextMatchResponse(BaseModel):
    extraction: ProfileExtractionResult
    profile: ExtractedProfile | None = None
    match: dict[str, Any] | None = None


class RetrievalCandidate(BaseModel):
    scheme: dict[str, Any]
    lexical_score: float
    semantic_score: float = 0.0
    hybrid_score: float
    retrieval_factors: list[str] = Field(default_factory=list)


class RetrievalResponse(BaseModel):
    candidates: list[RetrievalCandidate]
    retrieval_method: Literal["hybrid", "lexical"]
    scheme_data_version: str


class RetrievalRequest(BaseModel):
    query: str = ""
    profile: Profile | None = None
    top_k: int = Field(default=5, ge=1, le=50)


VerificationStatus = Literal["unverified", "needs_review", "verified"]


class SourceRecord(BaseModel):
    source_id: str = Field(min_length=1)
    source_name: str = Field(min_length=1)
    source_url: HttpUrl | None = None
    source_type: Literal["official_webpage", "official_pdf", "official_notification", "government_document", "synthetic_test_fixture", "other"]
    scheme_id: str | None = None
    title: str = Field(min_length=1)
    verification_status: VerificationStatus
    verified_at: date | None = None
    effective_from: date | None = None
    effective_until: date | None = None
    data_version: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_source(self) -> "SourceRecord":
        if self.verification_status == "verified" and self.verified_at is None:
            raise ValueError("verified_at is required for verified sources")
        if self.effective_from and self.effective_until and self.effective_from > self.effective_until:
            raise ValueError("effective_from cannot be later than effective_until")
        return self


class Evidence(BaseModel):
    source_id: str
    scheme_id: str | None = None
    title: str
    source_name: str
    source_url: HttpUrl | None = None
    snippet: str
    chunk_id: str
    verification_status: VerificationStatus
    data_version: str
    relevance_score: float = Field(ge=0)


class RagQueryRequest(BaseModel):
    query: str = Field(min_length=1)
    scheme_id: str | None = None
    top_k: int = Field(default=5, ge=1, le=20)


class RagQueryResponse(BaseModel):
    query: str
    results: list[Evidence]
    retrieval_method: Literal["lexical"]
    verified_only: bool = True


class IncomeRule(BaseModel):
    min: float | None = Field(default=None, ge=0)
    max: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_range(self) -> "IncomeRule":
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError("income.min cannot be greater than income.max")
        return self


class SchemeEligibility(BaseModel):
    categories: list[str] | None = None
    income: IncomeRule | None = None
    states: list[str] | None = None
    districts: list[str] | None = None
    min_age: int | None = Field(default=None, ge=0)
    max_age: int | None = Field(default=None, ge=0)
    gender: list[str] | None = None
    entrepreneur: bool | None = None
    student: bool | None = None
    disability: bool | None = None
    veteran: bool | None = None
    rural: bool | None = None
    business_stages: list[str] | None = None
    sectors: list[str] | None = None
    business_types: list[str] | None = None
    education_levels: list[str] | None = None

    @model_validator(mode="after")
    def validate_age_range(self) -> "SchemeEligibility":
        if self.min_age is not None and self.max_age is not None and self.min_age > self.max_age:
            raise ValueError("min_age cannot be greater than max_age")
        return self


class DocumentRequirement(BaseModel):
    name: str = Field(min_length=1)
    required: bool | None = None
    description: str | None = None


class ApplicationStep(BaseModel):
    name: str = Field(min_length=1)
    description: str | None = None


class Coverage(BaseModel):
    nationwide: bool | None = None
    states: list[str] | None = None
    districts: list[str] | None = None


SourceType = Literal["official_government", "official_ministry", "official_portal", "other"]


class SourceMetadata(BaseModel):
    source_name: str = Field(min_length=1)
    source_url: HttpUrl | None = None
    source_type: SourceType
    verified_at: date | None = None
    effective_from: date | None = None
    effective_until: date | None = None
    data_version: str = Field(min_length=1)
    verification_status: VerificationStatus = "unverified"

    @model_validator(mode="after")
    def validate_effective_range(self) -> "SourceMetadata":
        if self.effective_from is not None and self.effective_until is not None and self.effective_from > self.effective_until:
            raise ValueError("effective_from cannot be later than effective_until")
        if self.verification_status == "verified" and self.verified_at is None:
            raise ValueError("verified_at is required when verification_status is verified")
        return self


class Scheme(BaseModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    ministry: str = Field(min_length=1)
    description: str
    benefit: str
    max_assistance: float = Field(ge=0)
    official_url: HttpUrl
    eligibility: SchemeEligibility
    search_text: str
    documents: list[DocumentRequirement] | None = None
    application_steps: list[ApplicationStep] | None = None
    application_url: HttpUrl | None = None
    implementing_agency: str | None = None
    coverage: Coverage | None = None
    source: SourceMetadata


class SchemeCatalogue(BaseModel):
    scheme_data_version: str = Field(min_length=1)
    schemes: list[Scheme]

    @model_validator(mode="after")
    def validate_unique_ids(self) -> "SchemeCatalogue":
        ids = [scheme.id for scheme in self.schemes]
        duplicates = sorted({scheme_id for scheme_id in ids if ids.count(scheme_id) > 1})
        if duplicates:
            raise ValueError(f"duplicate scheme IDs: {', '.join(duplicates)}")
        return self


RuleStatus = Literal["passed", "failed", "unknown"]
EligibilityStatus = Literal["eligible", "not_eligible", "needs_information"]


class RuleResult(BaseModel):
    rule: str
    status: RuleStatus
    message: str


class EligibilityResult(BaseModel):
    status: EligibilityStatus
    rules: list[RuleResult] = Field(default_factory=list)
    passed_rules: list[str] = Field(default_factory=list)
    failed_rules: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class ScoreComponents(BaseModel):
    category: float = 0.0
    income: float = 0.0
    state: float = 0.0
    sector: float = 0.0
    business_stage: float = 0.0
    loan_amount: float = 0.0
    rural: float = 0.0
    text_similarity: float = 0.0

    @property
    def total(self) -> float:
        return sum(self.model_dump().values())


class MatchResult(BaseModel):
    scheme: dict[str, Any]
    status: EligibilityStatus
    match_score: float
    rules: list[RuleResult] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    score_components: ScoreComponents
    why_match: list[str] = Field(default_factory=list)
    why_not_eligible: list[str] = Field(default_factory=list)
    passed_rules: list["RuleExplanation"] = Field(default_factory=list)
    failed_rules: list["RuleExplanation"] = Field(default_factory=list)
    missing_information: list["MissingInformation"] = Field(default_factory=list)
    score_breakdown: ScoreComponents
    ranking_factors: list[str] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"]
    verification: "VerificationExplanation"


class RuleExplanation(BaseModel):
    rule: str
    message: str


class MissingInformation(BaseModel):
    field: str
    message: str


class VerificationExplanation(BaseModel):
    status: VerificationStatus
    message: str