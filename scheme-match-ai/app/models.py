from typing import Any, Literal

from pydantic import BaseModel, Field, HttpUrl, model_validator


class Profile(BaseModel):
    name: str = ""
    state: str | None = None
    district: str = ""
    age: int = Field(ge=18, le=100)
    gender: str = "Any"
    social_category: str
    annual_income: float = Field(ge=0)
    entrepreneur: bool = True
    business_stage: str = "Idea"
    sector: str
    business_type: str = "Proprietorship"
    disability: bool = False
    student: bool = False
    veteran: bool = False
    rural: bool = False
    loan_required: float = Field(default=0, ge=0)
    education_level: str = "Graduate"
    keywords: list[str] = Field(default_factory=list)


class ChatRequest(BaseModel):
    message: str
    profile: Profile | None = None


class IncomeRule(BaseModel):
    min: float | None = Field(default=None, ge=0)
    max: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_range(self) -> "IncomeRule":
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError("income.min cannot be greater than income.max")
        return self


class SchemeEligibility(BaseModel):
    categories: list[str] = Field(default_factory=list)
    income: IncomeRule | None = None
    states: list[str] = Field(default_factory=list)
    min_age: int | None = Field(default=None, ge=0)
    max_age: int | None = Field(default=None, ge=0)
    entrepreneur: bool | None = None
    student: bool | None = None
    disability: bool | None = None
    rural: bool | None = None
    business_stages: list[str] = Field(default_factory=list)
    sectors: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_age_range(self) -> "SchemeEligibility":
        if self.min_age is not None and self.max_age is not None and self.min_age > self.max_age:
            raise ValueError("min_age cannot be greater than max_age")
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
    passed_rules: list[str] = Field(default_factory=list)
    failed_rules: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    score_components: ScoreComponents