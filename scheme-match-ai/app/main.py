from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from pathlib import Path
import json, math, re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data" / "schemes.json"
app = FastAPI(title="SahayakAI — AI Scheme Matching", version="1.0.0")
app.mount("/static", StaticFiles(directory=BASE / "app" / "static"), name="static")

class Profile(BaseModel):
    name: str = ""
    state: str
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
    keywords: list[str] = []

class ChatRequest(BaseModel):
    message: str
    profile: Profile | None = None


def load_schemes():
    with open(DATA, "r", encoding="utf-8") as f:
        return json.load(f)


def income_ok(rule, income):
    if rule is None: return True
    if rule.get("max") is not None and income > rule["max"]: return False
    if rule.get("min") is not None and income < rule["min"]: return False
    return True


def hard_eligibility(s, p):
    e = s["eligibility"]
    reasons = []
    if e.get("categories") and p.social_category not in e["categories"]:
        return False, "Social category does not match"
    if not income_ok(e.get("income"), p.annual_income):
        return False, "Annual family income is outside the scheme limit"
    if e.get("states") and p.state not in e["states"]:
        return False, "State is outside the listed coverage"
    if e.get("min_age") and p.age < e["min_age"]:
        return False, "Applicant is below the minimum age"
    if e.get("max_age") and p.age > e["max_age"]:
        return False, "Applicant is above the maximum age"
    if e.get("entrepreneur") is True and not p.entrepreneur:
        return False, "Scheme is intended for entrepreneurs"
    if e.get("student") is True and not p.student:
        return False, "Student status is required"
    if e.get("disability") is True and not p.disability:
        return False, "Disability status is required"
    if e.get("rural") is True and not p.rural:
        return False, "Rural eligibility is required"
    if e.get("business_stages") and p.business_stage not in e["business_stages"]:
        return False, "Business stage does not match"
    return True, "Eligible on stated profile constraints"


def build_profile_text(p):
    return " ".join([
        p.social_category, p.state, p.sector, p.business_stage, p.business_type,
        p.education_level, "entrepreneur" if p.entrepreneur else "aspiring entrepreneur",
        "student" if p.student else "", "rural" if p.rural else "",
        "disability" if p.disability else "", " ".join(p.keywords)
    ])


def score_scheme(s, p):
    eligible, why = hard_eligibility(s, p)
    if not eligible:
        return 0.0, False, [why]
    e = s["eligibility"]
    score = 0.0
    reasons = []
    # Explicit matches dominate semantic similarity.
    if p.social_category in e.get("categories", []): score += 30; reasons.append("Your social category is explicitly covered")
    if e.get("income", {}).get("max") is not None:
        score += 20; reasons.append(f"Income falls within the ₹{e['income']['max']/100000:.1f} lakh ceiling")
    if p.state in e.get("states", []): score += 10; reasons.append("Your state is covered")
    if p.sector.lower() in [x.lower() for x in e.get("sectors", [])]: score += 15; reasons.append("Your business sector is a direct match")
    if p.business_stage in e.get("business_stages", []): score += 10; reasons.append("Your business stage is supported")
    if p.loan_required and s.get("max_assistance", 0) >= p.loan_required: score += 5; reasons.append("Requested assistance fits within the stated limit")
    if p.rural and e.get("rural") is True: score += 5; reasons.append("Rural applicants are covered")
    # semantic similarity from scheme text
    corpus = [build_profile_text(p), s["search_text"]]
    vec = TfidfVectorizer(stop_words="english")
    X = vec.fit_transform(corpus)
    sim = float(cosine_similarity(X[0:1], X[1:2])[0][0])
    score += sim * 5
    return min(99.0, round(score, 1)), True, reasons[:5]

@app.get("/")
def index():
    return FileResponse(BASE / "app" / "static" / "index.html")

@app.get("/api/health")
def health():
    return {"status":"ok", "service":"SahayakAI"}

@app.get("/api/schemes")
def schemes():
    return load_schemes()

@app.post("/api/match")
def match(p: Profile):
    schemes = load_schemes()
    results=[]
    for s in schemes:
        score, eligible, reasons = score_scheme(s,p)
        if eligible:
            results.append({"scheme":s,"match_score":score,"reasons":reasons})
    results.sort(key=lambda x: x["match_score"], reverse=True)
    return {"profile":p.model_dump(), "count":len(results), "results":results}

@app.post("/api/chat")
def chat(req: ChatRequest):
    msg=req.message.lower()
    p=req.profile
    if any(x in msg for x in ["eligible", "eligibility", "qualify"]):
        if not p: return {"reply":"Fill your applicant profile first and I can explain eligibility scheme-by-scheme."}
        m=match(p)
        if not m["results"]: return {"reply":"I could not find a scheme matching all hard eligibility filters. Try reviewing income, category, state, or business stage."}
        top=m["results"][0]
        return {"reply":f"Your strongest current match is {top['scheme']['name']} at {top['match_score']}%. " + "; ".join(top['reasons']) + "."}
    if "income" in msg:
        return {"reply":"Income is used as a hard eligibility filter where a scheme publishes a ceiling, then as an explanation factor in the match score."}
    if "how" in msg and "work" in msg:
        return {"reply":"SahayakAI first applies hard rules such as category, income, state and age. It then scores sector, business stage, assistance fit and semantic similarity, and shows the reasons instead of a black-box number."}
    return {"reply":"I can explain eligibility, income limits, matching logic, required documents, or help you compare your top schemes."}
