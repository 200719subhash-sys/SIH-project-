import json
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, HTTPException
from pathlib import Path

from .eligibility import hard_eligibility
from .matching import match_schemes
from .models import ChatRequest, Profile, Scheme

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data" / "schemes.json"
app = FastAPI(title="SahayakAI — AI Scheme Matching", version="1.0.0")
app.mount("/static", StaticFiles(directory=BASE / "app" / "static"), name="static")

class SchemeDataError(ValueError):
    pass


def load_schemes() -> list[Scheme]:
    try:
        with open(DATA, "r", encoding="utf-8") as f:
            raw_schemes = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise SchemeDataError(f"could not load scheme catalogue: {exc}") from exc
    if not isinstance(raw_schemes, list):
        raise SchemeDataError("scheme catalogue must contain a JSON array")
    try:
        schemes = [Scheme.model_validate(item) for item in raw_schemes]
    except Exception as exc:
        raise SchemeDataError(f"invalid scheme catalogue: {exc}") from exc
    ids = [scheme.id for scheme in schemes]
    duplicates = sorted({scheme_id for scheme_id in ids if ids.count(scheme_id) > 1})
    if duplicates:
        raise SchemeDataError(f"duplicate scheme IDs: {', '.join(duplicates)}")
    return schemes

@app.get("/")
def index():
    return FileResponse(BASE / "app" / "static" / "index.html")

@app.get("/api/health")
def health():
    return {"status":"ok", "service":"SahayakAI"}

@app.get("/api/schemes")
def schemes():
    try:
        return [scheme.model_dump(mode="json") for scheme in load_schemes()]
    except SchemeDataError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.post("/api/match")
def match(p: Profile):
    try:
        results, needs_information = match_schemes(load_schemes(), p)
    except SchemeDataError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="unexpected matching error") from exc
    return {
        "profile": p.model_dump(),
        "count": len(results),
        "results": [result.model_dump(mode="json") for result in results],
        "needs_information": [result.model_dump(mode="json") for result in needs_information],
    }

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
