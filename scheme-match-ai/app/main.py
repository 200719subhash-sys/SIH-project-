from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, HTTPException
from pathlib import Path

from .catalogue import SchemeDataError, load_catalogue
from .matching import match_schemes, score_scheme
from .models import ChatRequest, Profile, Scheme

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data" / "schemes.json"
app = FastAPI(title="SahayakAI — AI Scheme Matching", version="1.0.0")
app.mount("/static", StaticFiles(directory=BASE / "app" / "static"), name="static")

def load_schemes() -> list[Scheme]:
    return load_catalogue(DATA).schemes

@app.get("/")
def index():
    return FileResponse(BASE / "app" / "static" / "index.html")

@app.get("/api/health")
def health():
    return {"status":"ok", "service":"SahayakAI"}

@app.get("/api/schemes")
def schemes():
    try:
        catalogue = load_catalogue(DATA)
        return [scheme.model_dump(mode="json") for scheme in catalogue.schemes]
    except SchemeDataError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.post("/api/match")
def match(p: Profile):
    try:
        schemes = load_schemes()
        results, needs_information = match_schemes(schemes, p)
        not_eligible = []
        for scheme in schemes:
            result = score_scheme(scheme, p)
            if result.status == "not_eligible":
                not_eligible.append(result)
    except SchemeDataError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="unexpected matching error") from exc
    return {
        "profile": p.model_dump(),
        "count": len(results),
        "results": [result.model_dump(mode="json") for result in results],
        "needs_information": [result.model_dump(mode="json") for result in needs_information],
        "not_eligible": [result.model_dump(mode="json") for result in not_eligible],
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
