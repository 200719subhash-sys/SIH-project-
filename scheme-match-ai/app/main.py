from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, HTTPException
from pathlib import Path

from .catalogue import SchemeDataError, load_catalogue
from .chat import orchestrate_chat
from .matching import match_schemes, score_scheme
from .models import ChatRequest, ChatResponse, MatchProfile, Profile, ProfileExtractionRequest, RetrievalRequest, RetrievalResponse, TextMatchResponse, Scheme
from .profile_extraction import extracted_to_profile, extract_profile
from .retrieval import retrieve_schemes

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

def _match_payload(p: Profile) -> dict:
    try:
        catalogue = load_catalogue(DATA)
        schemes = catalogue.schemes
        retrieved, retrieval_method = retrieve_schemes(catalogue, profile=p, top_k=len(schemes))
        scheme_by_id = {scheme.id: scheme for scheme in schemes}
        schemes = [scheme_by_id[item.scheme["id"]] for item in retrieved]
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
        "retrieval": {
            "method": retrieval_method,
            "candidates": [item.model_dump(mode="json") for item in retrieved],
        },
    }


@app.post("/api/retrieve", response_model=RetrievalResponse)
def retrieve(request: RetrievalRequest):
    try:
        catalogue = load_catalogue(DATA)
        candidates, method = retrieve_schemes(catalogue, query=request.query, profile=request.profile, top_k=request.top_k)
        return RetrievalResponse(candidates=candidates, retrieval_method=method, scheme_data_version=catalogue.scheme_data_version)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except SchemeDataError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/match")
def match(p: MatchProfile):
    return _match_payload(p)


@app.post("/api/profile/extract")
def extract_profile_endpoint(request: ProfileExtractionRequest):
    try:
        return extract_profile(request.text).model_dump(mode="json")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Natural-language profile extraction is unavailable.") from exc


@app.post("/api/match/text", response_model=TextMatchResponse)
def match_text(request: ProfileExtractionRequest):
    try:
        extraction = extract_profile(request.text)
        profile = extracted_to_profile(extraction.profile)
        if profile.sector is None:
            return TextMatchResponse(extraction=extraction, profile=extraction.profile, match=None)
        return TextMatchResponse(extraction=extraction, profile=extraction.profile, match=_match_payload(MatchProfile.model_validate(profile.model_dump())))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Natural-language profile matching is unavailable.") from exc

@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        return orchestrate_chat(req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail="The assistant is temporarily unavailable.") from exc
