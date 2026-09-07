import os
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, HTTPException, UploadFile
from pathlib import Path

from .allowlist import load_allowlist
from .catalogue import SchemeDataError, load_catalogue
from .chat import orchestrate_chat
from .documents import DocumentExtractionError, DocumentValidationError, extract_profile_from_document
from .fetcher import HttpxFetcher
from .ingestion import IngestionError, IngestionService
from .matching import match_schemes, score_scheme
from .models import ChatRequest, ChatResponse, DocumentExtractResponse, IngestRequest, MatchProfile, Profile, ProfileExtractionRequest, RagQueryRequest, RagQueryResponse, RetrievalRequest, RetrievalResponse, SourceRecord, SupersedeRequest, TextMatchResponse, Scheme
from .ocr import configured_ocr_provider
from .profile_extraction import extracted_to_profile, extract_profile
from .rag import DEFAULT_RAG_CORPUS
from .retrieval import retrieve_schemes
from .source_store import SourceStore
from .sources import DEFAULT_SOURCE_REGISTRY

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data" / "schemes.json"
ALLOWLIST_PATH = BASE / "data" / "source_allowlist.json"
SOURCES_DIR = BASE / "data" / "sources"
app = FastAPI(title="SahayakAI — AI Scheme Matching", version="1.0.0")
app.mount("/static", StaticFiles(directory=BASE / "app" / "static"), name="static")

SOURCE_ADMIN_ENABLED = os.getenv("SOURCE_ADMIN_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}

_allowlist = load_allowlist(ALLOWLIST_PATH)
_source_store = SourceStore(SOURCES_DIR)
_ingestion = IngestionService(
    allowlist=_allowlist,
    fetcher=HttpxFetcher(),
    store=_source_store,
    rag_corpus=DEFAULT_RAG_CORPUS,
    source_registry=DEFAULT_SOURCE_REGISTRY,
    catalogue_path=DATA,
)
_ingestion.load_persisted()

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


@app.get("/api/sources", response_model=list[SourceRecord])
def sources():
    return DEFAULT_SOURCE_REGISTRY.all()


@app.post("/api/rag/query", response_model=RagQueryResponse)
def rag_query(request: RagQueryRequest):
    try:
        results = DEFAULT_RAG_CORPUS.retrieve_evidence(request.query, request.scheme_id, request.top_k, verified_only=True)
        return RagQueryResponse(query=request.query, results=results, retrieval_method="lexical", verified_only=True)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


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


@app.post("/api/documents/extract", response_model=DocumentExtractResponse)
def document_extract(file: UploadFile):
    """Extract structured profile information from an uploaded document.

    The document is treated as untrusted data.  Extracted facts are
    *not* authoritative and must be confirmed by the user before being
    used for matching.  A user document can never become verified
    government evidence.
    """
    try:
        data = file.file.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="could not read uploaded document") from exc
    finally:
        file.file.close()

    try:
        document, result = extract_profile_from_document(
            data,
            file.filename or "document.txt",
            file.content_type or "application/octet-stream",
            configured_ocr_provider(),
        )
    except DocumentValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except DocumentExtractionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail="document extraction failed") from exc

    return DocumentExtractResponse(
        filename=document.metadata.filename,
        content_type=document.metadata.content_type,
        size_bytes=document.metadata.size_bytes,
        extraction_method=document.extraction_method,
        ocr_used=document.ocr_used,
        normalized_text=document.normalized_text,
        warnings=document.warnings,
        profile=result.profile,
        extracted_fields=result.extracted_fields,
        missing_fields=result.missing_fields,
        uncertain_fields=result.uncertain_fields,
        needs_clarification=result.needs_clarification,
    )


# ----------------------------------------------------------------------
# Source admin / review endpoints (Phase 8)
#
# These are local/development gating endpoints, NOT production
# authentication.  When SOURCE_ADMIN_ENABLED is false (the default),
# all mutation/review endpoints return 404.
# ----------------------------------------------------------------------
def _require_source_admin() -> None:
    if not SOURCE_ADMIN_ENABLED:
        raise HTTPException(status_code=404, detail="Source admin endpoints are disabled")


@app.post("/api/sources/ingest")
def source_ingest(request: IngestRequest):
    _require_source_admin()
    try:
        state = _ingestion.ingest(request.source_id)
        return state.model_dump(mode="json")
    except IngestionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/sources/review")
def source_review():
    _require_source_admin()
    return [record.model_dump(mode="json") for record in _ingestion.review()]


@app.post("/api/sources/{source_id}/verify")
def source_verify(source_id: str):
    _require_source_admin()
    try:
        state = _ingestion.verify(source_id)
        return state.model_dump(mode="json")
    except IngestionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/sources/{source_id}/reject")
def source_reject(source_id: str, reason: str | None = None):
    _require_source_admin()
    try:
        state = _ingestion.reject(source_id, reason)
        return state.model_dump(mode="json")
    except IngestionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/sources/{source_id}/expire")
def source_expire(source_id: str):
    _require_source_admin()
    try:
        state = _ingestion.expire(source_id)
        return state.model_dump(mode="json")
    except IngestionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/sources/{source_id}/supersede")
def source_supersede(source_id: str, request: SupersedeRequest | None = None):
    _require_source_admin()
    try:
        replacement = request.replacement_source_id if request else None
        state = _ingestion.supersede(source_id, replacement)
        return state.model_dump(mode="json")
    except IngestionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/sources/{source_id}/refresh")
def source_refresh(source_id: str):
    _require_source_admin()
    try:
        state = _ingestion.refresh(source_id)
        return state.model_dump(mode="json")
    except IngestionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
