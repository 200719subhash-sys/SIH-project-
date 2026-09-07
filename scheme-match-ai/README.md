# SahayakAI — AI-Driven Scheme Matching

Hackathon-ready MVP for SIH26092: **AI-Driven Scheme Matching for Marginalized Entrepreneurs**.

## What is included
- FastAPI backend with REST endpoints.
- Explainable eligibility engine (category, income, state, age, entrepreneur/student status, business stage, sector).
- Lightweight semantic matching using TF-IDF + cosine similarity — no paid API or model key required.
- Scheme catalogue stored as JSON so the team can replace demo records with verified official scheme data.
- Responsive dark dashboard with profile form, ranked matches, reason chips, official-source links and an eligibility assistant.

## Run locally
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```
Open `http://127.0.0.1:8000`.

## API
- `GET /api/health`
- `GET /api/schemes`
- `POST /api/match`
- `POST /api/chat`

## Architecture
Browser → FastAPI → Eligibility Engine → Explainable Scoring → Scheme Catalogue.

The scoring pipeline intentionally separates deterministic **eligibility** from **ranking**. Eligibility returns `eligible`, `not_eligible`, or `needs_information`, with per-rule results. A scheme that fails a published hard condition is excluded rather than given a misleading match. A scheme with an unknown required profile value is reported separately as needing information. Eligible schemes receive a **match score** from explicit profile matches plus a small semantic similarity component; this score is not a probability or confidence value.

## Tests

Run the test suite from the project directory:
```bash
pytest
```

The tests cover eligibility boundaries and requirements, structured score components and ranking, catalogue-backed API responses, and FastAPI validation errors. The backend remains the authority for deterministic eligibility; no LLM or AI service is used by this phase.

## Natural-language profile extraction

Phase 4 adds `POST /api/profile/extract` and `POST /api/match/text`. Natural-language text is converted into a nullable structured profile, normalized deterministically, and then passed to the existing rule-based matching engine. Currency expressions such as `3.5 lakh` and `5 crore` are normalized locally. Missing or ambiguous facts remain unknown.

The application uses a small provider abstraction with a safe local extractor by default, so no API key is required. `LLM_PROVIDER`, `LLM_API_KEY`, and `LLM_MODEL` may be configured through environment variables for a future provider implementation; secrets must never be placed in frontend code. Extraction is not eligibility: the backend validates the profile and deterministic rules decide eligibility, ranking, and explanations. User text is treated as data, so instructions inside it cannot override this boundary.

## Structured scheme data

The catalogue is versioned and validated before it is used. Each scheme preserves the existing prototype information and has structured eligibility and unverified demo-source metadata. Documents, application steps, coverage, agencies, and application URLs remain unknown when the catalogue does not provide them; facts are not invented. Future authoritative government data must include its source and verification metadata. Phase 2 still does not use an LLM, RAG, OCR, embeddings, or a vector database.

## Important for SIH submission
The included scheme catalogue is **prototype/demo data**, not a claim that these are the complete or current government rules. Before final submission/demo, replace `data/schemes.json` with a reviewed dataset sourced from current official ministry/agency portals, add document requirements and application links, and show a data-refresh timestamp.

## Suggested production upgrades
1. PostgreSQL + versioned scheme records.
2. OCR/document extraction for certificates and income documents.
3. multilingual NLP (Hindi + regional languages).
4. RAG over verified government circulars/FAQs with citations.
5. consent, encryption, audit logs and role-based admin access.
6. district-level implementing-agency routing.
7. eligibility confidence + “missing information” state rather than binary assumptions.
ī