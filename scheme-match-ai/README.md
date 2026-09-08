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
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```
Open `http://127.0.0.1:8000`.

## Small-production deployment
The application is intentionally lightweight and does not depend on external AI services. For a small-production deployment, run the API behind a reverse proxy or process manager with a production environment:

```bash
set APP_ENV=production
set LOG_LEVEL=INFO
set CORS_ALLOW_ORIGINS=http://localhost:3000,https://your-domain.example
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2
```

Recommended safeguards:
- keep `SOURCE_ADMIN_ENABLED=false` unless a human-reviewed admin layer is configured
- do not expose secrets in frontend code or logs
- keep the scheme data source reviewed before production release
- verify the service using the health and readiness endpoints before serving traffic

The app is designed to accept local trusted settings only; it does not execute arbitrary URLs or treat user content as verified government evidence.

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

## Phase 5: AI scheme copilot

Chat now routes messages through a small typed intent orchestrator: user message → intent → allowlisted application tool → deterministic result → grounded response. Supported intents include recommendations, eligibility explanations, benefits, documents, application information, comparison, profile updates, general questions, clarification, and unknown requests.

The chat layer may extract and merge explicit profile facts, but it cannot modify scheme data, execute arbitrary tools, or decide eligibility. Scheme facts come only from the validated catalogue, and missing catalogue information is reported as unavailable. The local deterministic intent fallback works without an LLM provider; `LLM output does not determine eligibility.`

## Phase 6: Semantic + hybrid retrieval

Phase 6 adds a local retrieval layer with deterministic TF-IDF lexical ranking and an optional semantic encoder interface. When no semantic encoder is available, retrieval uses lexical mode and the application remains fully functional offline. Hybrid mode combines lexical and semantic relevance with deterministic weights; retrieval relevance is not an eligibility or approval decision.

`POST /api/retrieve` returns candidate schemes and retrieval scores only. `/api/match` evaluates the full validated catalogue through the deterministic eligibility engine and adds retrieval metadata without allowing retrieval scores to override eligibility. Current records remain subject to their catalogue verification status.

## Phase 7: Grounded RAG and source evidence

Phase 7 adds a source registry, deterministic document chunking, verified-only lexical evidence retrieval, citation metadata, `POST /api/rag/query`, and `GET /api/sources`. Only documents explicitly registered in the application corpus can produce evidence. No user-supplied URL is fetched, and retrieved text is treated as untrusted content rather than instructions.

The current demo catalogue has no registered verified documents, so the default RAG corpus is empty and factual source questions report that verified information is unavailable. Synthetic documents are used only in tests and are labelled `synthetic_test_fixture`. RAG can support explanations, but it never decides eligibility; the deterministic eligibility engine remains authoritative. No real government source was verified during this phase.

## Phase 8: Controlled official-source ingestion and refresh

Phase 8 turns the Phase 7 source registry into a controlled, auditable, human-reviewed ingestion and refresh system. A successful HTTP fetch is **never** treated as verification.

### Source lifecycle

Each source moves through explicit lifecycle states:

- `pending_review` — fetched and extracted, but not yet verified
- `verified` — explicitly verified by an operator; only verified snapshots serve RAG evidence
- `rejected` — rejected by an operator; removed from RAG, metadata retained for audit
- `expired` — expired; removed from RAG, historical snapshot retained
- `superseded` — replaced by another source; removed from RAG, audit history preserved

### Source allowlist

`data/source_allowlist.json` is the only way a source can be fetched. It is **empty by default**. Each entry contains `source_id`, exact `url`, `source_name`, `allowed_hostname`, `source_type`, optional `scheme_id`, and optional `title`/`description`. Exact hostname and URL matching is used; broad wildcard domains such as `*.gov.in` are not supported.

### URL safety and SSRF protection

- HTTPS only
- Rejects localhost, loopback, private IP ranges, link-local, multicast, unspecified, IPv6 ULA, IPv4-mapped private addresses, credentials in URLs, malformed URLs, and unsupported schemes
- DNS is resolved and every resolved IP is validated before fetching
- TLS verification remains enabled
- Redirects are not followed automatically; each redirect target is re-validated (HTTPS, allowlisted hostname, SSRF checks) with a maximum of 3 redirects
- Bounded timeouts: ~5s connect, ~10s read
- Response size limited to ~2 MiB (checked via Content-Length and while streaming)
- Only HTML/XHTML accepted; PDF is rejected clearly

### Ingestion lifecycle

1. `POST /api/sources/ingest` with `{"source_id": "..."}` fetches an allowlisted source, extracts normalized text, and stores it as `pending_review`. It is **not** available to verified RAG.
2. `POST /api/sources/{source_id}/verify` promotes the pending snapshot to verified, updates `data_version`, and replaces the verified RAG document.
3. `POST /api/sources/{source_id}/reject` marks rejected and removes from RAG.
4. `POST /api/sources/{source_id}/expire` marks expired and removes from RAG.
5. `POST /api/sources/{source_id}/supersede` marks superseded (optionally with a replacement source ID) and removes from RAG.
6. `POST /api/sources/{source_id}/refresh` re-fetches. If the normalized hash is unchanged, no new version is created. If a verified source changes, the old verified snapshot keeps serving while the new content is stored as `pending_review`; only explicit verification replaces the verified RAG content.

### Source admin gating

All source admin/review endpoints are gated by `SOURCE_ADMIN_ENABLED=false` (default). When disabled, they return 404. This is **local/development gating, not production authentication**.

### Conflict detection

Ingestion detects obvious conflicts between new source metadata and the existing catalogue (e.g., a `scheme_id` that does not exist in the catalogue). Conflicts are exposed as review flags only; they never mutate `data/schemes.json`, eligibility rules, benefits, thresholds, or matching logic.

### Persistence

Source state is persisted as JSON under `data/sources/`. The application loads persisted state at startup but **never fetches sources during startup**. The application remains fully usable when the source store is empty.

## Phase 9: Document and OCR foundation

Phase 9 adds a safe document-processing foundation. Users can upload government-related documents and extract structured information, while keeping extracted data separate from authoritative scheme facts. OCR/document extraction **never** decides eligibility.

### Document pipeline

```
document
→ validation
→ text extraction / OCR abstraction
→ normalized extracted text
→ entity/profile extraction
→ confidence / missing / uncertain fields
→ user confirmation
→ deterministic eligibility engine
```

### Supported document types

- **Text documents** (`.txt`, `.md`, `.text`) — extracted deterministically
- **Images** (`.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, `.tiff`) — through an OCR provider abstraction
- **PDFs** — rejected clearly unless a safe PDF extraction dependency is added; the provider abstraction is in place

OCR is **optional**. If no OCR engine/provider is configured, the application reports that OCR is unavailable rather than pretending OCR was performed. No OCR engine is bundled with this project. `OCR_PROVIDER=tesseract` can be set to use the `tesseract` CLI if installed on the system.

### Document security

- Uploaded documents are treated as untrusted data
- Filename, MIME/content type, size, and extension/content mismatch are validated
- Bounded file size (2 MiB)
- No uploaded file is executed
- Document text is never interpreted as system instructions
- Prompt injection contained in documents is inert
- Documents are processed in memory; no permanent storage layer is used
- No access to files outside the application-approved processing area

### Extraction result

`POST /api/documents/extract` (multipart upload) returns:

- extracted fields
- missing fields
- uncertain fields
- confidence where available
- warnings
- source document metadata

Uncertain facts are never silently converted into confirmed facts. User confirmation is required before using uncertain extracted profile facts for matching.

### Document/RAG separation

A user-uploaded document **never** automatically becomes an authoritative government source. Only explicitly registered and verified official sources can become verified RAG evidence. There is no API to add a user document to the verified RAG corpus.

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