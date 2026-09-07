"""Controlled source ingestion and refresh lifecycle.

A successful HTTP fetch is never treated as verification.  The
lifecycle is:

    allowlisted source
    -> fetch
    -> validate/extract
    -> pending_review
    -> persist pending snapshot
    -> NOT available to verified RAG

Only an explicit operator ``verify`` action promotes a pending snapshot
to the verified snapshot and makes it available to verified RAG.
"""

from __future__ import annotations

import hashlib
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from .allowlist import SourceAllowlist
from .extract import ExtractionError, extract_html_text
from .fetcher import FetchError, Fetcher
from .models import (
    ReviewSourceRecord,
    SourceRecord,
    SourceSnapshot,
    SourceState,
)
from .rag import RagCorpus, RegisteredDocument
from .source_store import SourceStore
from .sources import SourceRegistry
from .url_safety import UrlSafetyError


class IngestionError(ValueError):
    pass


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _content_hash(normalized_text: str) -> str:
    return hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()


def _next_data_version(current: str) -> str:
    try:
        return str(int(current) + 1)
    except ValueError:
        return "1"


class IngestionService:
    def __init__(
        self,
        allowlist: SourceAllowlist,
        fetcher: Fetcher,
        store: SourceStore,
        rag_corpus: RagCorpus,
        source_registry: SourceRegistry,
        catalogue_path: Path | None = None,
    ) -> None:
        self.allowlist = allowlist
        self.fetcher = fetcher
        self.store = store
        self.rag_corpus = rag_corpus
        self.source_registry = source_registry
        self.catalogue_path = catalogue_path
        self._states: dict[str, SourceState] = {}

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------
    def load_persisted(self) -> None:
        """Load persisted source states at startup.

        Never fetches sources.  Verified snapshots are promoted to the
        RAG corpus so previously verified evidence remains available.
        """
        for state in self.store.load_all():
            self._states[state.source_id] = state
            if state.lifecycle_status == "verified" and state.verified_snapshot is not None:
                self._promote_to_rag(state)

    # ------------------------------------------------------------------
    # Conflict detection (human review only)
    # ------------------------------------------------------------------
    def _detect_conflicts(self, source_id: str) -> list[str]:
        entry = self.allowlist.get(source_id)
        if entry is None:
            return []
        flags: list[str] = []
        if self.catalogue_path is not None and self.catalogue_path.exists():
            try:
                from .catalogue import load_catalogue

                catalogue = load_catalogue(self.catalogue_path)
                scheme_ids = {scheme.id for scheme in catalogue.schemes}
                if entry.scheme_id and entry.scheme_id not in scheme_ids:
                    flags.append(f"scheme_id {entry.scheme_id!r} is not present in the current catalogue")
                if entry.scheme_id and entry.scheme_id in scheme_ids:
                    scheme = next(item for item in catalogue.schemes if item.id == entry.scheme_id)
                    if entry.title and scheme.name.lower() not in entry.title.lower() and scheme.name.lower() not in (entry.description or "").lower():
                        flags.append(f"source title/description does not obviously reference scheme {scheme.name!r}")
            except Exception:
                # Conflict detection is best-effort and for human review only.
                pass
        return flags

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------
    def ingest(self, source_id: str) -> SourceState:
        entry = self.allowlist.get(source_id)
        if entry is None:
            raise IngestionError(f"source is not allowlisted: {source_id}")

        state = self._states.get(source_id) or SourceState(source_id=source_id, allowlisted=True)
        try:
            result = self.fetcher.fetch(str(entry.url), entry.allowed_hostname)
        except (FetchError, UrlSafetyError) as exc:
            state.errors = [str(exc)]
            state.lifecycle_status = "pending_review"
            self._states[source_id] = state
            self.store.save(state)
            raise IngestionError(f"fetch failed for {source_id}: {exc}") from exc

        try:
            normalized = extract_html_text(result.body)
        except ExtractionError as exc:
            state.errors = [str(exc)]
            state.lifecycle_status = "pending_review"
            self._states[source_id] = state
            self.store.save(state)
            raise IngestionError(f"extraction failed for {source_id}: {exc}") from exc

        content_hash = _content_hash(normalized)
        snapshot = SourceSnapshot(
            content_hash=content_hash,
            normalized_text=normalized,
            fetched_at=_now_iso(),
            data_version=state.data_version,
        )
        state.pending_snapshot = snapshot
        state.lifecycle_status = "pending_review"
        state.content_hash = content_hash
        state.fetched_at = snapshot.fetched_at
        state.errors = []
        state.conflict_flags = self._detect_conflicts(source_id)
        self._states[source_id] = state
        self.store.save(state)
        return state

    # ------------------------------------------------------------------
    # Verify
    # ------------------------------------------------------------------
    def verify(self, source_id: str) -> SourceState:
        state = self._states.get(source_id)
        if state is None:
            raise IngestionError(f"no ingested state for source: {source_id}")
        if state.pending_snapshot is None:
            raise IngestionError(f"no pending snapshot to verify for source: {source_id}")

        state.verified_snapshot = state.pending_snapshot
        state.pending_snapshot = None
        state.lifecycle_status = "verified"
        state.verified_at = _now_iso()
        state.data_version = _next_data_version(state.data_version)
        if state.verified_snapshot is not None:
            state.verified_snapshot = state.verified_snapshot.model_copy(
                update={"data_version": state.data_version, "reviewed_at": state.verified_at}
            )
        self._promote_to_rag(state)
        self.store.save(state)
        return state

    def _promote_to_rag(self, state: SourceState) -> None:
        entry = self.allowlist.get(state.source_id)
        if entry is None or state.verified_snapshot is None:
            return
        verified_at = date.today()
        if state.verified_at:
            try:
                verified_at = date.fromisoformat(state.verified_at[:10])
            except ValueError:
                pass
        record = SourceRecord(
            source_id=state.source_id,
            source_name=entry.source_name,
            source_url=entry.url,
            source_type="official_webpage",
            scheme_id=entry.scheme_id,
            title=entry.title or entry.source_name,
            verification_status="verified",
            verified_at=verified_at,
            data_version=state.data_version,
        )
        if self.source_registry.get(state.source_id) is not None:
            self.source_registry.remove(state.source_id)
        self.source_registry.register(record)
        self.rag_corpus.set_document(RegisteredDocument(state.source_id, state.verified_snapshot.normalized_text))

    def _remove_from_rag(self, source_id: str) -> None:
        self.rag_corpus.remove_document(source_id)
        if self.source_registry.get(source_id) is not None:
            self.source_registry.remove(source_id)

    # ------------------------------------------------------------------
    # Reject
    # ------------------------------------------------------------------
    def reject(self, source_id: str, reason: str | None = None) -> SourceState:
        state = self._states.get(source_id)
        if state is None:
            raise IngestionError(f"no ingested state for source: {source_id}")
        state.lifecycle_status = "rejected"
        state.rejected_reason = reason
        state.pending_snapshot = None
        self._remove_from_rag(source_id)
        self.store.save(state)
        return state

    # ------------------------------------------------------------------
    # Expire
    # ------------------------------------------------------------------
    def expire(self, source_id: str) -> SourceState:
        state = self._states.get(source_id)
        if state is None:
            raise IngestionError(f"no ingested state for source: {source_id}")
        state.lifecycle_status = "expired"
        state.expired_at = _now_iso()
        state.pending_snapshot = None
        self._remove_from_rag(source_id)
        self.store.save(state)
        return state

    # ------------------------------------------------------------------
    # Supersede
    # ------------------------------------------------------------------
    def supersede(self, source_id: str, replacement_source_id: str | None = None) -> SourceState:
        state = self._states.get(source_id)
        if state is None:
            raise IngestionError(f"no ingested state for source: {source_id}")
        if replacement_source_id and not self.allowlist.is_allowlisted(replacement_source_id):
            raise IngestionError(f"replacement source is not allowlisted: {replacement_source_id}")
        state.lifecycle_status = "superseded"
        state.superseded_by = replacement_source_id
        state.pending_snapshot = None
        self._remove_from_rag(source_id)
        self.store.save(state)
        return state

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------
    def refresh(self, source_id: str) -> SourceState:
        entry = self.allowlist.get(source_id)
        if entry is None:
            raise IngestionError(f"source is not allowlisted: {source_id}")
        state = self._states.get(source_id)
        if state is None:
            # No prior state: treat as a first ingest.
            return self.ingest(source_id)

        try:
            result = self.fetcher.fetch(str(entry.url), entry.allowed_hostname)
        except (FetchError, UrlSafetyError) as exc:
            state.errors = [str(exc)]
            self._states[source_id] = state
            self.store.save(state)
            raise IngestionError(f"refresh fetch failed for {source_id}: {exc}") from exc

        try:
            normalized = extract_html_text(result.body)
        except ExtractionError as exc:
            state.errors = [str(exc)]
            self._states[source_id] = state
            self.store.save(state)
            raise IngestionError(f"refresh extraction failed for {source_id}: {exc}") from exc

        content_hash = _content_hash(normalized)
        if state.verified_snapshot is not None and state.verified_snapshot.content_hash == content_hash:
            # Unchanged: do not create a new version.
            state.errors = []
            self._states[source_id] = state
            self.store.save(state)
            return state

        snapshot = SourceSnapshot(
            content_hash=content_hash,
            normalized_text=normalized,
            fetched_at=_now_iso(),
            data_version=state.data_version,
        )
        if state.lifecycle_status == "verified":
            # Keep the previous verified snapshot serving; store new content as pending.
            state.pending_snapshot = snapshot
            state.lifecycle_status = "pending_review"
            state.errors = []
            state.conflict_flags = self._detect_conflicts(source_id)
        else:
            # Pending/unverified: update/replace the pending snapshot.
            state.pending_snapshot = snapshot
            state.lifecycle_status = "pending_review"
            state.errors = []
            state.conflict_flags = self._detect_conflicts(source_id)
        state.content_hash = content_hash
        state.fetched_at = snapshot.fetched_at
        self._states[source_id] = state
        self.store.save(state)
        return state

    # ------------------------------------------------------------------
    # Review
    # ------------------------------------------------------------------
    def review(self) -> list[ReviewSourceRecord]:
        records: list[ReviewSourceRecord] = []
        for state in sorted(self._states.values(), key=lambda item: item.source_id):
            records.append(
                ReviewSourceRecord(
                    source_id=state.source_id,
                    lifecycle_status=state.lifecycle_status,
                    allowlisted=state.allowlisted,
                    content_hash=state.content_hash,
                    fetched_at=state.fetched_at,
                    data_version=state.data_version,
                    errors=list(state.errors),
                    conflict_flags=list(state.conflict_flags),
                    has_pending_snapshot=state.pending_snapshot is not None,
                    has_verified_snapshot=state.verified_snapshot is not None,
                    superseded_by=state.superseded_by,
                    rejected_reason=state.rejected_reason,
                    verified_at=state.verified_at,
                )
            )
        return records

    def get_state(self, source_id: str) -> SourceState | None:
        return self._states.get(source_id)