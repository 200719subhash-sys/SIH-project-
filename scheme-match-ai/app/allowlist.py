"""Source allowlist management.

The allowlist is the *only* way a source can be fetched.  An ingestion
request must reference a ``source_id`` that exists in the allowlist; a
user can never submit an arbitrary URL to trigger fetching.

The allowlist is intentionally empty by default.  Operators must
explicitly add sources with exact URLs and hostnames.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, HttpUrl, ValidationError


class AllowlistEntry(BaseModel):
    source_id: str = Field(min_length=1)
    url: HttpUrl
    source_name: str = Field(min_length=1)
    allowed_hostname: str = Field(min_length=1)
    source_type: str = Field(default="official_webpage")
    scheme_id: str | None = None
    title: str | None = None
    description: str | None = None


class AllowlistError(ValueError):
    pass


class SourceAllowlist:
    """Exact-match allowlist of approved source URLs."""

    def __init__(self, entries: list[AllowlistEntry] | None = None) -> None:
        self._entries: dict[str, AllowlistEntry] = {}
        for entry in entries or []:
            self.add(entry)

    def add(self, entry: AllowlistEntry) -> None:
        if entry.source_id in self._entries:
            raise AllowlistError(f"duplicate allowlisted source ID: {entry.source_id}")
        self._entries[entry.source_id] = entry

    def get(self, source_id: str) -> AllowlistEntry | None:
        return self._entries.get(source_id)

    def all(self) -> list[AllowlistEntry]:
        return list(self._entries.values())

    def is_allowlisted(self, source_id: str) -> bool:
        return source_id in self._entries

    def exact_url_matches(self, source_id: str, url: str) -> bool:
        entry = self._entries.get(source_id)
        if entry is None:
            return False
        return str(entry.url).rstrip("/") == url.rstrip("/")

    def hostname_matches(self, source_id: str, hostname: str) -> bool:
        entry = self._entries.get(source_id)
        if entry is None:
            return False
        return entry.allowed_hostname.lower() == hostname.lower()


def load_allowlist(path: Path) -> SourceAllowlist:
    """Load the allowlist from a JSON file.

    The file must be a JSON object mapping ``source_id`` to entry
    metadata, or a list of entry objects.  An empty or missing file
    yields an empty allowlist.
    """
    if not path.exists():
        return SourceAllowlist()
    try:
        with path.open("r", encoding="utf-8") as file:
            raw = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise AllowlistError(f"could not load source allowlist: {exc}") from exc

    if isinstance(raw, dict):
        entries: list[dict[str, Any]] = []
        for source_id, value in raw.items():
            if not isinstance(value, dict):
                raise AllowlistError(f"allowlist entry for {source_id!r} must be an object")
            entry = dict(value)
            entry.setdefault("source_id", source_id)
            entries.append(entry)
    elif isinstance(raw, list):
        entries = [item for item in raw if isinstance(item, dict)]
    else:
        raise AllowlistError("invalid source allowlist: root must be an object or array")

    allowlist = SourceAllowlist()
    for entry in entries:
        try:
            allowlist.add(AllowlistEntry.model_validate(entry))
        except ValidationError as exc:
            raise AllowlistError(f"invalid allowlist entry: {exc}") from exc
    return allowlist