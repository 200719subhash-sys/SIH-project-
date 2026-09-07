"""JSON/file persistence for source lifecycle state.

Source state is stored under ``data/sources/`` as one JSON file per
source.  The application loads persisted state at startup but never
fetches sources during startup.  The application remains fully usable
when the source store is empty.
"""

from __future__ import annotations

import json
from pathlib import Path

from .models import SourceState


class SourceStoreError(ValueError):
    pass


class SourceStore:
    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def _path_for(self, source_id: str) -> Path:
        # Prevent path traversal: source_id must be a safe filename.
        if not source_id or source_id in {".", ".."} or "/" in source_id or "\\" in source_id:
            raise SourceStoreError(f"invalid source ID: {source_id!r}")
        return self.directory / f"{source_id}.json"

    def save(self, state: SourceState) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self._path_for(state.source_id)
        try:
            with path.open("w", encoding="utf-8") as file:
                json.dump(state.model_dump(mode="json"), file, indent=2, ensure_ascii=False)
        except OSError as exc:
            raise SourceStoreError(f"could not persist source state: {exc}") from exc

    def load(self, source_id: str) -> SourceState | None:
        path = self._path_for(source_id)
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as file:
                raw = json.load(file)
        except (OSError, json.JSONDecodeError) as exc:
            raise SourceStoreError(f"could not load source state: {exc}") from exc
        try:
            return SourceState.model_validate(raw)
        except Exception as exc:
            raise SourceStoreError(f"invalid persisted source state: {exc}") from exc

    def load_all(self) -> list[SourceState]:
        if not self.directory.exists():
            return []
        states: list[SourceState] = []
        for path in sorted(self.directory.glob("*.json")):
            source_id = path.stem
            state = self.load(source_id)
            if state is not None:
                states.append(state)
        return states

    def delete(self, source_id: str) -> None:
        path = self._path_for(source_id)
        if path.exists():
            try:
                path.unlink()
            except OSError as exc:
                raise SourceStoreError(f"could not delete source state: {exc}") from exc