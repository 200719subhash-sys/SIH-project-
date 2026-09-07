from collections.abc import Iterable

from .models import SourceRecord


class SourceRegistry:
    """In-memory registry of explicitly approved source documents and metadata."""

    def __init__(self, sources: Iterable[SourceRecord] = ()) -> None:
        self._sources: dict[str, SourceRecord] = {}
        for source in sources:
            self.register(source)

    def register(self, source: SourceRecord) -> None:
        if source.source_id in self._sources:
            raise ValueError(f"duplicate source ID: {source.source_id}")
        self._sources[source.source_id] = source

    def get(self, source_id: str) -> SourceRecord | None:
        return self._sources.get(source_id)

    def all(self) -> list[SourceRecord]:
        return list(self._sources.values())

    def verified(self) -> list[SourceRecord]:
        return [source for source in self._sources.values() if source.verification_status == "verified"]

    def remove(self, source_id: str) -> None:
        self._sources.pop(source_id, None)


DEFAULT_SOURCE_REGISTRY = SourceRegistry()
