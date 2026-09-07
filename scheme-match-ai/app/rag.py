import re
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .models import Evidence
from .sources import DEFAULT_SOURCE_REGISTRY, SourceRegistry


@dataclass(frozen=True)
class RegisteredDocument:
    source_id: str
    content: str


@dataclass(frozen=True)
class DocumentChunk:
    source_id: str
    chunk_id: str
    content: str


def normalize_document_text(content: str) -> str:
    return re.sub(r"\s+", " ", content or "").strip()


def chunk_document(document: RegisteredDocument, chunk_size: int = 800) -> list[DocumentChunk]:
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    text = normalize_document_text(document.content)
    if not text:
        return []
    words = text.split(" ")
    chunks = []
    for index in range(0, len(words), chunk_size):
        content = " ".join(words[index:index + chunk_size]).strip()
        if content:
            chunks.append(DocumentChunk(document.source_id, f"{document.source_id}-chunk-{index // chunk_size + 1}", content))
    return chunks


class RagCorpus:
    def __init__(self, source_registry: SourceRegistry, documents: list[RegisteredDocument] | None = None) -> None:
        self.source_registry = source_registry
        self.documents = documents or []

    def add_document(self, document: RegisteredDocument) -> None:
        if not self.source_registry.get(document.source_id):
            raise ValueError(f"document references unknown source: {document.source_id}")
        self.documents.append(document)

    def set_document(self, document: RegisteredDocument) -> None:
        """Replace (or add) the document for a source.

        This is the explicit operation used when a verified snapshot is
        promoted.  It never appends pending content onto a verified
        source accidentally.
        """
        if not self.source_registry.get(document.source_id):
            raise ValueError(f"document references unknown source: {document.source_id}")
        self.remove_document(document.source_id)
        self.documents.append(document)

    def remove_document(self, source_id: str) -> None:
        """Remove all documents for a source (used on reject/expire/supersede)."""
        self.documents = [document for document in self.documents if document.source_id != source_id]

    def chunks(self, verified_only: bool = True) -> list[DocumentChunk]:
        allowed = {source.source_id for source in (self.source_registry.verified() if verified_only else self.source_registry.all())}
        return [chunk for document in self.documents if document.source_id in allowed for chunk in chunk_document(document)]

    def retrieve_evidence(self, query: str, scheme_id: str | None = None, top_k: int = 5, verified_only: bool = True) -> list[Evidence]:
        if not query.strip():
            raise ValueError("query must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        chunks = self.chunks(verified_only=verified_only)
        if scheme_id:
            chunks = [chunk for chunk in chunks if self.source_registry.get(chunk.source_id).scheme_id == scheme_id]
        if not chunks:
            return []
        matrix = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit_transform([query] + [chunk.content for chunk in chunks])
        scores = cosine_similarity(matrix[0:1], matrix[1:]).ravel()
        ranked = sorted(zip(chunks, scores), key=lambda item: (-float(item[1]), item[0].chunk_id))[:top_k]
        results = []
        for chunk, score in ranked:
            source = self.source_registry.get(chunk.source_id)
            if source is None:
                continue
            results.append(Evidence(source_id=source.source_id, scheme_id=source.scheme_id, title=source.title, source_name=source.source_name, source_url=source.source_url, snippet=chunk.content, chunk_id=chunk.chunk_id, verification_status=source.verification_status, data_version=source.data_version, relevance_score=round(float(score), 6)))
        return results



DEFAULT_RAG_CORPUS = RagCorpus(DEFAULT_SOURCE_REGISTRY)
