"""Document processing foundation.

Uploaded documents are treated as untrusted data.  They are validated,
extracted to normalized text, and passed through the existing profile
extraction pipeline.  Extracted data is kept separate from authoritative
scheme facts and can never become verified government evidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .models import ExtractedProfile, ProfileExtractionResult
from .profile_extraction import extract_profile

MAX_DOCUMENT_BYTES = 2 * 1024 * 1024  # 2 MiB
MAX_FILENAME_LENGTH = 255
ALLOWED_TEXT_EXTENSIONS = {".txt", ".md", ".text"}
ALLOWED_TEXT_MIMES = {"text/plain", "text/markdown", "application/octet-stream"}
ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}
ALLOWED_IMAGE_MIMES = {
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/bmp",
    "image/tiff",
}
ALLOWED_PDF_EXTENSIONS = {".pdf"}
ALLOWED_PDF_MIMES = {"application/pdf"}


class DocumentValidationError(ValueError):
    pass


class DocumentTooLargeError(DocumentValidationError):
    pass


class UnsupportedDocumentTypeError(DocumentValidationError):
    pass


class DocumentExtractionError(ValueError):
    pass


@dataclass(frozen=True)
class DocumentMetadata:
    filename: str
    content_type: str
    size_bytes: int
    extension: str | None


@dataclass(frozen=True)
class ExtractedDocument:
    metadata: DocumentMetadata
    normalized_text: str
    extraction_method: str  # "text", "ocr", "pdf", or "unavailable"
    warnings: list[str]
    ocr_used: bool = False


class OcrProvider(Protocol):
    """Abstraction for OCR engines.

    A provider may be unavailable; callers must handle
    OcrUnavailableError and never claim OCR was performed when no OCR
    engine actually ran.
    """

    def extract_text(self, image_bytes: bytes) -> str:
        ...


class OcrUnavailableError(RuntimeError):
    pass


class NoOcrProvider:
    """Default provider used when no OCR engine is configured."""

    def extract_text(self, image_bytes: bytes) -> str:
        raise OcrUnavailableError("no OCR engine is configured")


def _safe_filename(filename: str) -> str:
    if not filename or not filename.strip():
        raise DocumentValidationError("filename must not be empty")
    if len(filename) > MAX_FILENAME_LENGTH:
        raise DocumentValidationError(f"filename exceeds {MAX_FILENAME_LENGTH} characters")
    # Reject path separators and traversal.
    if "/" in filename or "\\" in filename or filename in {".", ".."}:
        raise DocumentValidationError("filename must not contain path separators")
    if filename.startswith("."):
        raise DocumentValidationError("filename must not be a hidden file")
    return filename


def _extension(filename: str) -> str | None:
    suffix = Path(filename).suffix.lower()
    return suffix or None


def validate_document(filename: str, content_type: str, size_bytes: int) -> DocumentMetadata:
    """Validate an uploaded document's metadata.

    Raises DocumentValidationError for unsafe or unsupported input.
    """
    safe_name = _safe_filename(filename)
    if size_bytes < 0:
        raise DocumentValidationError("size must not be negative")
    if size_bytes > MAX_DOCUMENT_BYTES:
        raise DocumentTooLargeError(f"document exceeds {MAX_DOCUMENT_BYTES} bytes")

    extension = _extension(safe_name)
    mime = (content_type or "").split(";")[0].strip().lower()

    if extension in ALLOWED_TEXT_EXTENSIONS:
        if mime and mime not in ALLOWED_TEXT_MIMES:
            raise UnsupportedDocumentTypeError(f"content type {mime!r} does not match a text document")
        return DocumentMetadata(safe_name, mime or "text/plain", size_bytes, extension)

    if extension in ALLOWED_IMAGE_EXTENSIONS:
        if mime and mime not in ALLOWED_IMAGE_MIMES:
            raise UnsupportedDocumentTypeError(f"content type {mime!r} does not match an image document")
        return DocumentMetadata(safe_name, mime or "application/octet-stream", size_bytes, extension)

    if extension in ALLOWED_PDF_EXTENSIONS:
        if mime and mime not in ALLOWED_PDF_MIMES:
            raise UnsupportedDocumentTypeError(f"content type {mime!r} does not match a PDF document")
        return DocumentMetadata(safe_name, mime or "application/pdf", size_bytes, extension)

    raise UnsupportedDocumentTypeError(f"unsupported document extension: {extension or 'none'}")


def _decode_text(data: bytes) -> str:
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def extract_text_document(data: bytes) -> str:
    """Extract normalized text from a plain-text document."""
    if not data:
        raise DocumentExtractionError("document is empty")
    text = _decode_text(data)
    # Normalize whitespace deterministically.
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        raise DocumentExtractionError("no extractable text found in document")
    return normalized


def extract_document(
    data: bytes,
    filename: str,
    content_type: str,
    ocr_provider: OcrProvider | None = None,
) -> ExtractedDocument:
    """Validate and extract a document.

    Returns an ExtractedDocument with normalized text and metadata.
    Raises DocumentValidationError or DocumentExtractionError on failure.
    """
    metadata = validate_document(filename, content_type, len(data))

    if metadata.extension in ALLOWED_TEXT_EXTENSIONS:
        text = extract_text_document(data)
        return ExtractedDocument(
            metadata=metadata,
            normalized_text=text,
            extraction_method="text",
            warnings=[],
            ocr_used=False,
        )

    if metadata.extension in ALLOWED_IMAGE_EXTENSIONS:
        provider = ocr_provider or NoOcrProvider()
        try:
            text = provider.extract_text(data)
        except OcrUnavailableError as exc:
            raise DocumentExtractionError(str(exc)) from exc
        normalized = re.sub(r"\s+", " ", text).strip()
        if not normalized:
            raise DocumentExtractionError("OCR returned no extractable text")
        return ExtractedDocument(
            metadata=metadata,
            normalized_text=normalized,
            extraction_method="ocr",
            warnings=[],
            ocr_used=True,
        )

    if metadata.extension in ALLOWED_PDF_EXTENSIONS:
        # PDF extraction is not implemented without a safe dependency.
        raise DocumentExtractionError(
            "PDF extraction is not available in this build; provide a text document or an image for OCR"
        )

    raise UnsupportedDocumentTypeError(f"unsupported document extension: {metadata.extension or 'none'}")


def extract_profile_from_document(
    data: bytes,
    filename: str,
    content_type: str,
    ocr_provider: OcrProvider | None = None,
) -> tuple[ExtractedDocument, ProfileExtractionResult]:
    """Extract a document and run profile extraction on its text.

    The extracted profile is *not* authoritative.  It must be confirmed
    by the user before being used for matching.
    """
    document = extract_document(data, filename, content_type, ocr_provider)
    result = extract_profile(document.normalized_text)
    return document, result