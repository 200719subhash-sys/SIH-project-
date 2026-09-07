"""Phase 9 tests: OCR provider abstraction."""

from __future__ import annotations

import pytest

from app.documents import NoOcrProvider, OcrUnavailableError
from app.ocr import TesseractOcrProvider, configured_ocr_provider


def test_no_ocr_provider_reports_unavailable():
    provider = NoOcrProvider()
    with pytest.raises(OcrUnavailableError, match="no OCR engine"):
        provider.extract_text(b"fake-image")


def test_configured_ocr_provider_defaults_to_none(monkeypatch):
    monkeypatch.delenv("OCR_PROVIDER", raising=False)
    provider = configured_ocr_provider()
    assert isinstance(provider, NoOcrProvider)


def test_configured_ocr_provider_tesseract(monkeypatch):
    monkeypatch.setenv("OCR_PROVIDER", "tesseract")
    provider = configured_ocr_provider()
    assert isinstance(provider, TesseractOcrProvider)


def test_configured_ocr_provider_unsupported(monkeypatch):
    monkeypatch.setenv("OCR_PROVIDER", "magic-engine")
    with pytest.raises(OcrUnavailableError, match="unsupported OCR provider"):
        configured_ocr_provider()


def test_tesseract_missing_executable_reports_unavailable():
    provider = TesseractOcrProvider(executable="definitely-not-a-real-tesseract-binary")
    with pytest.raises(OcrUnavailableError, match="tesseract executable not found"):
        provider.extract_text(b"fake-image")