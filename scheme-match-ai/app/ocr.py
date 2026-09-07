"""OCR provider abstraction.

OCR is optional.  If no OCR engine/provider is configured, the
application reports that OCR is unavailable rather than pretending OCR
was performed.  No OCR engine is bundled with this project.
"""

from __future__ import annotations

import os
from typing import Protocol

from .documents import OcrProvider, OcrUnavailableError


class ConfiguredOcrProvider(Protocol):
    def extract_text(self, image_bytes: bytes) -> str:
        ...


class TesseractOcrProvider:
    """Optional provider that shells out to the ``tesseract`` CLI.

    This provider is only used when ``OCR_PROVIDER=tesseract`` is set
    and the ``tesseract`` executable is available on PATH.  It is not
    installed by this project's dependencies.
    """

    def __init__(self, executable: str = "tesseract") -> None:
        self.executable = executable

    def extract_text(self, image_bytes: bytes) -> str:
        import subprocess
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "input.png"
            image_path.write_bytes(image_bytes)
            try:
                result = subprocess.run(
                    [self.executable, str(image_path), "stdout"],
                    capture_output=True,
                    timeout=30,
                    check=False,
                )
            except FileNotFoundError as exc:
                raise OcrUnavailableError("tesseract executable not found") from exc
            except subprocess.TimeoutExpired as exc:
                raise OcrUnavailableError("tesseract timed out") from exc
            if result.returncode != 0:
                raise OcrUnavailableError("tesseract failed to process the image")
            return result.stdout.decode("utf-8", errors="replace")


def configured_ocr_provider() -> OcrProvider:
    """Return the configured OCR provider, or a NoOcrProvider fallback."""
    provider = os.getenv("OCR_PROVIDER", "").strip().lower()
    if not provider or provider == "none":
        from .documents import NoOcrProvider

        return NoOcrProvider()
    if provider == "tesseract":
        return TesseractOcrProvider()
    raise OcrUnavailableError(f"unsupported OCR provider: {provider}")