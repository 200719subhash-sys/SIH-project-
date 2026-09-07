"""Controlled HTML extraction.

Source content is treated as untrusted data.  We never execute page
JavaScript and never trust instructions contained inside source
documents.  Extraction is deterministic and normalizes document text
using the same semantics as the RAG layer.
"""

from __future__ import annotations

import html
import re

from .rag import normalize_document_text

# Tags whose content is navigation/script/style noise.
_BLOCK_TAGS = {
    "script",
    "style",
    "noscript",
    "template",
    "svg",
    "iframe",
    "nav",
    "header",
    "footer",
    "aside",
    "form",
    "button",
    "input",
    "select",
    "option",
    "textarea",
    "link",
    "meta",
    "title",
    "head",
}

_HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
_PARAGRAPH_TAGS = {"p", "div", "section", "article", "li", "tr", "td", "th", "blockquote", "pre", "br", "hr"}


class ExtractionError(ValueError):
    pass


def _strip_tags(block: str) -> str:
    """Remove all HTML tags from a block, keeping text content."""
    # Remove comments first.
    block = re.sub(r"<!--.*?-->", " ", block, flags=re.DOTALL)
    # Replace block-level tags with a space separator.
    block = re.sub(r"<(?:/{0,1})(?:%s)\b[^>]*>" % "|".join(_BLOCK_TAGS), " ", block, flags=re.IGNORECASE)
    # Replace heading/paragraph tags with a space separator.
    block = re.sub(r"<(?:/{0,1})(?:%s)\b[^>]*>" % "|".join(_HEADING_TAGS | _PARAGRAPH_TAGS), " ", block, flags=re.IGNORECASE)
    # Remove any remaining tags.
    block = re.sub(r"<[^>]+>", " ", block)
    # Decode HTML entities.
    block = html.unescape(block)
    return block


def extract_html_text(raw_html: bytes | str) -> str:
    """Extract normalized text from an HTML document.

    Raises ExtractionError if the content is not usable HTML.
    """
    if isinstance(raw_html, bytes):
        try:
            text = raw_html.decode("utf-8", errors="replace")
        except Exception as exc:  # pragma: no cover - decode with replace rarely fails
            raise ExtractionError(f"could not decode HTML: {exc}") from exc
    else:
        text = raw_html

    if not text or not text.strip():
        raise ExtractionError("empty HTML content")

    lowered = text.lower()
    if "<html" not in lowered and "<body" not in lowered and "<p" not in lowered and "<div" not in lowered:
        # Not obviously HTML; reject rather than pretending.
        raise ExtractionError("content does not appear to be HTML")

    # Remove script/style blocks entirely (including their content).
    text = re.sub(r"<script\b[^>]*>.*?</script>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<style\b[^>]*>.*?</style>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<noscript\b[^>]*>.*?</noscript>", " ", text, flags=re.IGNORECASE | re.DOTALL)

    stripped = _strip_tags(text)
    normalized = normalize_document_text(stripped)
    if not normalized:
        raise ExtractionError("no extractable text found in HTML content")
    return normalized