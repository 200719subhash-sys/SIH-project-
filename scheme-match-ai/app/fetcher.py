"""Controlled HTTP fetcher with SSRF protection and bounded resources.

The fetcher never treats a successful HTTP response as verification.
It only retrieves bytes from an already-allowlisted URL after URL and
DNS safety validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .url_safety import MAX_REDIRECTS, SafeUrl, UrlSafetyError, resolve_and_validate, validate_redirect_url, validate_url

CONNECT_TIMEOUT_SECONDS = 5.0
READ_TIMEOUT_SECONDS = 10.0
MAX_RESPONSE_BYTES = 2 * 1024 * 1024  # 2 MiB


class FetchError(RuntimeError):
    pass


class UnsupportedContentTypeError(FetchError):
    pass


class ResponseTooLargeError(FetchError):
    pass


class RedirectError(FetchError):
    pass


@dataclass(frozen=True)
class FetchResult:
    url: str
    status_code: int
    content_type: str
    body: bytes
    final_url: str


class Fetcher(Protocol):
    """Abstraction used by the ingestion pipeline.

    Production implementations may use httpx.  Tests must use a
    FakeFetcher and never perform live Internet requests.
    """

    def fetch(self, url: str, allowlist_hostname: str) -> FetchResult:
        ...


class HttpxFetcher:
    """Production fetcher using httpx with strict safety checks."""

    def __init__(
        self,
        connect_timeout: float = CONNECT_TIMEOUT_SECONDS,
        read_timeout: float = READ_TIMEOUT_SECONDS,
        max_bytes: int = MAX_RESPONSE_BYTES,
        max_redirects: int = MAX_REDIRECTS,
    ) -> None:
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout
        self.max_bytes = max_bytes
        self.max_redirects = max_redirects

    def _validate_target(self, url: str, allowlist_hostname: str) -> SafeUrl:
        safe = validate_url(url)
        if safe.hostname != allowlist_hostname.lower():
            raise UrlSafetyError(f"hostname {safe.hostname!r} is not the allowlisted hostname {allowlist_hostname!r}")
        resolve_and_validate(safe.hostname)
        return safe

    def fetch(self, url: str, allowlist_hostname: str) -> FetchResult:
        import httpx

        safe = self._validate_target(url, allowlist_hostname)
        current_url = safe.url
        redirects = 0

        while True:
            safe = self._validate_target(current_url, allowlist_hostname)
            try:
                with httpx.Client(
                    verify=True,
                    follow_redirects=False,
                    timeout=httpx.Timeout(self.connect_timeout, read=self.read_timeout),
                ) as client:
                    response = client.get(safe.url)
            except httpx.HTTPError as exc:
                raise FetchError(f"fetch failed: {exc}") from exc

            if response.status_code in (301, 302, 303, 307, 308):
                redirects += 1
                if redirects > self.max_redirects:
                    raise RedirectError(f"too many redirects (max {self.max_redirects})")
                location = response.headers.get("location")
                if not location:
                    raise RedirectError("redirect response missing Location header")
                # Validate the redirect target again with the same rules.
                redirect_safe = validate_redirect_url(location)
                if redirect_safe.hostname != allowlist_hostname.lower():
                    raise RedirectError("redirect target hostname is not allowlisted")
                resolve_and_validate(redirect_safe.hostname)
                current_url = redirect_safe.url
                continue

            content_type = (response.headers.get("content-type") or "").split(";")[0].strip().lower()
            if content_type and content_type not in {"text/html", "application/xhtml+xml"}:
                raise UnsupportedContentTypeError(f"unsupported content type: {content_type or 'unknown'}")

            content_length = response.headers.get("content-length")
            if content_length:
                try:
                    if int(content_length) > self.max_bytes:
                        raise ResponseTooLargeError(f"response exceeds {self.max_bytes} bytes")
                except ValueError:
                    pass

            body = bytearray()
            for chunk in response.iter_bytes():
                body.extend(chunk)
                if len(body) > self.max_bytes:
                    raise ResponseTooLargeError(f"response exceeds {self.max_bytes} bytes")

            return FetchResult(
                url=safe.url,
                status_code=response.status_code,
                content_type=content_type or "text/html",
                body=bytes(body),
                final_url=current_url,
            )