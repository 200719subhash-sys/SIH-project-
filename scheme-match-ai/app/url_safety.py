"""URL safety and SSRF protection.

Only HTTPS URLs are accepted.  Localhost, loopback, private, link-local,
multicast, unspecified, IPv6 ULA, and IPv4-mapped private addresses are
rejected.  DNS is resolved and every resolved IP is validated before a
fetch is allowed.
"""

from __future__ import annotations

import ipaddress
import re
import socket
from dataclasses import dataclass
from urllib.parse import urlparse

MAX_REDIRECTS = 3


class UrlSafetyError(ValueError):
    pass


@dataclass(frozen=True)
class SafeUrl:
    scheme: str
    hostname: str
    port: int | None
    path: str
    url: str


def _is_private_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if isinstance(ip, ipaddress.IPv4Address):
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_unspecified or ip.is_reserved:
            return True
        # IPv4-mapped private addresses (e.g. ::ffff:10.0.0.1) are handled
        # by the IPv6 branch below via ipv4_mapped.
        return False
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_unspecified or ip.is_site_local:
            return True
        if ip.ipv4_mapped is not None and _is_private_ip(ip.ipv4_mapped):
            return True
        # IPv6 ULA (unique local address, fc00::/7)
        if int(ip) & 0xFE00_0000_0000_0000_0000_0000_0000_0000 == 0xFC00_0000_0000_0000_0000_0000_0000_0000:
            return True
        return False
    return False


def validate_url(url: str) -> SafeUrl:
    """Validate a URL for fetching.

    Raises UrlSafetyError for any unsafe or malformed URL.
    """
    if not isinstance(url, str) or not url.strip():
        raise UrlSafetyError("URL must not be empty")
    try:
        parsed = urlparse(url)
    except ValueError as exc:
        raise UrlSafetyError(f"malformed URL: {exc}") from exc

    if parsed.scheme != "https":
        raise UrlSafetyError("only HTTPS URLs are allowed")
    if not parsed.hostname:
        raise UrlSafetyError("URL must include a hostname")
    if parsed.username is not None or parsed.password is not None:
        raise UrlSafetyError("URLs must not contain credentials")

    hostname = parsed.hostname.lower()
    if hostname in {"localhost", "localhost.localdomain"}:
        raise UrlSafetyError("localhost is not allowed")
    if hostname.endswith(".localhost"):
        raise UrlSafetyError("localhost subdomains are not allowed")
    if re.fullmatch(r"\d+\.\d+\.\d+\.\d+", hostname):
        try:
            ip = ipaddress.ip_address(hostname)
        except ValueError as exc:
            raise UrlSafetyError(f"malformed IP address: {exc}") from exc
        if ip.is_loopback:
            raise UrlSafetyError("loopback addresses are not allowed")
        if _is_private_ip(ip):
            raise UrlSafetyError("private IP addresses are not allowed")
    if ":" in hostname and not hostname.startswith("["):
        # Bare IPv6 without brackets is malformed for URL purposes.
        raise UrlSafetyError("malformed IPv6 address in URL")

    return SafeUrl(
        scheme=parsed.scheme,
        hostname=hostname,
        port=parsed.port,
        path=parsed.path or "/",
        url=url,
    )


def resolve_and_validate(hostname: str) -> list[str]:
    """Resolve a hostname and validate every resolved IP address.

    Returns the list of validated IP strings.  Raises UrlSafetyError if
    any resolved address is unsafe or if resolution fails.
    """
    try:
        infos = socket.getaddrinfo(hostname, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise UrlSafetyError(f"DNS resolution failed: {exc}") from exc
    if not infos:
        raise UrlSafetyError("DNS resolution returned no addresses")

    validated: list[str] = []
    for info in infos:
        address = info[4][0]
        try:
            ip = ipaddress.ip_address(address)
        except ValueError as exc:
            raise UrlSafetyError(f"resolved address is not a valid IP: {address}") from exc
        if _is_private_ip(ip):
            raise UrlSafetyError(f"resolved address is not publicly routable: {address}")
        validated.append(address)
    return validated


def validate_redirect_url(url: str) -> SafeUrl:
    """Validate a redirect Location URL with the same rules as the initial URL."""
    return validate_url(url)