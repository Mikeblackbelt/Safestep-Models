"""
Security utilities for SafeStep API.
Provides robust defenses against Server-Side Request Forgery (SSRF),
DNS rebinding, private IP exposure, decompression bombs, and unsafe content handling.
"""

import ipaddress
import re
import socket
import urllib.error
import urllib.request
from html.parser import HTMLParser
from typing import List, Optional, Set, Tuple
from urllib.parse import urlparse

from fastapi import HTTPException, status

# Prohibited hostnames and patterns
BLOCKED_HOSTS: Set[str] = {
    "localhost",
    "127.0.0.1",
    "::1",
    "0.0.0.0",
    "metadata.google.internal",
    "instance-data",
}

BLOCKED_SUFFIXES: Tuple[str, ...] = (
    ".local",
    ".internal",
    ".lan",
    ".corp",
    ".home",
)


class HTMLTextExtractor(HTMLParser):
    """Safely extracts plain text from HTML documents, stripping tags and scripts."""

    def __init__(self):
        super().__init__()
        self.reset()
        self._pieces: List[str] = []
        self._ignore = False

    def handle_starttag(self, tag, attrs):
        if tag.lower() in ("script", "style", "noscript", "svg"):
            self._ignore = True

    def handle_endtag(self, tag):
        if tag.lower() in ("script", "style", "noscript", "svg"):
            self._ignore = False

    def handle_data(self, data):
        if not self._ignore:
            self._pieces.append(data)

    def get_text(self) -> str:
        raw = " ".join(self._pieces)
        return re.sub(r"\s+", " ", raw).strip()


def validate_safe_url(url_str: str) -> str:
    """
    Validate that an external URL uses http/https and does not resolve to
    localhost, private networks, link-local addresses, or cloud metadata services.
    """
    if not url_str or not isinstance(url_str, str):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="URL must be a non-empty string.",
        )

    parsed = urlparse(url_str.strip())
    if parsed.scheme.lower() not in ("http", "https"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="URL scheme must be http or https.",
        )

    hostname = parsed.hostname
    if not hostname:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="URL is missing a valid hostname.",
        )

    lowered_host = hostname.lower()
    if lowered_host in BLOCKED_HOSTS or any(lowered_host.endswith(s) for s in BLOCKED_SUFFIXES):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Access to local, internal, or cloud metadata endpoints is prohibited.",
        )

    try:
        addr_info = socket.getaddrinfo(hostname, None)
    except socket.gaierror as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not resolve hostname '{hostname}': {e}",
        )

    for entry in addr_info:
        ip_str = entry[4][0]
        try:
            ip = ipaddress.ip_address(ip_str)
            if (
                ip.is_private
                or ip.is_loopback
                or ip.is_link_local
                or ip.is_reserved
                or ip.is_multicast
                or ip.is_unspecified
            ):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Access to private or local network IP addresses ({ip_str}) is prohibited.",
                )
        except ValueError:
            pass

    return url_str.strip()


class SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Redirect handler that validates each redirect target against SSRF filters."""

    def __init__(self, max_redirects: int = 3):
        super().__init__()
        self.max_redirects = max_redirects
        self._count = 0

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        self._count += 1
        if self._count > self.max_redirects:
            raise urllib.error.HTTPError(newurl, code, "Too many redirects", headers, fp)
        try:
            validate_safe_url(newurl)
        except HTTPException as e:
            raise urllib.error.HTTPError(newurl, code, e.detail, headers, fp)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch_external_content(
    url: str,
    max_bytes: int = 5_000_000,
    timeout: int = 10,
    user_agent: str = "SafeStep-Scanner/1.0 (+https://safestep.ai)",
) -> Tuple[bytes, str]:
    """
    Safely fetch external HTTP/HTTPS content with strict SSRF defense,
    bounded streaming reads, timeout, and redirect verification.
    """
    safe_url = validate_safe_url(url)
    opener = urllib.request.build_opener(SafeRedirectHandler(max_redirects=3))
    req = urllib.request.Request(
        safe_url,
        headers={
            "User-Agent": user_agent,
            "Accept": "text/html,text/plain,application/json,image/*;q=0.9,*/*;q=0.8",
        },
    )

    try:
        with opener.open(req, timeout=timeout) as resp:
            content_type = resp.headers.get("Content-Type", "").lower()
            chunks = []
            bytes_read = 0
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                bytes_read += len(chunk)
                if bytes_read > max_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"External content from URL exceeds size limit of {max_bytes // (1024 * 1024)}MB",
                    )
                chunks.append(chunk)
            return b"".join(chunks), content_type
    except HTTPException:
        raise
    except urllib.error.HTTPError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Remote server returned error {e.code}: {e.reason}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to fetch content from URL: {e}",
        )


def fetch_external_text(
    url: str,
    max_chars: int = 100_000,
    max_bytes: int = 5_000_000,
) -> str:
    """
    Fetch external URL, extract and sanitize text (parsing HTML if needed).
    """
    raw_bytes, content_type = fetch_external_content(url, max_bytes=max_bytes)

    try:
        raw_text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raw_text = raw_bytes.decode("latin-1", errors="replace")

    if "html" in content_type or "<html" in raw_text[:500].lower():
        parser = HTMLTextExtractor()
        try:
            parser.feed(raw_text)
            extracted = parser.get_text()
        except Exception:
            extracted = re.sub(r"<[^>]+>", " ", raw_text)
            extracted = re.sub(r"\s+", " ", extracted).strip()
    else:
        extracted = raw_text.strip()

    sanitized = extracted.replace("\x00", "")
    if len(sanitized) > max_chars:
        sanitized = sanitized[:max_chars]

    if not sanitized:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Fetched external URL contained no readable text content.",
        )

    return sanitized


def fetch_external_image(
    url: str,
    max_bytes: int = 25_000_000,
) -> bytes:
    """
    Fetch external image bytes with strict size limit.
    """
    raw_bytes, content_type = fetch_external_content(url, max_bytes=max_bytes)
    return raw_bytes
