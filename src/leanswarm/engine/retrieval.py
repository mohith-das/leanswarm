"""Fetch and normalize web sources to ground a simulation seed.

This module is used by CALLERS of the engine (CLI, web UI run manager) — the
engine itself never fetches. All network access goes through httpx with strict
timeouts, size caps, and an SSRF guard that refuses private/reserved addresses.
"""

from __future__ import annotations

import ipaddress
import socket
from html.parser import HTMLParser

import httpx

from leanswarm.engine.models import RetrievedSource

MAX_SOURCES = 6
PER_SOURCE_CHARS = 4000
TOTAL_CORPUS_CHARS = 16000
FETCH_TIMEOUT_SECONDS = 15.0
MAX_RESPONSE_BYTES = 2_000_000

_SKIP_TAGS = {"script", "style", "noscript", "svg", "head", "nav", "footer", "iframe"}


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            text = data.strip()
            if text:
                self._chunks.append(text)

    def text(self) -> str:
        return "\n".join(self._chunks)


def html_to_text(html: str) -> str:
    parser = _TextExtractor()
    try:
        parser.feed(html)
    except Exception:
        pass
    return parser.text()


class UnsafeURLError(Exception):
    pass


def assert_public_http_url(url: str) -> None:
    """Raise UnsafeURLError unless url is http(s) to a public-internet host."""
    from urllib.parse import urlparse

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise UnsafeURLError(f"unsupported scheme: {parsed.scheme or '(none)'}")
    host = parsed.hostname
    if not host:
        raise UnsafeURLError("missing host")
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise UnsafeURLError(f"cannot resolve host: {host}") from exc
    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        if (
            address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_reserved
            or address.is_multicast
            or address.is_unspecified
        ):
            raise UnsafeURLError(f"host resolves to a non-public address: {host}")


async def fetch_url(client: httpx.AsyncClient, url: str) -> RetrievedSource:
    assert_public_http_url(url)
    response = await client.get(
        url,
        timeout=FETCH_TIMEOUT_SECONDS,
        follow_redirects=True,
        headers={"User-Agent": "leanswarm-retrieval/1.0"},
    )
    response.raise_for_status()
    assert_public_http_url(str(response.url))
    body = response.content[:MAX_RESPONSE_BYTES]
    content_type = response.headers.get("content-type", "")
    raw = body.decode(response.encoding or "utf-8", errors="replace")
    text = html_to_text(raw) if "html" in content_type else raw
    text = text[:PER_SOURCE_CHARS]
    title = ""
    if "html" in content_type:
        lower = raw.lower()
        start = lower.find("<title")
        if start != -1:
            start = lower.find(">", start)
            end = lower.find("</title>", start)
            if start != -1 and end != -1:
                title = raw[start + 1 : end].strip()[:120]
    return RetrievedSource(url=url, title=title, chars=len(text), via="direct", text=text)


async def search_web(
    client: httpx.AsyncClient,
    query: str,
    credentials: dict[str, str],
    max_results: int,
) -> list[str]:
    """Return result URLs from Tavily (preferred) or Brave. Empty if no key."""
    tavily_key = credentials.get("TAVILY_API_KEY", "")
    if tavily_key:
        response = await client.post(
            "https://api.tavily.com/search",
            json={"api_key": tavily_key, "query": query, "max_results": max_results},
            timeout=FETCH_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()
        return [str(item["url"]) for item in data.get("results", []) if item.get("url")]
    brave_key = credentials.get("BRAVE_API_KEY", "")
    if brave_key:
        response = await client.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": max_results},
            headers={"X-Subscription-Token": brave_key, "Accept": "application/json"},
            timeout=FETCH_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()
        results = data.get("web", {}).get("results", [])
        return [str(item["url"]) for item in results if item.get("url")]
    return []


async def gather_sources(
    question: str,
    urls: list[str],
    credentials: dict[str, str],
    use_search: bool,
    max_sources: int = 4,
) -> tuple[list[RetrievedSource], list[str]]:
    """Fetch explicit URLs, then top up from search. Returns (sources, errors).

    Failures are collected as human-readable strings, never raised — a dead link
    must not kill a run.
    """
    max_sources = max(1, min(MAX_SOURCES, max_sources))
    sources: list[RetrievedSource] = []
    errors: list[str] = []
    async with httpx.AsyncClient() as client:
        candidate_urls = list(dict.fromkeys(urls))[:max_sources]
        if use_search and len(candidate_urls) < max_sources:
            try:
                found = await search_web(
                    client, question, credentials, max_sources - len(candidate_urls)
                )
                via_search = [u for u in found if u not in candidate_urls]
            except Exception as exc:
                via_search = []
                errors.append(f"search failed: {type(exc).__name__}")
            candidate_urls.extend(via_search[: max_sources - len(candidate_urls)])
        search_set = set(candidate_urls) - set(urls)
        budget = TOTAL_CORPUS_CHARS
        for url in candidate_urls:
            if budget <= 0:
                break
            try:
                source = await fetch_url(client, url)
            except Exception as exc:
                errors.append(f"{url}: {type(exc).__name__}")
                continue
            if url in search_set:
                source.via = "search"
            source.text = source.text[:budget]
            source.chars = len(source.text)
            budget -= source.chars
            sources.append(source)
    return sources, errors


def build_corpus(seed_document: str, sources: list[RetrievedSource]) -> str:
    if not sources:
        return seed_document
    parts = [seed_document.strip()]
    for source in sources:
        header = f"--- SOURCE: {source.title or source.url} ({source.url}) ---"
        parts.append(f"{header}\n{source.text}")
    return "\n\n".join(parts)
