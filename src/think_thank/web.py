"""Read-only web discovery and source extraction.

Search snippets are discovery hints, never source text or evidence. ``read`` is
required to obtain a Source. No JavaScript, embedded resources, or forms run.

Security boundary: every target (including redirects and search results) must
resolve exclusively to public addresses. Proxy environment variables are ignored.
DNS is checked *before* httpx connects, but addresses are not pinned to the socket:
a malicious DNS server can rebind between validation and connection. For hostile
networks, also enforce public-only egress at the network/proxy layer. This adapter
is not a replacement for that boundary. Timeouts apply to individual I/O operations,
not a total wall-clock deadline. PDFs are parsed locally only when pypdf is installed.
"""

import hashlib
import io
import ipaddress
import math
import re
import socket
from typing import List, Optional, Tuple
from urllib.parse import parse_qs, urldefrag, urljoin, urlsplit

import httpx
from bs4 import BeautifulSoup

from .models import SearchResult, Source

MAX_DOWNLOAD_BYTES = 5 * 1024 * 1024
MAX_TEXT_CHARS = 30_000
MAX_REDIRECTS = 5
_REDIRECTS = {301, 302, 303, 307, 308}


class WebError(RuntimeError):
    """A network, response, or extraction failure safe for callers to report."""


class UnsafeURLError(WebError):
    """The requested URL violates the public-web-only policy."""


def _public_ip(value: str) -> bool:
    try:
        address = ipaddress.ip_address(value)
        if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
            address = address.ipv4_mapped
        return (address.is_global and not address.is_multicast
                and not address.is_reserved and not address.is_unspecified)
    except ValueError:
        return False


def validate_url(url: str) -> str:
    """Validate syntax and current DNS answers; return a fragment-free URL.

    Deliberately reject credentials, obfuscated hosts, control characters, and
    any hostname with even one nonpublic DNS answer. DNS rebinding caveat above.
    """
    if not isinstance(url, str) or not url or re.search(r"[\s\x00-\x1f\x7f\\]", url):
        raise UnsafeURLError("URL is empty or contains whitespace/control characters")
    try:
        parts = urlsplit(url)
        host = parts.hostname
        port = parts.port
        if (parts.scheme.lower() not in {"http", "https"} or not host
                or parts.username is not None or parts.password is not None):
            raise ValueError("scheme, host, or credentials")
        host = host.rstrip(".").lower().encode("idna").decode("ascii")
        if not host or "%" in host or port == 0:
            raise ValueError("invalid host or port")
    except (ValueError, UnicodeError) as exc:
        raise UnsafeURLError("Only absolute HTTP(S) URLs without credentials are allowed") from exc
    if (host == "localhost" or host.endswith((".localhost", ".local", ".internal", ".lan", ".home"))
            or "." not in host and ":" not in host):
        raise UnsafeURLError("Local hostnames are not allowed")
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None:
        addresses = [str(literal)]
    else:
        try:
            answers = socket.getaddrinfo(host, port or (443 if parts.scheme.lower() == "https" else 80),
                                         type=socket.SOCK_STREAM)
            addresses = [answer[4][0] for answer in answers]
        except (OSError, UnicodeError) as exc:
            raise WebError("Could not resolve the URL hostname") from exc
    if not addresses or not all(_public_ip(address) for address in addresses):
        raise UnsafeURLError("URL resolves to a nonpublic network address")
    return urldefrag(url)[0]


class WebClient:
    """Synchronous public-web adapter. Call ``close`` when finished.

    ``search_provider`` is 'duckduckgo' (unauthenticated HTML) or 'tavily'
    (requires ``tavily_api_key``). Errors raise WebError; invalid configuration
    raises ValueError. A genuine empty search returns [], not fabricated results.
    Tavily uses a search-only POST; source retrieval exclusively uses GET.
    """

    def __init__(self, search_provider: str = "duckduckgo",
                 tavily_api_key: Optional[str] = None, timeout: float = 20.0):
        if search_provider not in {"duckduckgo", "tavily"}:
            raise ValueError("search_provider must be 'duckduckgo' or 'tavily'")
        if search_provider == "tavily" and not tavily_api_key:
            raise ValueError("tavily_api_key is required for Tavily search")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive and finite")
        self.search_provider = search_provider
        self.tavily_api_key = tavily_api_key
        self._client = httpx.Client(
            timeout=timeout, trust_env=False, follow_redirects=False,
            headers={"User-Agent": "ThinkThank/0.1 (research source reader)",
                     "Accept-Encoding": "identity"},
        )

    def close(self) -> None:
        self._client.close()

    def _get(self, url: str) -> Tuple[str, bytes, str, str]:
        """GET bounded bytes, validating each redirect before making a request."""
        current = url
        for hop in range(MAX_REDIRECTS + 1):
            current = validate_url(current)
            try:
                with self._client.stream("GET", current) as response:
                    if response.status_code in _REDIRECTS:
                        location = response.headers.get("location")
                        if not location:
                            raise WebError("Redirect response has no Location header")
                        if hop == MAX_REDIRECTS:
                            raise WebError("Too many redirects")
                        current = urljoin(current, location)
                        continue
                    response.raise_for_status()
                    data = self._bounded_body(response)
                    return (str(response.url), data,
                            response.headers.get("content-type", ""), response.encoding or "utf-8")
            except httpx.HTTPError as exc:
                # Do not echo exception URLs/query strings, which may contain secrets.
                raise WebError("HTTP retrieval failed ({})".format(type(exc).__name__)) from exc
        raise WebError("Too many redirects")

    @staticmethod
    def _bounded_body(response: httpx.Response) -> bytes:
        # Identity encoding keeps even a compressed bomb outside our parser. A
        # server ignoring Accept-Encoding must be retried elsewhere, not inflated.
        if response.headers.get("content-encoding", "identity").lower().strip() not in {"", "identity"}:
            raise WebError("Compressed responses are not supported; identity encoding is required")
        length = response.headers.get("content-length")
        if length:
            try:
                declared = int(length)
            except ValueError as exc:
                raise WebError("Invalid Content-Length") from exc
            if declared < 0 or declared > MAX_DOWNLOAD_BYTES:
                raise WebError("Response exceeds the download limit or has an invalid length")
        data = bytearray()
        for chunk in response.iter_bytes(chunk_size=64 * 1024):
            if len(data) + len(chunk) > MAX_DOWNLOAD_BYTES:
                raise WebError("Response exceeds the download limit")
            data.extend(chunk)
        return bytes(data)

    def search(self, query: str, limit: int = 5) -> List[SearchResult]:
        """Return validated discovery links, never treat their snippets as evidence."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a nonempty string")
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ValueError("limit must be a positive integer")
        limit = min(limit, 20)
        if self.search_provider == "tavily":
            candidates = self._tavily(query, limit)
        else:
            url = str(httpx.URL("https://html.duckduckgo.com/html/", params={"q": query}))
            _, body, _, encoding = self._get(url)
            try:
                decoded = body.decode(encoding, errors="replace")
            except LookupError:
                decoded = body.decode("utf-8", errors="replace")
            soup = BeautifulSoup(decoded, "html.parser")
            candidates = []
            for result in soup.select(".result"):
                anchor = result.select_one("a.result__a")
                if anchor is None:
                    continue
                raw_href = anchor.get("href")
                if not isinstance(raw_href, str) or not raw_href:
                    continue
                try:
                    href = urljoin("https://html.duckduckgo.com/", raw_href)
                    parsed = urlsplit(href)
                    if parsed.hostname in {"duckduckgo.com", "html.duckduckgo.com", "www.duckduckgo.com"}:
                        href = parse_qs(parsed.query).get("uddg", [href])[0]
                except ValueError:
                    continue
                snippet = result.select_one(".result__snippet")
                candidates.append({"url": href, "title": anchor.get_text(" ", strip=True),
                                   "content": snippet.get_text(" ", strip=True) if snippet else ""})
            if not candidates and (soup.select_one(".anomaly-modal, #challenge-form")
                                   or "bots use DuckDuckGo" in soup.get_text()):
                raise WebError("DuckDuckGo returned a bot challenge, not search results")
        results: List[SearchResult] = []
        seen = set()
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            try:
                url = validate_url(candidate.get("url", ""))
            except WebError:
                continue
            if url in seen:
                continue
            seen.add(url)
            title = candidate.get("title")
            snippet = candidate.get("content", candidate.get("snippet", ""))
            results.append(SearchResult(url=url, title=title if isinstance(title, str) else url,
                                        snippet=snippet[:2000] if isinstance(snippet, str) else ""))
            if len(results) == limit:
                break
        return results

    def _tavily(self, query: str, limit: int) -> list:
        # Never forward the API key to redirects, even to another public host.
        endpoint = validate_url("https://api.tavily.com/search")
        try:
            with self._client.stream("POST", endpoint, json={
                "api_key": self.tavily_api_key, "query": query, "max_results": limit,
                "include_answer": False, "include_raw_content": False,
            }) as response:
                response.raise_for_status()
                body = self._bounded_body(response)
            import json
            payload = json.loads(body)
            if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
                raise WebError("Tavily returned an invalid search response")
            return payload["results"]
        except httpx.HTTPError as exc:
            raise WebError("Tavily search failed ({})".format(type(exc).__name__)) from exc
        except (ValueError, UnicodeError) as exc:
            raise WebError("Tavily returned invalid JSON") from exc

    def read(self, url: str, source_id: str) -> Source:
        """Retrieve HTML, plain text, or an optional PDF; hash stored text as UTF-8.

        Source.url is the final validated URL. Text is capped at 30,000 characters;
        oversized downloads, unsupported media, and empty extraction raise WebError.
        """
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError("source_id must be a nonempty string")
        final_url, body, content_type, encoding = self._get(url)
        media = content_type.split(";", 1)[0].strip().lower()
        title, author, published_at = final_url, None, None
        if media == "application/pdf":
            text, pdf_title, author = self._pdf(body)
            title = pdf_title or title
        elif media in {"text/html", "application/xhtml+xml", "text/plain", ""}:
            try:
                decoded = body.decode(encoding, errors="replace")
            except LookupError:
                decoded = body.decode("utf-8", errors="replace")
            if "\x00" in decoded:
                raise WebError("Binary content cannot be read as text")
            is_html = media in {"text/html", "application/xhtml+xml"} or (
                not media and re.search(r"<(?:!doctype\s+html|html|head|body)\b", decoded[:1024], re.I))
            if is_html:
                soup = BeautifulSoup(decoded, "html.parser")

                def metadata(*names: str) -> Optional[str]:
                    for name in names:
                        tag = soup.find("meta", attrs={"name": name}) or soup.find("meta", attrs={"property": name})
                        if tag and isinstance(tag.get("content"), str) and tag["content"].strip():
                            return tag["content"].strip()
                    return None

                title = metadata("og:title", "twitter:title") or (
                    soup.title.get_text(" ", strip=True) if soup.title else "") or title
                author = metadata("author", "article:author", "dc.creator")
                published_at = metadata("article:published_time", "datePublished", "date", "pubdate", "dc.date")
                for tag in soup.select("script, style, noscript, template, svg, nav, footer, header, form, iframe, object, embed, canvas, head, [hidden], [aria-hidden='true']"):
                    tag.decompose()
                text = soup.get_text("\n", strip=True)
            else:
                text = decoded
            text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
        else:
            raise WebError("Unsupported content type: {}".format(media))
        text = text.strip()
        if not text:
            raise WebError("The response contained no extractable text")
        truncated = len(text) > MAX_TEXT_CHARS
        text = text[:MAX_TEXT_CHARS]
        return Source(id=source_id, url=final_url, title=title, text=text, author=author,
                      published_at=published_at, truncated=truncated,
                      content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest())

    @staticmethod
    def _pdf(body: bytes) -> Tuple[str, Optional[str], Optional[str]]:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise WebError("PDF extraction requires the optional pypdf dependency") from exc
        try:
            reader = PdfReader(io.BytesIO(body))
            if reader.is_encrypted:
                raise WebError("Encrypted PDFs are not supported")
            metadata = reader.metadata
            pieces = []
            length = 0
            for page in reader.pages:
                text = page.extract_text() or ""
                pieces.append(text)
                length += len(text) + 1
                if length > MAX_TEXT_CHARS:
                    break
            return ("\n".join(pieces), str(metadata.title) if metadata and metadata.title else None,
                    str(metadata.author) if metadata and metadata.author else None)
        except WebError:
            raise
        except Exception as exc:
            raise WebError("PDF text extraction failed") from exc
