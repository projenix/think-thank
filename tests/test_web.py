"""Offline adapter contract and security regression tests; never use live DNS/HTTP."""
import hashlib
import json
import socket
import sys

import httpx
import pytest

from think_thank import web


@pytest.fixture(autouse=True)
def public_dns(monkeypatch):
    def resolve(host, port, **kwargs):
        if host == "private.example":
            addresses = ["10.0.0.1"]
        elif host == "mixed.example":
            addresses = ["93.184.216.34", "127.0.0.1"]
        elif host == "missing.example":
            raise socket.gaierror("not found")
        else:
            addresses = ["93.184.216.34"]
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port)) for ip in addresses]
    monkeypatch.setattr(web.socket, "getaddrinfo", resolve)


@pytest.fixture
def make_client():
    clients = []

    def make(handler, **kwargs):
        client = web.WebClient(**kwargs)
        client._client.close()
        client._client = httpx.Client(transport=httpx.MockTransport(handler),
                                     trust_env=False, follow_redirects=False)
        clients.append(client)
        return client

    yield make
    for client in clients:
        client.close()


@pytest.mark.parametrize("url", [
    "file:///etc/passwd", "ftp://example.com/a", "javascript:alert(1)",
    "http://localhost/", "http://localhost./", "http://host.local/", "http://intranet/",
    "http://127.0.0.1/", "http://10.2.3.4/", "http://169.254.169.254/",
    "http://172.16.1.1/", "http://192.168.1.1/", "http://0.0.0.0/",
    "http://224.0.0.1/", "http://100.64.0.1/", "http://[::1]/",
    "http://[fc00::1]/", "http://[::ffff:127.0.0.1]/", "http://[fe80::1%25eth0]/",
    "https://user:password@example.com/", "https://user@example.com/",
    "https://private.example/", "https://mixed.example/", "https://example.com:bad/",
    "https://example.com:0/", "https://example.com\\@localhost/", "https://example.com/\n",
    "https://%31%32%37.0.0.1/", "//example.com/a", "",
])
def test_reject_unsafe_urls_before_request(url, make_client):
    def handler(request):
        pytest.fail("Unsafe URL reached HTTP transport")
    with pytest.raises(web.UnsafeURLError):
        make_client(handler).read(url, "S1")


def test_dns_failure():
    with pytest.raises(web.WebError, match="resolve"):
        web.validate_url("https://missing.example/")


def test_public_url_and_fragment():
    assert web.validate_url("https://example.com/a#section") == "https://example.com/a"
    assert web.validate_url("https://8.8.8.8/") == "https://8.8.8.8/"


@pytest.mark.parametrize("location", ["http://169.254.169.254/latest", "http://private.example/",
                                       "//localhost/admin", "file:///etc/passwd",
                                       "https://user:secret@example.com/"])
def test_redirect_ssrf_is_blocked(location, make_client):
    seen = []
    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(302, headers={"location": location})
    with pytest.raises(web.UnsafeURLError):
        make_client(handler).read("https://example.com/start", "S1")
    assert seen == ["https://example.com/start"]


def test_html_extraction_and_metadata_after_redirect(make_client):
    seen = []
    def handler(request):
        seen.append(str(request.url))
        assert request.method == "GET"
        if request.url.path == "/start":
            return httpx.Response(302, headers={"location": "/article#intro"})
        return httpx.Response(200, headers={"content-type": "text/html; charset=utf-8"}, text='''
            <html><head><title>Page title</title><meta property="og:title" content="Research &amp; results">
            <meta name="author" content="A. Scholar"><meta property="article:published_time" content="2026-01-02">
            <script>never execute()</script></head><body><nav>menu</nav><header>banner</header>
            <main><h1>Finding</h1><p>First &amp; second.</p><script>secret script</script>
            <style>.bad {}</style><div hidden>hidden data</div><iframe src="http://localhost/"></iframe>
            </main><footer>footer</footer></body></html>''')
    source = make_client(handler).read("https://example.com/start", "S42")
    assert source.id == "S42"
    assert source.url == "https://example.com/article"
    assert source.title == "Research & results"
    assert source.author == "A. Scholar"
    assert source.published_at == "2026-01-02"
    assert source.text == "Finding\nFirst & second."
    assert source.content_hash == hashlib.sha256(source.text.encode()).hexdigest()
    assert source.retrieved_at and not source.truncated
    assert len(seen) == 2


def test_plain_text_truncation_hash(make_client):
    text = "é" * (web.MAX_TEXT_CHARS + 5)
    client = make_client(lambda request: httpx.Response(200, text=text,
                        headers={"content-type": "text/plain; charset=utf-8"}))
    source = client.read("https://example.com/a", "S1")
    assert source.text == text[:web.MAX_TEXT_CHARS]
    assert source.truncated
    assert source.content_hash == hashlib.sha256(source.text.encode("utf-8")).hexdigest()


@pytest.mark.parametrize("response, message", [
    (httpx.Response(200, content=b"x", headers={"content-length": str(web.MAX_DOWNLOAD_BYTES + 1)}), "download limit"),
    (httpx.Response(200, content=b"x", headers={"content-length": "oops"}), "Content-Length"),
    (httpx.Response(200, content=b"x", headers={"content-type": "image/png"}), "Unsupported"),
    (httpx.Response(200, text="<html><script>x</script></html>", headers={"content-type": "text/html"}), "no extractable"),
    (httpx.Response(200, content=b"abc\x00def"), "Binary"),
    (httpx.Response(404), "HTTP retrieval failed"),
    (httpx.Response(302), "Location"),
])
def test_bad_responses(response, message, make_client):
    with pytest.raises(web.WebError, match=message):
        make_client(lambda request: response).read("https://example.com/", "S1")


def test_streamed_body_limit(make_client, monkeypatch):
    monkeypatch.setattr(web, "MAX_DOWNLOAD_BYTES", 10)
    def handler(request):
        return httpx.Response(200, stream=httpx.ByteStream(b"x" * 11))
    with pytest.raises(web.WebError, match="download limit"):
        make_client(handler).read("https://example.com/", "S1")


def test_compressed_body_rejected(make_client):
    def handler(request):
        return httpx.Response(200, headers={"content-encoding": "gzip"}, stream=httpx.ByteStream(b"unused"))
    with pytest.raises(web.WebError, match="Compressed"):
        make_client(handler).read("https://example.com/", "S1")


def test_redirect_loop_bounded(make_client):
    seen = []
    def handler(request):
        seen.append(request)
        return httpx.Response(301, headers={"location": "/loop"})
    with pytest.raises(web.WebError, match="Too many redirects"):
        make_client(handler).read("https://example.com/", "S1")
    assert len(seen) == web.MAX_REDIRECTS + 1


def test_transport_error_sanitized(make_client):
    def handler(request):
        raise httpx.ReadTimeout("sensitive URL?key=secret", request=request)
    with pytest.raises(web.WebError, match="ReadTimeout") as exc:
        make_client(handler).read("https://example.com/", "S1")
    assert "secret" not in str(exc.value)


def test_ddg_links_unwrapped_deduplicated_and_filtered(make_client):
    def handler(request):
        assert request.url.host == "html.duckduckgo.com"
        assert request.url.params["q"] == "a & b"
        return httpx.Response(200, text='''
          <div class="result"><a class="result__a" href="/l/?uddg=https%3A%2F%2Fexample.com%2Fpaper">Paper</a>
          <div class="result__snippet">Discovery only, not evidence.</div></div>
          <div class="result"><a class="result__a" href="https://example.com/paper#x">Duplicate</a></div>
          <div class="result"><a class="result__a" href="http://127.0.0.1/">Unsafe</a></div>
          <div class="result"><a class="result__a" href="https://other.example/">Other</a></div>''')
    results = make_client(handler).search("a & b")
    assert [r.url for r in results] == ["https://example.com/paper", "https://other.example/"]
    assert results[0].snippet == "Discovery only, not evidence."
    assert not hasattr(results[0], "text")


def test_ddg_challenge(make_client):
    client = make_client(lambda request: httpx.Response(200, text='<form id="challenge-form"></form>'))
    with pytest.raises(web.WebError, match="bot challenge"):
        client.search("query")


def test_tavily_search(make_client):
    def handler(request):
        assert request.method == "POST"
        payload = json.loads(request.content)
        assert payload == {"api_key": "test-key", "query": "question", "max_results": 2,
                           "include_answer": False, "include_raw_content": False}
        return httpx.Response(200, json={"results": [
            {"url": "https://example.com/", "title": "Page", "content": "Hint"},
            {"url": "http://private.example/", "title": "No"}]})
    results = make_client(handler, search_provider="tavily", tavily_api_key="test-key").search("question", 2)
    assert len(results) == 1 and results[0].snippet == "Hint"


def test_tavily_never_redirects_api_key(make_client):
    seen = []
    def handler(request):
        seen.append(request)
        return httpx.Response(307, headers={"location": "https://other.example/"})
    with pytest.raises(web.WebError, match="Tavily search failed"):
        make_client(handler, search_provider="tavily", tavily_api_key="secret").search("question")
    assert len(seen) == 1


@pytest.mark.parametrize("body", [b"not json", b'[]', b'{"results": {}}'])
def test_tavily_invalid_payload(make_client, body):
    with pytest.raises(web.WebError, match="Tavily returned"):
        make_client(lambda request: httpx.Response(200, content=body),
                    search_provider="tavily", tavily_api_key="key").search("query")


def test_missing_optional_pdf_dependency(make_client, monkeypatch):
    monkeypatch.setitem(sys.modules, "pypdf", None)
    client = make_client(lambda request: httpx.Response(200, content=b"%PDF-1.7",
                        headers={"content-type": "application/pdf"}))
    with pytest.raises(web.WebError, match="optional pypdf"):
        client.read("https://example.com/paper.pdf", "S1")


def test_configuration_and_close(make_client):
    with pytest.raises(ValueError):
        web.WebClient(search_provider="other")
    with pytest.raises(ValueError):
        web.WebClient(search_provider="tavily")
    with pytest.raises(ValueError):
        web.WebClient(timeout=0)
    client = make_client(lambda request: pytest.fail("unexpected network request"))
    for query, limit in [("", 5), ("query", 0), ("query", -1)]:
        with pytest.raises(ValueError):
            client.search(query, limit)
    client.close()
    assert client._client.is_closed


def test_every_redirect_is_validated(make_client):
    seen = []
    def handler(request):
        seen.append(str(request.url))
        if request.url.path == "/start":
            return httpx.Response(302, headers={"location": "https://other.example/step"})
        return httpx.Response(302, headers={"location": "http://mixed.example/private"})
    with pytest.raises(web.UnsafeURLError):
        make_client(handler).read("https://example.com/start", "S1")
    assert seen == ["https://example.com/start", "https://other.example/step"]


def test_constructor_disables_environment_and_automatic_redirects(monkeypatch):
    captured = {}
    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)
        def close(self):
            pass
    monkeypatch.setattr(web.httpx, "Client", FakeClient)
    client = web.WebClient(timeout=7)
    assert captured["trust_env"] is False
    assert captured["follow_redirects"] is False
    assert captured["timeout"] == 7
    assert captured["headers"]["Accept-Encoding"] == "identity"
    client.close()


def test_optional_pdf_extraction(make_client, monkeypatch):
    from types import SimpleNamespace
    reader = SimpleNamespace(is_encrypted=False,
                             metadata=SimpleNamespace(title="PDF title", author="PDF author"),
                             pages=[SimpleNamespace(extract_text=lambda: "PDF passage")])
    monkeypatch.setitem(sys.modules, "pypdf", SimpleNamespace(PdfReader=lambda stream: reader))
    client = make_client(lambda request: httpx.Response(200, content=b"%PDF-1.7",
                        headers={"content-type": "application/pdf"}))
    source = client.read("https://example.com/paper.pdf", "S1")
    assert (source.title, source.author, source.text) == ("PDF title", "PDF author", "PDF passage")


def test_empty_search_is_not_error(make_client):
    assert make_client(lambda request: httpx.Response(200, text="No results.")).search("query") == []
