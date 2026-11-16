"""Local-only web UI/API. Run exactly one server worker per data directory.

Workers and progress logs are process-local; SQLite is the snapshot authority.
A stale running snapshot is inactive and may be explicitly resumed after restart.
Provider URLs and credentials are environment-only, never browser-controlled.
"""
from collections import deque
from contextlib import ExitStack, asynccontextmanager, contextmanager
import json
import os
from pathlib import Path
import threading
import time
from typing import Literal, Optional
from urllib.parse import urlsplit, urlunsplit
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .demo import DemoModel, DemoWeb
from .engine import ResearchEngine
from .llm import ModelClient
from .models import Limits, ResearchState, utcnow
from .reporting import render_report
from .store import Store
from .web import WebClient

DEMO_EVENT_DELAY = 0.25
MAX_ACTIVE = 3
_FAILURE = 'Research worker failed. Check local provider configuration and resume to retry.'
_STATIC = Path(__file__).parent / 'static'


class WebLimits(Limits):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)


class NewSession(BaseModel):
    model_config = ConfigDict(extra='forbid')
    topic: str = Field(min_length=1, max_length=10000)
    mode: Literal['live', 'demo'] = 'live'
    model: Optional[str] = Field(default=None, min_length=1, max_length=1000)
    search: Optional[Literal['duckduckgo', 'tavily']] = None
    limits: WebLimits = Field(default_factory=WebLimits)

    @field_validator('topic', 'model')
    @classmethod
    def nonblank(cls, value):
        if value is not None:
            value = value.strip()
            if not value or any(ord(c) < 32 and c not in '\n\t' for c in value):
                raise ValueError('A nonblank text value is required.')
        return value


class ResumeSession(BaseModel):
    model_config = ConfigDict(extra='forbid')
    limits: Optional[WebLimits] = None


def _settings(model=None, search=None):
    return {
        'mode': 'live',
        'model': model if model is not None else os.getenv('THINK_THANK_MODEL', 'gpt-4.1-mini'),
        'base_url': os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1'),
        'search': search if search is not None else os.getenv('THINK_THANK_SEARCH', 'duckduckgo'),
    }


def _safe_url(value):
    try:
        url = urlsplit(value)
        if url.scheme not in ('http', 'https') or not url.hostname:
            return '(invalid configuration)'
        host = '[' + url.hostname + ']' if ':' in url.hostname else url.hostname
        if url.port:
            host += ':' + str(url.port)
        return urlunsplit((url.scheme, host, url.path, '', ''))
    except (ValueError, TypeError):
        return '(invalid configuration)'


def _public(value):
    """Defense in depth for legacy snapshots and provider error messages."""
    secrets = [os.getenv(name) for name in ('OPENAI_API_KEY', 'TAVILY_API_KEY')]

    def clean(item):
        if isinstance(item, str):
            for secret in secrets:
                if secret:
                    item = item.replace(secret, '[redacted]')
            return item
        if isinstance(item, list):
            return [clean(x) for x in item]
        if isinstance(item, dict):
            result = {}
            for key, val in item.items():
                if key == 'config' and isinstance(val, dict):
                    val = {k: v for k, v in val.items() if k in ('mode', 'model', 'search', 'base_url')}
                    if 'base_url' in val:
                        val['base_url'] = _safe_url(val['base_url'])
                result[key] = clean(val)
            return result
        return item
    return clean(value)


@contextmanager
def _store(root):
    store = Store(root)
    try:
        yield store
    finally:
        store.close()


def _load(store, session_id):
    try:
        return store.load(session_id)
    except (ValueError, FileNotFoundError):
        raise HTTPException(404, 'Session not found.') from None


def _clients(stack, config):
    if config.get('mode') == 'demo':
        model = DemoModel()
        stack.callback(model.close)
        web = DemoWeb()
    else:
        model = ModelClient(model=config['model'], base_url=config['base_url'],
                            api_key=os.getenv('OPENAI_API_KEY'))
        stack.callback(model.close)
        web = WebClient(search_provider=config['search'], tavily_api_key=os.getenv('TAVILY_API_KEY'))
    stack.callback(web.close)
    return model, web


class SessionManager:
    def __init__(self, root):
        self.root = Path(root)
        self.lock = threading.RLock()
        self.active = {}
        self.events = {}
        self.sequence = {}
        self.closed = False

    def _event(self, session_id, message):
        with self.lock:
            seq = self.sequence.get(session_id, 0) + 1
            self.sequence[session_id] = seq
            self.events.setdefault(session_id, deque(maxlen=500)).append(
                {'seq': seq, 'at': utcnow(), 'message': _public(str(message))[:4000]})

    def _capacity(self, session_id=None):
        if self.closed:
            raise HTTPException(503, 'Server is shutting down.')
        if session_id in self.active:
            raise HTTPException(409, 'Session is already active.')
        if len(self.active) >= MAX_ACTIVE:
            raise HTTPException(409, 'At most three sessions may be active.')

    def _validate(self, config):
        if config.get('mode') == 'demo':
            return
        try:
            # Constructors validate configuration without making HTTP requests.
            with ExitStack() as stack:
                _clients(stack, config)
        except Exception:
            raise HTTPException(422, 'Invalid live provider configuration. Check model, search, and environment credentials/URL.') from None

    def start(self, body=None, session_id=None, limits=None):
        with self.lock, _store(self.root) as store:
            self._capacity(session_id)
            if session_id is not None:
                state = _load(store, session_id)
                if state.status == 'completed':
                    raise HTTPException(409, 'Completed sessions cannot be resumed.')
                config = ({'mode': 'demo'} if state.config.get('mode') == 'demo' else
                          _settings(state.config.get('model'), state.config.get('search')))
                if limits is not None:
                    state.limits = limits
            else:
                config = {'mode': 'demo'} if body.mode == 'demo' else _settings(body.model, body.search)
                state = ResearchState(id=uuid.uuid4().hex[:16], topic=body.topic, limits=body.limits)
            self._validate(config)
            # Never persist credentials or the raw environment URL. Resume always
            # uses the current environment URL, not a saved/CLI-supplied endpoint.
            state.config = {k: v for k, v in config.items() if k != 'base_url'}
            state.status, state.stop_reason = 'running', ''
            store.save(state)
            stop = threading.Event()
            thread = threading.Thread(target=self._run, args=(state.id, config, stop), daemon=True,
                                      name='research-' + state.id)
            self.active[state.id] = (stop, thread)
            try:
                thread.start()
            except Exception:
                self.active.pop(state.id, None)
                state.status, state.stop_reason = 'failed', _FAILURE
                store.save(state)
                raise HTTPException(503, 'Unable to start research worker.') from None
            return {'id': state.id}

    def _run(self, session_id, config, stop):
        def progress(message):
            self._event(session_id, message)
            if config.get('mode') == 'demo':
                stop.wait(DEMO_EVENT_DELAY)
            if stop.is_set():
                raise KeyboardInterrupt

        try:
            with _store(self.root) as store, ExitStack() as stack:
                model, web = _clients(stack, config)
                engine = ResearchEngine(model, web, store, progress=progress, should_stop=stop.is_set)
                state = engine.run(store.load(session_id))
                self._event(session_id, 'Session ' + state.status + '.')
        except BaseException as exc:
            # Reload the last committed snapshot, not partially mutated state.
            # Never expose exception messages/tracebacks (which can contain keys).
            try:
                with _store(self.root) as store:
                    state = store.load(session_id)
                    state.status = 'paused' if isinstance(exc, KeyboardInterrupt) else 'failed'
                    state.stop_reason = ('Pause requested; resume this session to continue.'
                                         if state.status == 'paused' else _FAILURE)
                    if state.status == 'failed':
                        state.errors.append(_FAILURE)
                    store.save(state)
                self._event(session_id, 'Session ' + state.status + '.')
            except Exception:
                self._event(session_id, 'Unable to save worker status; last checkpoint preserved.')
        finally:
            with self.lock:
                self.active.pop(session_id, None)

    def shutdown(self):
        with self.lock:
            self.closed = True
            workers = list(self.active.values())
            for stop, _ in workers:
                stop.set()
        deadline = time.monotonic() + 2
        for _, thread in workers:
            thread.join(timeout=max(0, deadline - time.monotonic()))


class _TrustedLocalHost(TrustedHostMiddleware):
    """Starlette splits Host at the first colon; handle bracketed IPv6 safely."""
    async def __call__(self, scope, receive, send):
        if scope['type'] in ('http', 'websocket'):
            hosts = [v.decode('latin-1') for k, v in scope.get('headers', []) if k == b'host']
            if len(hosts) != 1:
                return await Response('Invalid host header', status_code=400)(scope, receive, send)
            try:
                parsed = urlsplit('http://' + hosts[0])
                valid = (parsed.hostname in self.allowed_hosts and parsed.username is None and
                         parsed.password is None and not parsed.path and not parsed.query and
                         not parsed.fragment and not any(c.isspace() for c in hosts[0]))
                parsed.port  # Validate the port too.
            except ValueError:
                valid = False
            if not valid:
                return await Response('Invalid host header', status_code=400)(scope, receive, send)
            if hosts[0].startswith('['):
                return await self.app(scope, receive, send)
        return await super().__call__(scope, receive, send)


def _origin_tuple(value):
    url = urlsplit(value)
    if (url.scheme not in ('http', 'https') or not url.hostname or url.username is not None
            or url.password is not None or url.path or url.query or url.fragment):
        raise ValueError('Invalid origin')
    return url.scheme, url.hostname.lower(), url.port or (443 if url.scheme == 'https' else 80)


def create_app(data_dir=Path('.think-thank')):
    """Construct without opening databases, spawning threads, or reading secrets."""
    manager = SessionManager(data_dir)

    @asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            manager.shutdown()

    app = FastAPI(title='Think Thank', lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.manager = manager

    @app.middleware('http')
    async def security(request, call_next):
        if request.method not in ('GET', 'HEAD', 'OPTIONS'):
            allowed = request.headers.get('x-think-thank') == '1'
            origin = request.headers.get('origin')
            if origin is not None:
                try:
                    allowed = allowed and _origin_tuple(origin) == _origin_tuple(
                        request.url.scheme + '://' + request.headers.get('host', ''))
                except ValueError:
                    allowed = False
            if not allowed:
                return JSONResponse({'detail': 'Local request header and matching origin required.'}, status_code=403)
        response = await call_next(request)
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Referrer-Policy'] = 'no-referrer'
        response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
        response.headers['Cache-Control'] = 'no-store'
        return response

    app.add_middleware(_TrustedLocalHost,
                       allowed_hosts=['localhost', '127.0.0.1', '::1', 'testserver', 'testclient'],
                       www_redirect=False)

    @app.exception_handler(RequestValidationError)
    async def invalid_input(request, exc):
        # FastAPI's default validation response echoes input; do not echo secrets.
        return JSONResponse({'detail': 'Invalid request fields or limits.'}, status_code=422)

    @app.get('/api/config')
    def config():
        values = _settings()
        return _public({'model': values['model'], 'base_url': _safe_url(values['base_url']),
                        'search': values['search'], 'has_api_key': bool(os.getenv('OPENAI_API_KEY')),
                        'has_tavily_key': bool(os.getenv('TAVILY_API_KEY')), 'defaults': Limits().model_dump()})

    @app.get('/api/sessions')
    def sessions():
        with manager.lock, _store(manager.root) as store:
            return _public([dict(row, active=row['id'] in manager.active) for row in store.list_sessions()])

    @app.get('/api/sessions/{session_id}')
    def session(session_id: str):
        with manager.lock, _store(manager.root) as store:
            state = _load(store, session_id)
            worker = manager.active.get(session_id)
            return _public(dict(state.model_dump(), active=worker is not None,
                                pause_requested=bool(worker and worker[0].is_set()),
                                events=list(manager.events.get(session_id, []))))

    @app.post('/api/sessions', status_code=201)
    def create(body: NewSession):
        return manager.start(body=body)

    @app.post('/api/sessions/{session_id}/pause')
    def pause(session_id: str):
        with manager.lock, _store(manager.root) as store:
            state = _load(store, session_id)
            worker = manager.active.get(session_id)
            if worker:
                worker[0].set()
                return {'status': 'pause_requested'}
            if state.status == 'running':
                state.status, state.stop_reason = 'paused', 'Inactive session paused; resume to continue.'
                store.save(state)
            return {'status': state.status}

    @app.post('/api/sessions/{session_id}/resume')
    def resume(session_id: str, body: ResumeSession):
        return manager.start(session_id=session_id, limits=body.limits)

    @app.get('/api/sessions/{session_id}/report')
    def report(session_id: str, format: Literal['md', 'json'] = 'md'):
        with _store(manager.root) as store:
            state = _load(store, session_id)
        if state.report is None:
            raise HTTPException(409, 'No report is available yet.')
        state = ResearchState.model_validate(_public(state.model_dump()))
        content = (render_report(state) if format == 'md' else
                   json.dumps(state.report.model_dump(), ensure_ascii=False, indent=2))
        return Response(_public(content), media_type='text/markdown' if format == 'md' else 'application/json',
                        headers={'Content-Disposition': 'attachment; filename="%s-report.%s"' % (state.id, format)})

    @app.get('/')
    def index():
        if not (_STATIC / 'index.html').is_file():
            raise HTTPException(503, 'Web UI assets are not installed.')
        return FileResponse(_STATIC / 'index.html')

    app.mount('/static', StaticFiles(directory=str(_STATIC), check_dir=False), name='static')
    return app


def serve(data_dir=Path('.think-thank'), host='127.0.0.1', port=8000):
    """Serve locally with a single process; multiple web workers are unsupported."""
    if host not in ('127.0.0.1', 'localhost', '::1'):
        raise ValueError('The web server must bind to 127.0.0.1, localhost, or ::1.')
    import uvicorn
    uvicorn.run(create_app(data_dir), host=host, port=port, workers=1, proxy_headers=False)
