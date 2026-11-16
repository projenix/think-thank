"""Offline HTTP, worker lifecycle, and local-only security contracts."""
from concurrent.futures import ThreadPoolExecutor
import json
import socket
import threading
import time

import pytest
from fastapi.testclient import TestClient

from think_thank import server
from think_thank.models import Limits, ResearchState
from think_thank.store import Store

HEADERS = {'X-Think-Thank': '1'}


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Tests must not contact the network')
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setattr(server, 'DEMO_EVENT_DELAY', 0.005)
    for name in ('OPENAI_API_KEY', 'TAVILY_API_KEY', 'OPENAI_BASE_URL', 'THINK_THANK_MODEL', 'THINK_THANK_SEARCH'):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def client(tmp_path):
    with TestClient(server.create_app(tmp_path)) as client:
        yield client


def create(client, **kwargs):
    response = client.post('/api/sessions', headers=HEADERS,
                           json={'topic': 'A fictional research question', 'mode': 'demo', **kwargs})
    assert response.status_code == 201, response.text
    return response.json()['id']


def wait_for(client, session_id, predicate=lambda s: not s['active'], timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get('/api/sessions/' + session_id)
        assert response.status_code == 200, response.text
        state = response.json()
        if predicate(state):
            return state
        time.sleep(0.01)
    pytest.fail('Worker did not reach expected state: ' + json.dumps(state))


def save(root, state):
    store = Store(root)
    try:
        store.save(state)
    finally:
        store.close()


def test_construction_is_side_effect_free_and_defaults(tmp_path, client):
    root = tmp_path / 'unused'
    app = server.create_app(root)
    assert not root.exists()
    assert not app.state.manager.active
    config = client.get('/api/config').json()
    assert config == {'model': 'gpt-4.1-mini', 'base_url': 'https://api.openai.com/v1',
                      'search': 'duckduckgo', 'has_api_key': False, 'has_tavily_key': False,
                      'defaults': Limits().model_dump()}
    assert client.get('/api/sessions').json() == []


def test_background_demo_completion_and_downloads(client):
    session_id = create(client)
    state = wait_for(client, session_id)
    assert state['status'] == 'completed', state
    assert not state['pause_requested']
    assert len(state['sources']) == len(state['evidence']) == 4
    assert state['report']
    assert state['events']
    assert [e['seq'] for e in state['events']] == sorted({e['seq'] for e in state['events']})
    assert all(set(e) == {'seq', 'at', 'message'} for e in state['events'])
    rows = client.get('/api/sessions').json()
    assert len(rows) == 1
    assert set(rows[0]) == {'id', 'topic', 'status', 'iteration', 'updated_at', 'stop_reason', 'active'}
    assert rows[0]['active'] is False
    for format in ('md', 'json'):
        report = client.get('/api/sessions/' + session_id + '/report?format=' + format)
        assert report.status_code == 200
        assert 'attachment;' in report.headers['content-disposition']
        assert session_id in report.headers['content-disposition']
        if format == 'json':
            assert report.json() == state['report']
        else:
            assert 'fictional' in report.text
    assert client.post('/api/sessions/' + session_id + '/resume', headers=HEADERS, json={}).status_code == 409


def test_pause_resume_and_limits(client, monkeypatch):
    monkeypatch.setattr(server, 'DEMO_EVENT_DELAY', 0.1)
    session_id = create(client)
    wait_for(client, session_id, lambda s: bool(s['events']))
    response = client.post('/api/sessions/' + session_id + '/pause', headers=HEADERS, json={})
    assert response.json() == {'status': 'pause_requested'}
    state = wait_for(client, session_id)
    assert state['status'] == 'paused'
    assert not state['pause_requested']
    response = client.post('/api/sessions/' + session_id + '/resume', headers=HEADERS,
                           json={'limits': {'max_iterations': 15}})
    assert response.json() == {'id': session_id}
    state = wait_for(client, session_id)
    assert state['status'] == 'completed'
    assert state['limits']['max_iterations'] == 15
    assert any('paused' in e['message'] for e in state['events'])


def test_stale_running_is_inactive_and_resumable(tmp_path, client):
    save(tmp_path, ResearchState(id='stale', topic='Old session', config={'mode': 'demo'}))
    state = client.get('/api/sessions/stale').json()
    assert state['status'] == 'running'
    assert not state['active'] and not state['pause_requested']
    assert state['events'] == []
    assert client.post('/api/sessions/stale/resume', headers=HEADERS, json={}).status_code == 200
    assert wait_for(client, 'stale')['status'] == 'completed'


def test_concurrent_duplicate_and_maximum(client, tmp_path, monkeypatch):
    monkeypatch.setattr(server, 'DEMO_EVENT_DELAY', 5)
    save(tmp_path, ResearchState(id='race', topic='Race', status='paused', config={'mode': 'demo'}))
    def resume(_):
        return client.post('/api/sessions/race/resume', headers=HEADERS, json={}).status_code
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(resume, range(2)))
    assert sorted(results) == [200, 409]
    create(client)
    create(client)
    response = client.post('/api/sessions', headers=HEADERS, json={'topic': 'Fourth', 'mode': 'demo'})
    assert response.status_code == 409
    assert len(client.get('/api/sessions').json()) == 3


@pytest.mark.parametrize('body', [
    {}, {'topic': ''}, {'topic': '   '}, {'topic': 'x' * 10001},
    {'topic': 'x', 'mode': 'wrong'}, {'topic': 'x', 'model': '  '},
    {'topic': 'x', 'search': 'arbitrary'}, {'topic': 'x', 'base_url': 'https://evil.example'},
    {'topic': 'x', 'api_key': 'do-not-echo-me'},
    {'topic': 'x', 'limits': {'max_iterations': 0}},
    {'topic': 'x', 'limits': {'max_minutes': 'Infinity'}},
    {'topic': 'x', 'limits': {'max_sources': 501}},
    {'topic': 'x', 'limits': {'unknown': 1}},
])
def test_invalid_input(client, body):
    response = client.post('/api/sessions', headers=HEADERS, json=body)
    assert response.status_code == 422
    assert 'do-not-echo-me' not in response.text
    assert client.get('/api/sessions').json() == []


def test_live_validation_before_spawning_and_no_secret_exposure(client, monkeypatch):
    response = client.post('/api/sessions', headers=HEADERS, json={'topic': 'No API key'})
    assert response.status_code == 422
    monkeypatch.setenv('OPENAI_API_KEY', 'super-secret-openai')
    monkeypatch.setenv('TAVILY_API_KEY', 'super-secret-tavily')
    monkeypatch.setenv('OPENAI_BASE_URL', 'https://user:super-secret-openai@example.com/v1?key=super-secret-tavily')
    config = client.get('/api/config')
    assert config.json()['has_api_key'] is True
    assert config.json()['has_tavily_key'] is True
    assert config.json()['base_url'] == 'https://example.com/v1'
    assert 'super-secret' not in config.text
    response = client.post('/api/sessions', headers=HEADERS, json={'topic': 'Invalid endpoint'})
    assert response.status_code == 422
    assert 'super-secret' not in response.text
    assert client.get('/api/sessions').json() == []
    monkeypatch.setenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
    monkeypatch.setenv('THINK_THANK_SEARCH', 'bad-provider')
    assert client.post('/api/sessions', headers=HEADERS, json={'topic': 'Bad search'}).status_code == 422


def test_ephemeral_validation_closes_clients_and_resume_ignores_saved_url(client, tmp_path, monkeypatch):
    seen = []
    class Model(server.DemoModel):
        def __init__(self, **kwargs):
            super().__init__()
            seen.append(('model', kwargs['base_url'], threading.get_ident()))
        def close(self):
            seen.append(('close_model',))
            super().close()
    class Web(server.DemoWeb):
        def __init__(self, **kwargs):
            super().__init__()
        def close(self):
            seen.append(('close_web',))
            super().close()
    monkeypatch.setattr(server, 'ModelClient', Model)
    monkeypatch.setattr(server, 'WebClient', Web)
    monkeypatch.setenv('OPENAI_BASE_URL', 'https://configured.example/v1')
    save(tmp_path, ResearchState(id='live', topic='Topic', status='paused', config={
        'mode': 'live', 'base_url': 'https://untrusted-saved-url.example/v1', 'model': 'local-model'}))
    assert client.post('/api/sessions/live/resume', headers=HEADERS, json={}).status_code == 200
    state = wait_for(client, 'live')
    assert state['status'] == 'completed'
    models = [entry for entry in seen if entry[0] == 'model']
    assert len(models) == 2
    assert {m[1] for m in models} == {'https://configured.example/v1'}
    assert models[0][2] != models[1][2]
    assert seen.count(('close_model',)) == seen.count(('close_web',)) == 2
    assert 'base_url' not in state['config']


@pytest.mark.parametrize('origin,allowed', [
    (None, True), ('http://testserver', True), ('http://testserver:80', True),
    ('http://localhost', False), ('http://testserver:8000', False),
    ('https://testserver', False), ('http://evil.example', False),
    ('null', False), ('http://testserver.evil.example', False),
    ('http://testserver@evil.example', False), ('http://testserver/path', False),
])
def test_mutation_origin(client, origin, allowed):
    headers = dict(HEADERS)
    if origin is not None:
        headers['Origin'] = origin
    response = client.post('/api/sessions', headers=headers, json={'topic': 'Origin', 'mode': 'demo'})
    assert response.status_code == (201 if allowed else 403)
    assert 'access-control-allow-origin' not in response.headers


def test_mutations_require_custom_header_and_foreign_hosts_rejected(client):
    for route in ('/api/sessions', '/api/sessions/missing/pause', '/api/sessions/missing/resume'):
        for headers in ({}, {'X-Think-Thank': '0'}, {'Origin': 'http://testserver'}):
            assert client.post(route, headers=headers, json={}).status_code == 403
    for host in ('evil.example', 'localhost.evil.example', '0.0.0.0', '[::2]',
                 'localhost:bad', 'localhost:8000@evil.example', 'localhost:8000/path'):
        assert client.get('/api/config', headers={'Host': host}).status_code == 400
    for host in ('localhost:8000', '127.0.0.1:8000', '[::1]:8000', 'testclient'):
        assert client.get('/api/config', headers={'Host': host}).status_code == 200
    response = client.post('/api/sessions', headers={**HEADERS, 'Host': '[::1]:8000', 'Origin': 'http://[::1]:8000'},
                           json={'topic': 'IPv6', 'mode': 'demo'})
    assert response.status_code == 201
    response = client.get('/api/config')
    assert response.headers['x-content-type-options'] == 'nosniff'
    assert "frame-ancestors 'none'" in response.headers['content-security-policy']
    assert response.headers['cache-control'] == 'no-store'


def test_missing_sessions_and_report_validation(client, monkeypatch):
    assert client.get('/api/sessions/missing').status_code == 404
    assert client.get('/api/sessions/bad!').status_code == 404
    for action in ('pause', 'resume'):
        assert client.post('/api/sessions/missing/' + action, headers=HEADERS, json={}).status_code == 404
    monkeypatch.setattr(server, 'DEMO_EVENT_DELAY', 5)
    session_id = create(client)
    assert client.get('/api/sessions/' + session_id + '/report').status_code == 409
    assert client.get('/api/sessions/' + session_id + '/report?format=html').status_code == 422
    assert client.post('/api/sessions/' + session_id + '/resume', headers=HEADERS,
                       json={'limits': {'max_iterations': -1}}).status_code == 422


def test_failure_preserves_checkpoint_and_sanitizes(client, monkeypatch, tmp_path):
    class Broken(server.DemoModel):
        def complete(self, *args, **kwargs):
            raise RuntimeError('super-secret-openai PRIVATE TRACEBACK')
    monkeypatch.setattr(server, 'DemoModel', Broken)
    monkeypatch.setenv('OPENAI_API_KEY', 'super-secret-openai')
    session_id = create(client)
    state = wait_for(client, session_id)
    assert state['status'] == 'failed'
    assert state['topic'] == 'A fictional research question'
    assert state['errors'] == [server._FAILURE]
    assert 'super-secret' not in json.dumps(state)
    assert 'PRIVATE TRACEBACK' not in json.dumps(state)
    store = Store(tmp_path)
    try:
        assert store.load(session_id).status == 'failed'
    finally:
        store.close()


def test_bounded_events_and_legacy_secret_redaction(client, tmp_path, monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'secret-sentinel')
    save(tmp_path, ResearchState(id='old', topic='Topic', errors=['secret-sentinel'],
         config={'mode': 'demo', 'unknown_setting': 'secret-sentinel',
                 'base_url': 'https://user:password@example.com/v1?token=credential'}))
    manager = client.app.state.manager
    for index in range(510):
        manager._event('old', 'event secret-sentinel ' + str(index))
    response = client.get('/api/sessions/old')
    assert 'secret-sentinel' not in response.text and 'password' not in response.text
    state = response.json()
    assert state['config'] == {'mode': 'demo', 'base_url': 'https://example.com/v1'}
    assert len(state['events']) == 500
    assert state['events'][0]['seq'] == 11
    assert state['events'][-1]['seq'] == 510


def test_shutdown_requests_cooperative_pause(tmp_path, monkeypatch):
    monkeypatch.setattr(server, 'DEMO_EVENT_DELAY', 5)
    app = server.create_app(tmp_path)
    with TestClient(app) as client:
        session_id = create(client)
        wait_for(client, session_id, lambda s: bool(s['events']))
    assert app.state.manager.closed
    assert not app.state.manager.active
    store = Store(tmp_path)
    try:
        assert store.load(session_id).status == 'paused'
    finally:
        store.close()


def test_static_assets(client, tmp_path, monkeypatch):
    static = tmp_path / 'assets'
    static.mkdir()
    (static / 'index.html').write_text('<html>Think Thank</html>')
    (static / 'app.js').write_text('console.log("local");')
    monkeypatch.setattr(server, '_STATIC', static)
    with TestClient(server.create_app(tmp_path)) as ui:
        assert ui.get('/').status_code == 200
        assert 'Think Thank' in ui.get('/').text
        assert ui.get('/static/app.js').status_code == 200
        assert ui.get('/static/../research.sqlite3').status_code == 404


@pytest.mark.parametrize('host', ['0.0.0.0', '::', '192.168.1.5', 'evil.example'])
def test_serve_refuses_nonlocal_binds(tmp_path, host):
    with pytest.raises(ValueError, match='must bind'):
        server.serve(tmp_path, host=host)


def test_serve_forces_single_worker(tmp_path, monkeypatch):
    import uvicorn
    calls = []
    monkeypatch.setattr(uvicorn, 'run', lambda *args, **kwargs: calls.append((args, kwargs)))
    server.serve(tmp_path, host='localhost', port=8765)
    assert calls[0][1] == {'host': 'localhost', 'port': 8765, 'workers': 1, 'proxy_headers': False}
