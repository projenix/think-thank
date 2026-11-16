"""Integration contracts for the CLI server command and cooperative engine pause."""
from think_thank.cli import main
from think_thank.demo import DemoModel, DemoWeb, DEMO_TOPIC
from think_thank.engine import ResearchEngine
from think_thank.store import Store


def test_serve_cli_passes_options(monkeypatch, tmp_path):
    from think_thank import server
    calls = []
    monkeypatch.setattr(server, 'serve', lambda **kwargs: calls.append(kwargs))
    assert main(['--data-dir', str(tmp_path), 'serve', '--port', '8123']) == 0
    assert calls == [{'data_dir': tmp_path, 'host': '127.0.0.1', 'port': 8123}]


def test_cooperative_pause_before_first_request_and_resume(tmp_path):
    store = Store(tmp_path)
    try:
        model = DemoModel()
        engine = ResearchEngine(model, DemoWeb(), store, should_stop=lambda: True)
        state = engine.create(DEMO_TOPIC, config={'mode': 'demo'})
        state = engine.run(state)
        assert state.status == 'paused'
        assert model.total_input_tokens == 0
        assert store.load(state.id).status == 'paused'
        engine = ResearchEngine(DemoModel(), DemoWeb(), store)
        assert engine.run(store.load(state.id)).status == 'completed'
    finally:
        store.close()
