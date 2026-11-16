import json
from think_thank.cli import main
from think_thank.store import Store


def test_cli_demo_list_show_resume(tmp_path, capsys):
    base = ['--data-dir', str(tmp_path)]
    assert main(base + ['list']) == 0
    assert 'No research' in capsys.readouterr().out
    assert main(base + ['demo', '--quiet']) == 0
    assert 'Status: completed' in capsys.readouterr().out
    store = Store(tmp_path)
    session = store.list_sessions()[0]['id']
    store.close()
    assert main(base + ['show', session, '--json']) == 0
    data = json.loads(capsys.readouterr().out)
    assert data['status'] == 'completed'
    assert data['config']['mode'] == 'demo'
    assert main(base + ['show', session]) == 0
    assert 'fictional' in capsys.readouterr().out.lower()
    assert main(base + ['list']) == 0
    assert session in capsys.readouterr().out
    assert main(base + ['resume', session]) == 0
    assert 'already completed' in capsys.readouterr().out


def test_missing_credentials_fail_cleanly(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    monkeypatch.delenv('OPENAI_BASE_URL', raising=False)
    assert main(['--data-dir', str(tmp_path), 'research', 'Test topic']) == 1
    assert 'API key' in capsys.readouterr().err


def test_unknown_session(tmp_path, capsys):
    assert main(['--data-dir', str(tmp_path), 'show', 'absent']) == 1
    assert 'not found' in capsys.readouterr().err


def test_resume_demo_with_increased_cumulative_limits(tmp_path, capsys):
    from think_thank.demo import DemoModel, DemoWeb, DEMO_TOPIC
    from think_thank.engine import ResearchEngine
    from think_thank.models import Limits
    store = Store(tmp_path)
    engine = ResearchEngine(DemoModel(), DemoWeb(), store)
    state = engine.create(DEMO_TOPIC, Limits(max_iterations=1), {'mode': 'demo'})
    engine.run(state)
    assert state.status == 'limited'
    store.close()
    assert main(['--data-dir', str(tmp_path), 'resume', state.id, '--max-iterations', '5', '--quiet']) == 0
    assert 'Status: completed' in capsys.readouterr().out
