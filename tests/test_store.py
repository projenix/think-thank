"""Offline tests for canonical state, recoverable artifacts and path safety."""
import json
from pathlib import Path

import pytest

from think_thank import store as module
from think_thank.models import (Decision, Evidence, Hypothesis, Plan, Question,
                               Report, ResearchState, Source)
from think_thank.store import Store


@pytest.fixture
def state():
    return ResearchState(
        id="research-123", topic="A research topic", iteration=2,
        sources=[Source(id="s1", url="https://example.org", title="Original", text="Exact text.\n第二行")],
        evidence=[Evidence(id="e1", source_id="s1", claim="A claim", quote="Exact text.")],
        plan=Plan(scope="Scope", success_criteria=["Cited"], questions=[Question(id="q1", text="Why?")], outline=["Intro"]),
        hypotheses=[Hypothesis(id="h1", statement="Maybe", test="Check", supporting_evidence=["e1"])],
        decisions=[Decision(iteration=1, action="research", rationale="Need sources")],
        outline=["Results"],
        report=Report(title="Draft", executive_summary="Summary", findings=[], competing_explanations=[], limitations=[], open_questions=[]),
    )


def test_roundtrip_and_artifacts(tmp_path, state):
    store = Store(tmp_path)
    store.save(state)
    assert store.load(state.id) == state
    session = tmp_path / "sessions" / state.id
    assert (session / "sources/s1.txt").read_text() == state.sources[0].text
    assert json.loads((session / "report.json").read_text()) == state.report.model_dump(mode="json")
    assert (session / "report.md").read_text().startswith("# A research topic\n")
    for key in ("plan", "evidence", "hypotheses", "decisions", "outline"):
        expected = state.model_dump(mode="json")[key]
        assert json.loads((session / (key + ".json")).read_text()) == expected
        assert json.loads((session / "checkpoints/0002" / (key + ".json")).read_text()) == expected
    assert not list(session.rglob(".tmp-*"))
    store.close()
    reopened = Store(tmp_path)
    assert reopened.load(state.id) == state
    reopened.close()


def test_upsert_listing_and_checkpoints(tmp_path, state):
    store = Store(tmp_path)
    store.save(state)
    state.iteration = 3
    state.status = "limited"
    state.stop_reason = "token budget"
    state.outline = ["Revised"]
    store.save(state)
    rows = store.list_sessions()
    assert rows == [{key: getattr(state, key) for key in ("id", "topic", "status", "iteration", "updated_at", "stop_reason")}]
    folder = tmp_path / "sessions" / state.id
    assert json.loads((folder / "checkpoints/0002/outline.json").read_text()) == ["Results"]
    assert json.loads((folder / "checkpoints/0003/outline.json").read_text()) == ["Revised"]
    store.close()


def test_sqlite_not_json_is_canonical(tmp_path, state):
    store = Store(tmp_path)
    store.save(state)
    folder = tmp_path / "sessions" / state.id
    (folder / "state.json").write_text("corrupt")
    (folder / "evidence.json").unlink()
    assert store.load(state.id) == state
    store.save(store.load(state.id))
    assert json.loads((folder / "state.json").read_text())["id"] == state.id
    store.close()


def test_export_failure_keeps_committed_state_and_previous_artifact(tmp_path, state, monkeypatch):
    store = Store(tmp_path)
    store.save(state)
    previous = (tmp_path / "sessions" / state.id / "state.json").read_text()
    state.topic = "New topic"

    def fail(*args):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(module.os, "replace", fail)
    with pytest.raises(OSError):
        store.save(state)
    assert store.load(state.id).topic == "New topic"
    assert (tmp_path / "sessions" / state.id / "state.json").read_text() == previous
    assert not list(tmp_path.rglob(".tmp-*"))
    store.close()


@pytest.mark.parametrize("bad", ["", ".", "..", "../outside", "/tmp/abs", "x/y", "x\\y", "a\x00b", "a b", "é", "a" * 129, "x' OR 1=1--", "%2e%2e"])
def test_invalid_session_ids(tmp_path, bad):
    store = Store(tmp_path)
    with pytest.raises(ValueError):
        store.save(ResearchState(id=bad, topic="topic"))
    with pytest.raises(ValueError):
        store.load(bad)
    assert store.list_sessions() == []
    store.close()


def test_invalid_source_id_and_duplicates(tmp_path, state):
    store = Store(tmp_path)
    state.sources[0].id = "../../escape"
    with pytest.raises(ValueError):
        store.save(state)
    state.sources[0].id = "s1"
    state.sources.append(state.sources[0].model_copy())
    with pytest.raises(ValueError):
        store.save(state)
    assert store.list_sessions() == []
    store.close()


@pytest.mark.parametrize("part", ["sessions", "session", "sources", "checkpoints"])
def test_symlink_directories_rejected(tmp_path, state, part):
    root = tmp_path / "data"
    outside = tmp_path / "outside"
    outside.mkdir()
    store = Store(root)
    session = root / "sessions" / state.id
    paths = {"sessions": root / "sessions", "session": session,
             "sources": session / "sources", "checkpoints": session / "checkpoints"}
    target = paths[part]
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.rmdir()
    target.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        store.save(state)
    assert list(outside.iterdir()) == []
    store.close()


def test_file_symlink_never_overwrites_target(tmp_path):
    target = tmp_path / "private"
    target.write_text("unchanged")
    link = tmp_path / "link"
    link.symlink_to(target)
    with pytest.raises(ValueError):
        module._atomic_write(link, "overwrite")
    assert target.read_text() == "unchanged"


def test_credentials_not_persisted_and_caller_not_modified(tmp_path, state, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "environment-secret")
    state.config = {"model": "offline", "api_key": "config-secret", "Authorization": "Bearer hidden", "password": "password-secret"}
    store = Store(tmp_path)
    store.save(state)
    assert store.load(state.id).config == {"model": "offline"}
    assert state.config["api_key"] == "config-secret"
    store.close()
    for path in tmp_path.rglob("*"):
        if path.is_file():
            data = path.read_bytes()
            for secret in (b"environment-secret", b"config-secret", b"Bearer hidden", b"password-secret"):
                assert secret not in data


def test_missing_closed_and_report_removal(tmp_path, state):
    store = Store(tmp_path)
    with pytest.raises(FileNotFoundError):
        store.load("missing")
    store.save(state)
    state.report = None
    store.save(state)
    folder = tmp_path / "sessions" / state.id
    assert not (folder / "report.md").exists()
    assert not (folder / "report.json").exists()
    store.close()
    store.close()
    for operation in (lambda: store.load(state.id), lambda: store.save(state), store.list_sessions):
        with pytest.raises(RuntimeError, match="closed"):
            operation()
