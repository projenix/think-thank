"""Offline demo contract and complete engine/export integration tests."""
import hashlib
import json
import socket

import httpx
import pytest

from think_thank.demo import DEMO_TOPIC, NOTICE, DemoModel, DemoWeb
from think_thank.engine import ResearchEngine
from think_thank.llm import ModelError
from think_thank.models import Action, Extraction, Limits, Plan, Report, Review
from think_thank.reporting import valid_evidence
from think_thank.store import Store
from think_thank.web import WebError


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('The demo must never access the network')
    monkeypatch.setattr(socket, 'getaddrinfo', forbidden)
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setattr(httpx.Client, 'send', forbidden)


@pytest.fixture
def adapters(tmp_path):
    model, web, store = DemoModel(), DemoWeb(), Store(tmp_path)
    try:
        yield model, web, store
    finally:
        model.close()
        web.close()
        store.close()


def test_demo_completes_with_citations_and_exported_artifacts(adapters):
    model, web, store = adapters
    engine = ResearchEngine(model, web, store)
    state = engine.run(engine.create(DEMO_TOPIC))
    assert state.status == 'completed'
    assert state.iteration == 2
    assert state.counterchecks == 1
    assert state.review_ready
    assert not state.errors
    assert len(state.plan.questions) == 2
    assert all(q.status == 'answered' and len(q.evidence_ids) == 2 for q in state.plan.questions)
    assert len(state.sources) == len(state.evidence) == 4
    assert [e.id for e in state.evidence] == ['E1', 'E2', 'E3', 'E4']
    assert set(valid_evidence(state)) == {'E1', 'E2', 'E3', 'E4'}
    hypothesis = state.hypotheses[0]
    assert hypothesis.status == 'mixed'
    assert hypothesis.supporting_evidence == ['E1', 'E2']
    assert hypothesis.opposing_evidence == ['E3', 'E4']
    assert 'confirmatory' in state.decisions[0].rationale
    assert any('contrary' in q and 'limitations' in q for q in state.searched_queries)
    assert NOTICE in state.report.executive_summary
    assert set(eid for f in state.report.findings for eid in f.evidence_ids) == set(valid_evidence(state))
    assert 0 < state.tokens_used == model.total_input_tokens + model.total_output_tokens < 100
    assert state.estimated_cost_usd > 0  # Simulated accounting, not an invoice.
    assert store.load(state.id).model_dump() == state.model_dump()

    directory = store.root / 'sessions' / state.id
    for name in ('state', 'plan', 'evidence', 'hypotheses', 'decisions', 'outline', 'report'):
        assert json.loads((directory / (name + '.json')).read_text()) is not None
    report = (directory / 'report.md').read_text()
    assert 'fictional demonstration data' in report
    assert 'not real research' in report
    for source, evidence in zip(state.sources, state.evidence):
        assert source.url.startswith('https://example.org/fictional-demo/')
        assert NOTICE in source.text
        assert evidence.quote in source.text
        assert evidence.quote in report
        assert '[%s → %s]' % (evidence.id, source.id) in report
        assert (directory / 'sources' / (source.id + '.txt')).read_text() == source.text
    first_plan = json.loads((directory / 'checkpoints' / '0001' / 'plan.json').read_text())
    assert first_plan['questions'][1]['status'] == 'open'
    final_hypotheses = json.loads((directory / 'checkpoints' / '0002' / 'hypotheses.json').read_text())
    assert final_hypotheses[0]['status'] == 'mixed'
    count = state.tokens_used
    assert engine.run(state) is state
    assert state.tokens_used == count


def test_web_is_deterministic_and_counterchecks_use_different_sources():
    web = DemoWeb()
    positive = web.search(DEMO_TOPIC)
    contrary = web.search(DEMO_TOPIC + ' contrary evidence limitations')
    assert len(positive) == len(contrary) == 2
    assert not {r.url for r in positive} & {r.url for r in contrary}
    assert web.search(DEMO_TOPIC, limit=1) == positive[:1]
    assert web.search(DEMO_TOPIC, limit=0) == []
    first = web.read(positive[0].url, 'S-custom')
    assert first == web.read(positive[0].url, 'S-custom')
    assert first.id == 'S-custom'
    assert first.content_hash == hashlib.sha256(first.text.encode()).hexdigest()
    first.text = 'Caller mutation must not affect fixtures'
    assert NOTICE in web.read(positive[0].url, 'S-custom').text
    with pytest.raises(WebError, match='no network'):
        web.read('https://example.org/not-a-fixture', 'S1')
    web.close()
    web.close()
    with pytest.raises(WebError, match='closed'):
        web.search(DEMO_TOPIC)
    with pytest.raises(WebError, match='closed'):
        web.read(positive[0].url, 'S1')


def test_model_guard_accounting_determinism_and_close():
    model = DemoModel()
    calls = []
    model.before_request = lambda incoming, outgoing: calls.append(
        (incoming, outgoing, model.total_input_tokens, model.total_output_tokens))
    plan = model.complete('plan', {'topic': DEMO_TOPIC}, Plan, max_tokens=1)
    assert calls == [(1, 1, 0, 0)]
    assert model.total_input_tokens == model.total_output_tokens == 1
    assert model.complete('plan', {'topic': DEMO_TOPIC}, Plan) == plan

    def denied(*args):
        raise RuntimeError('Budget denied')
    model.before_request = denied
    with pytest.raises(RuntimeError, match='Budget denied'):
        model.complete('plan', {'topic': DEMO_TOPIC}, Plan)
    assert model.total_input_tokens == model.total_output_tokens == 2
    model.close()
    model.close()
    with pytest.raises(ModelError, match='closed'):
        model.complete('plan', {}, Plan)


def test_model_preserves_supplied_ids_and_exact_quotes():
    model, web = DemoModel(), DemoWeb()
    plan = model.complete('plan', {}, Plan)
    plan.questions[0].id, plan.questions[1].id = 'Q-custom-a', 'Q-custom-b'
    evidence = []
    results = web.search(DEMO_TOPIC) + web.search('contrary limitations')
    for index, result in enumerate(results):
        source = web.read(result.url, 'S-custom-%d' % index)
        extracted = model.complete('extract', {
            'source': source.model_dump(), 'existing_evidence_ids': ['E1', 'E2']}, Extraction)
        assert len(extracted.evidence) == 1
        item = extracted.evidence[0]
        assert item.id == 'E3'
        assert item.source_id == source.id
        assert item.quote in source.text
        item.id = 'E-custom-%d' % index
        evidence.append(item.model_dump())
    context = {'iteration': 2, 'counterchecks': 1, 'plan': plan.model_dump(),
               'evidence': evidence, 'hypotheses': [{'id': 'H-custom'}]}
    review = model.complete('review', context, Review)
    assert review.ready
    assert [q.id for q in review.questions] == ['Q-custom-a', 'Q-custom-b']
    assert review.hypotheses[0].id == 'H-custom'
    assert review.hypotheses[0].supporting_evidence == ['E-custom-0', 'E-custom-1']
    assert review.hypotheses[0].opposing_evidence == ['E-custom-2', 'E-custom-3']
    report = model.complete('report', context, Report)
    assert {eid for f in report.findings for eid in f.evidence_ids} == {e['id'] for e in evidence}
    for update in ({'iteration': 1}, {'counterchecks': 0}, {'evidence': evidence[:2]}):
        assert not model.complete('review', dict(context, **update), Review).ready
    context['iteration'] = 1
    first = model.complete('action', context, Action)
    assert not first.disconfirming
    context['iteration'] = 2
    second = model.complete('action', context, Action)
    assert second.disconfirming
    assert first.queries != second.queries
    assert second.question_ids == ['Q-custom-a', 'Q-custom-b']
    source.text = 'The expected quotation is absent.'
    assert not model.complete('extract', {'source': source.model_dump()}, Extraction).evidence


def test_limited_demo_does_not_claim_complete_coverage(adapters):
    model, web, store = adapters
    engine = ResearchEngine(model, web, store)
    state = engine.run(engine.create(DEMO_TOPIC, Limits(max_iterations=1)))
    assert state.status == 'limited'
    assert not state.review_ready
    assert state.counterchecks == 0
    assert NOTICE in state.report.executive_summary
    assert any('incomplete evidence coverage' in text for text in state.report.limitations)
    # A fresh adapter can resume from persisted context, not private call counters.
    restored = store.load(state.id)
    restored.limits.max_iterations = 2
    resumed = ResearchEngine(DemoModel(), DemoWeb(), store).run(restored)
    assert resumed.status == 'completed'
    assert resumed.iteration == 2
    assert len(resumed.evidence) == 4


@pytest.mark.parametrize('task,schema,max_tokens', [
    ('unknown', Plan, 10), ('plan', Report, 10), ('plan', Plan, 0),
    ('plan', Plan, True),
])
def test_model_rejects_invalid_requests(task, schema, max_tokens):
    model = DemoModel()
    with pytest.raises(ModelError):
        model.complete(task, {}, schema, max_tokens=max_tokens)
    assert model.total_input_tokens == model.total_output_tokens == 0
