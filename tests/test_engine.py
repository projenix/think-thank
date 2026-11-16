"""Offline orchestration regressions, including durable checkpoints and budgets.

Adapters deliberately return schema-valid but untrustworthy model output: the
engine must enforce provenance and completion independently of model prompts.
"""
import copy
import hashlib
import socket

import httpx
import pytest

from think_thank import engine as engine_module
from think_thank import llm
from think_thank.demo import DEMO_TOPIC, DemoModel, DemoWeb
from think_thank.engine import BudgetExceeded, ResearchEngine
from think_thank.llm import ModelError
from think_thank.models import (
    Action, Evidence, Extraction, Finding, Hypothesis, Limits, Plan, Question,
    Report, Review, SearchResult, Source,
)
from think_thank.store import Store


class ScriptedModel(DemoModel):
    def __init__(self, **handlers):
        super().__init__()
        self.handlers = handlers
        self.calls = []

    def complete(self, task, context, schema, max_tokens=4000):
        self.calls.append((task, copy.deepcopy(context)))
        if task not in self.handlers:
            return super().complete(task, context, schema, max_tokens)
        if self.before_request:
            self.before_request(1, 1)
        self.total_input_tokens += 1
        self.total_output_tokens += 1
        handler = self.handlers[task]
        if isinstance(handler, BaseException):
            raise handler
        result = handler(context) if callable(handler) else handler
        return schema.model_validate(result.model_dump())

    @property
    def tasks(self):
        return [task for task, _ in self.calls]


class RecordingWeb(DemoWeb):
    def __init__(self, empty=False):
        super().__init__()
        self.queries = []
        self.reads = []
        self.empty = empty

    def search(self, query, limit=5):
        self.queries.append(query)
        return [] if self.empty else super().search(query, limit)

    def read(self, url, source_id):
        self.reads.append(url)
        return super().read(url, source_id)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Engine tests must not use real networking')
    monkeypatch.setattr(socket, 'getaddrinfo', forbidden)
    monkeypatch.setattr(socket.socket, 'connect', forbidden)


@pytest.fixture
def clock(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(engine_module.time, 'monotonic', lambda: now[0])
    return now


@pytest.fixture
def rig(tmp_path, clock):
    store = Store(tmp_path)
    adapters = []

    def make(model=None, web=None, limits=None):
        model = model or ScriptedModel()
        web = web or RecordingWeb()
        adapters.extend((model, web))
        engine = ResearchEngine(model, web, store)
        state = engine.create(DEMO_TOPIC, limits)
        # Bind the clock/state for focused tests of individual transitions.
        engine.state = state
        engine._started = clock[0]
        return engine, state

    yield make
    for adapter in adapters:
        adapter.close()
    store.close()


def plan():
    return Plan(scope='Test scope', success_criteria=['Evidence and counterchecks'],
                questions=[Question(id='Q1', text='Does it work?')], outline=['Results'])


def evidence(eid='E1', source_id='S1', quote='Exact observation.'):
    return Evidence(id=eid, source_id=source_id, claim='An observation', quote=quote)


def source(sid='S1', url='https://example.org/source', text='Exact observation.'):
    return Source(id=sid, url=url, title='Test source', text=text,
                  content_hash=hashlib.sha256(text.encode()).hexdigest())


def review(**changes):
    fields = dict(questions=[], hypotheses=[], outline=['Updated outline'],
                  gaps=['Uncertainty remains'], next_priority='Check alternatives', ready=False)
    fields.update(changes)
    return Review(**fields)


def report(findings=()):
    return Report(title='Test report', executive_summary='Provisional synthesis',
                  findings=list(findings), competing_explanations=[],
                  limitations=[], open_questions=[])


def ready_state(state):
    state.plan = plan()
    state.sources = [source()]
    state.processed_source_ids = ['S1']
    state.evidence = [evidence()]
    state.plan.questions[0].status = 'answered'
    state.plan.questions[0].evidence_ids = ['E1']
    state.counterchecks = 1
    state.review_ready = True


def test_final_report_removes_unknown_citations_and_downgrades_findings(rig):
    findings = [Finding(heading=name, text='Claim', evidence_ids=ids, confidence='high')
                for name, ids in [('mixed', ['E1', 'missing', 'E1']),
                                  ('invented', ['missing']), ('uncited', []),
                                  ('valid', ['E1', 'E1'])]]
    engine, state = rig(ScriptedModel(report=report(findings)))
    state.evidence = [evidence()]
    engine._finalize()
    assert [f.evidence_ids for f in state.report.findings] == [['E1'], [], [], ['E1']]
    assert [f.confidence for f in state.report.findings] == ['low', 'low', 'low', 'high']
    assert sum('lacked valid supporting citations' in x for x in state.report.limitations) == 3
    assert any('do not establish truth' in x for x in state.report.limitations)
    assert engine.store.load(state.id).report == state.report


def test_extraction_enforces_exact_quotes_source_ids_and_deduplication(rig):
    exact = evidence(eid='model-controlled')
    rows = [exact, exact.model_copy(), evidence(quote='exact observation.'),
            evidence(quote='Exact  observation.'), evidence(quote='Invented quotation'),
            evidence(quote=' '), evidence(source_id='S999')]
    engine, state = rig(ScriptedModel(extract=Extraction(evidence=rows)))
    state.plan, state.sources = plan(), [source(text='Prefix. Exact observation. Suffix.')]
    engine._extract_pending()
    assert state.evidence == [evidence()]
    assert len(state.errors) == 5
    assert state.processed_source_ids == ['S1']
    engine._extract_pending()
    assert engine.model.tasks == ['extract']
    assert engine.store.load(state.id).evidence == state.evidence


@pytest.mark.parametrize('missing', ['plan', 'evidence', 'countercheck', 'ready', 'answer', 'citation'])
def test_completion_requires_every_gate(rig, missing):
    engine, state = rig()
    ready_state(state)
    assert engine._can_finish()
    if missing == 'plan':
        state.plan = None
    elif missing == 'evidence':
        state.evidence = []
    elif missing == 'countercheck':
        state.counterchecks = 0
    elif missing == 'ready':
        state.review_ready = False
    elif missing == 'answer':
        state.plan.questions[0].status = 'partial'
    else:
        state.plan.questions[0].evidence_ids = []
    assert not engine._can_finish()


def test_one_unanswered_question_blocks_completion(rig):
    engine, state = rig()
    ready_state(state)
    state.plan.questions.append(Question(id='Q2', text='What are the limitations?'))
    assert not engine._can_finish()


def test_model_finish_is_replaced_with_actual_disconfirming_research(rig):
    model = ScriptedModel(action=Action(action='finish', rationale='Trust me'), review=review(ready=True))
    engine, state = rig(model, RecordingWeb(empty=True), Limits(max_iterations=1))
    result = engine.run(state)
    assert result.status == 'limited'
    assert result.counterchecks == 0
    assert engine.web.queries == [DEMO_TOPIC + ' conflicting evidence limitations']
    assert result.decisions[0].action == 'research'
    assert 'Completion gate not met' in result.decisions[0].rationale


def test_plan_ids_and_question_status_are_application_controlled(rig):
    proposal = plan()
    proposal.questions[0] = Question(id='../invented', text='Question', status='answered', evidence_ids=['bogus'])
    engine, state = rig(ScriptedModel(plan=proposal, action=KeyboardInterrupt()))
    engine.run(state)
    assert state.status == 'paused'
    assert state.plan.questions == [Question(id='Q1', text='Question')]
    assert engine.store.load(state.id).plan == state.plan


def test_review_ignores_unknown_questions_filters_refs_and_downgrades_answers(rig):
    updates = [Question(id='Q1', text='Do not overwrite original text', status='answered', evidence_ids=['bad']),
               Question(id='Q2', text='Other', status='answered', evidence_ids=['E1', 'E1', 'bad']),
               Question(id='invented', text='Injected question', status='answered', evidence_ids=['E1'])]
    engine, state = rig(ScriptedModel(review=review(questions=updates, ready=True)))
    state.plan = plan()
    state.plan.questions.append(Question(id='Q2', text='Second original question'))
    state.evidence = [evidence()]
    engine._review()
    assert [q.id for q in state.plan.questions] == ['Q1', 'Q2']
    assert state.plan.questions[0].text == 'Does it work?'
    assert state.plan.questions[0].status == 'partial'
    assert state.plan.questions[0].evidence_ids == []
    assert state.plan.questions[1].status == 'answered'
    assert state.plan.questions[1].evidence_ids == ['E1']
    assert not engine._can_finish()


def test_hypotheses_merge_by_id_and_drop_invalid_or_duplicate_references(rig):
    old = Hypothesis(id='H1', statement='Old', test='Compare groups')
    untouched = Hypothesis(id='H2', statement='Keep this alternative', test='Collect data')
    updated = Hypothesis(id='H1', statement='Revised', test='Compare groups', status='supported',
                         supporting_evidence=['E1', 'bad', 'E1'], opposing_evidence=['E2', 'bad'])
    unsupported = Hypothesis(id='H3', statement='Unsupported', test='Collect data', status='supported',
                             supporting_evidence=['missing'])
    unopposed = Hypothesis(id='H4', statement='Not weakened', test='Collect data', status='weakened',
                          opposing_evidence=['missing'])
    engine, state = rig(ScriptedModel(review=review(hypotheses=[updated, unsupported, unopposed])))
    state.plan, state.evidence, state.hypotheses = plan(), [evidence(), evidence('E2')], [old, untouched]
    engine._review()
    by_id = {h.id: h for h in state.hypotheses}
    assert len(state.hypotheses) == 4
    assert by_id['H1'].statement == 'Revised'
    assert by_id['H1'].supporting_evidence == ['E1']
    assert by_id['H1'].opposing_evidence == ['E2']
    assert by_id['H1'].status == 'mixed'
    assert by_id['H2'] == untouched
    assert by_id['H3'].status == by_id['H4'].status == 'unresolved'
    assert by_id['H3'].supporting_evidence == by_id['H4'].opposing_evidence == []


def test_duplicate_source_content_and_urls_do_not_create_extra_evidence(rig):
    class MirrorWeb(RecordingWeb):
        def search(self, query, limit=5):
            return [SearchResult(url=url, title='Mirror') for url in
                    ['https://example.org/a', 'https://example.org/a', 'https://example.org/b']]

        def read(self, url, source_id):
            self.reads.append(url)
            return source(source_id, url)

    engine, state = rig(ScriptedModel(extract=Extraction(evidence=[evidence()])), MirrorWeb())
    state.plan = plan()
    action = Action(action='research', rationale='Compare mirrors', queries=['one'], disconfirming=True)
    engine._research(action)
    assert len(state.sources) == len(state.evidence) == 1
    assert len(engine.web.reads) == 2
    assert engine.model.tasks == ['extract']
    assert state.counterchecks == 1
    action.queries = ['two']
    engine._research(action)
    assert len(state.sources) == 1
    assert state.counterchecks == 1  # Re-reading a mirror is not another countercheck.


def test_iteration_limit_stops_research_but_allows_final_synthesis(rig):
    engine, state = rig(limits=Limits(max_iterations=1))
    engine.run(state)
    assert state.status == 'limited'
    assert state.stop_reason == 'Iteration limit reached.'
    assert state.iteration == 1
    assert engine.model.tasks.count('action') == engine.model.tasks.count('review') == 1
    assert engine.model.tasks[-1] == 'report'
    assert state.report is not None


@pytest.mark.parametrize('kind', ['runtime', 'tokens', 'cost'])
def test_exhausted_budget_makes_no_model_call_including_report(rig, kind):
    engine, state = rig()
    state.plan = plan()
    if kind == 'runtime':
        state.elapsed_seconds = state.limits.max_minutes * 60
    elif kind == 'tokens':
        state.tokens_used = state.limits.max_tokens
    else:
        state.estimated_cost_usd = state.limits.max_cost_usd
    engine.run(state)
    assert state.status == 'limited'
    assert kind.rstrip('s') in state.stop_reason.lower()
    assert engine.model.tasks == []
    assert state.report is not None
    assert 'stopped before a final model synthesis' in state.report.executive_summary
    assert engine.store.load(state.id).model_dump() == state.model_dump()


@pytest.mark.parametrize('limits', [Limits(max_tokens=1000), Limits(max_cost_usd=0.000001)])
def test_insufficient_reservation_prevents_initial_and_paid_final_call(rig, limits):
    engine, state = rig(limits=limits)
    engine.run(state)
    assert state.status == 'limited'
    assert 'Insufficient' in state.stop_reason
    assert engine.model.tasks == []
    assert state.tokens_used == state.estimated_cost_usd == 0
    assert state.report.findings == []


def test_runtime_expiring_during_action_prevents_search_review_and_report(rig, clock):
    def slow_action(context):
        clock[0] += 61
        return Action(action='research', rationale='Search', queries=['new query'])
    engine, state = rig(ScriptedModel(action=slow_action), limits=Limits(max_minutes=1))
    engine.run(state)
    assert state.status == 'limited'
    assert state.stop_reason == 'Runtime budget reached.'
    assert state.elapsed_seconds == 61
    assert engine.model.tasks == ['plan', 'action']
    assert engine.web.queries == []


def test_accounting_tracks_deltas_and_resumed_usage_without_double_charging(rig, clock):
    model = ScriptedModel()
    model.total_input_tokens, model.total_output_tokens = 100, 200
    engine, state = rig(model)
    state.tokens_used, state.estimated_cost_usd = 50, 0.25
    state.elapsed_seconds = 10
    engine._elapsed_before = 10
    model.total_input_tokens += 20
    model.total_output_tokens += 5
    clock[0] += 7
    engine._save()
    engine._save()
    assert state.tokens_used == 75
    assert state.estimated_cost_usd == pytest.approx(0.25004)
    assert state.elapsed_seconds == 17
    assert engine.store.load(state.id).tokens_used == 75


def test_no_progress_stops_after_three_iterations(rig):
    engine, state = rig(web=RecordingWeb(empty=True))
    engine.run(state)
    assert state.status == 'limited'
    assert state.iteration == state.no_progress_iterations == 3
    assert 'Diminishing returns' in state.stop_reason
    assert engine.model.tasks.count('action') == engine.model.tasks.count('review') == 3
    assert state.evidence == []
    assert state.counterchecks == 0


def test_new_evidence_resets_no_progress_counter(rig):
    engine, state = rig(limits=Limits(max_iterations=1))
    state.no_progress_iterations = 2
    engine.run(state)
    assert state.evidence
    assert state.no_progress_iterations == 0


def test_disconfirming_query_injected_on_second_iteration_without_counterchecks(rig):
    def confirm(context):
        i = context['iteration']
        return Action(action='research', rationale='Only confirm',
                      queries=['confirm %d %d' % (i, n) for n in range(3)])
    engine, state = rig(ScriptedModel(action=confirm), RecordingWeb(empty=True), Limits(max_iterations=2))
    engine.run(state)
    injected = DEMO_TOPIC + ' contrary evidence limitations alternative explanations'
    assert engine.web.queries == ['confirm 1 0', 'confirm 1 1', 'confirm 1 2', injected,
                                  'confirm 2 0', 'confirm 2 1']
    assert state.counterchecks == 0  # Injecting a query alone is not successful evidence gathering.


def test_model_failure_preserves_evidence_and_uses_fallback_if_report_also_fails(rig):
    model = ScriptedModel(action=ModelError('Action failed'), report=ModelError('Report failed'))
    engine, state = rig(model)
    ready_state(state)
    state.review_ready = False
    engine.run(state)
    assert state.status == 'failed'
    assert state.stop_reason == 'Action failed'
    assert state.errors == ['Action failed', 'Report failed']
    assert state.evidence == [evidence()]
    assert state.report.findings[0].evidence_ids == ['E1']
    assert state.report.findings[0].confidence == 'low'
    assert state.tokens_used == 4  # Failed calls are still accounted for.
    assert engine.store.load(state.id).model_dump() == state.model_dump()


@pytest.mark.parametrize('interruption', [KeyboardInterrupt(), ModelError('Extraction failed')],
                         ids=['ctrl-c', 'model-error'])
def test_pending_source_is_checkpointed_before_extraction_and_resumes(rig, interruption):
    engine, state = rig(limits=Limits(max_iterations=1))

    def interrupt(context):
        saved = engine.store.load(state.id)
        assert saved.sources[0].id == context['source']['id']
        assert saved.evidence == saved.processed_source_ids == []
        raise interruption

    engine.model.handlers['extract'] = interrupt
    engine.run(state)
    assert state.status == ('paused' if isinstance(interruption, KeyboardInterrupt) else 'failed')
    assert len(state.sources) == 1
    assert state.processed_source_ids == []
    if state.status == 'paused':
        assert 'report' not in engine.model.tasks
        assert state.report is None
    restored = engine.store.load(state.id)
    resumed_model, resumed_web = ScriptedModel(), RecordingWeb()
    resumed = ResearchEngine(resumed_model, resumed_web, engine.store).run(restored)
    assert resumed_model.tasks == ['extract', 'report']
    assert resumed_web.queries == resumed_web.reads == []
    assert resumed.processed_source_ids == ['S1']
    assert len(resumed.evidence) == len(resumed.sources) == 1
    assert resumed.status == 'limited'
    assert resumed.tokens_used == state.tokens_used + 4
    assert engine.store.load(state.id).model_dump() == resumed.model_dump()


def test_ctrl_c_during_report_leaves_resumable_state(rig):
    engine, state = rig(ScriptedModel(report=KeyboardInterrupt()))
    ready_state(state)
    engine.run(state)
    assert state.status == 'paused'
    assert state.stop_reason == 'Interrupted during report synthesis.'
    assert state.report is None
    restored = engine.store.load(state.id)
    model = ScriptedModel(report=report())
    resumed = ResearchEngine(model, RecordingWeb(), engine.store).run(restored)
    assert resumed.status == 'completed'
    assert model.tasks == ['report']
    assert resumed.evidence == state.evidence


def test_completed_session_is_idempotent(rig):
    engine, state = rig()
    state.status = 'completed'
    before = state.model_dump()
    assert engine.run(state) is state
    assert state.model_dump() == before
    assert engine.model.tasks == []
    assert engine.web.queries == []


@pytest.mark.parametrize('failure,allowed', [('http-retry', 1), ('http-retry', 2),
                                            ('transport-retry', 1), ('transport-retry', 2),
                                            ('json-repair', 1)])
@pytest.mark.parametrize('budget', ['tokens', 'cost', 'runtime'])
def test_real_model_client_checks_engine_budget_before_every_retry(rig, monkeypatch, clock, failure, allowed, budget):
    """Deny the second or third HTTP attempt; no retry bypasses the guard."""
    monkeypatch.setattr(llm.time, 'sleep', lambda seconds: None)
    model = llm.ModelClient(api_key='test-key')
    model._client.close()
    engine, state = rig(model)
    state.plan = plan()
    requests = []

    def handler(request):
        requests.append(request)
        # Tighten after the permitted attempts to leave insufficient headroom
        # for another reservation or final synthesis. No HTTP call is real.
        if len(requests) == allowed:
            if budget == 'tokens':
                state.limits.max_tokens = model.total_input_tokens + model.total_output_tokens + 1
            elif budget == 'cost':
                state.limits.max_cost_usd = ((model.total_input_tokens * state.limits.input_usd_per_million
                                             + model.total_output_tokens * state.limits.output_usd_per_million)
                                            / 1e6 + 0.000001)
            else:
                clock[0] += state.limits.max_minutes * 60
        if failure == 'transport-retry':
            raise httpx.ConnectError('Mock connection failed', request=request)
        if failure == 'http-retry':
            return httpx.Response(503, json={'error': 'unavailable'})
        return httpx.Response(200, json={'choices': [{'message': {'content': '{invalid json'},
                                                     'finish_reason': 'stop'}]})

    model._client = httpx.Client(transport=httpx.MockTransport(handler), trust_env=False)
    engine.run(state)
    assert len(requests) == model.requests == allowed
    assert state.status == 'limited'
    assert 'budget' in state.stop_reason.lower()
    assert state.tokens_used == model.total_input_tokens + model.total_output_tokens > 0
    assert state.estimated_cost_usd > 0
    assert state.report is not None
    assert 'stopped before a final model synthesis' in state.report.executive_summary
    assert engine.store.load(state.id).model_dump() == state.model_dump()


def test_completion_rejects_nonexistent_question_evidence_in_resumed_state(rig):
    engine, state = rig()
    ready_state(state)
    state.plan.questions[0].evidence_ids = ['nonexistent']
    engine.store.save(state)
    engine.state = engine.store.load(state.id)
    assert not engine._can_finish()


def test_mixed_hypothesis_without_valid_support_or_opposition_is_unresolved(rig):
    hypothesis = Hypothesis(id='H1', statement='Unsupported mixed claim', test='Check it', status='mixed',
                            supporting_evidence=['bad'], opposing_evidence=['also-bad'])
    engine, state = rig(ScriptedModel(review=review(hypotheses=[hypothesis])))
    state.plan = plan()
    engine._review()
    assert state.hypotheses[0].supporting_evidence == state.hypotheses[0].opposing_evidence == []
    assert state.hypotheses[0].status == 'unresolved'


def test_budget_guard_accepts_exact_reservation_and_rejects_one_more_token(rig):
    engine, state = rig(limits=Limits(max_tokens=1000, max_cost_usd=1))
    state.tokens_used = 400
    engine._guard_request(200, 400)
    with pytest.raises(BudgetExceeded, match='Insufficient token budget'):
        engine._guard_request(200, 401)
    assert engine.model.tasks == []


def test_budget_guard_accepts_exact_cost_and_rejects_more(rig):
    engine, state = rig(limits=Limits(max_cost_usd=0.5, input_usd_per_million=1000,
                                     output_usd_per_million=1000))
    state.estimated_cost_usd = 0.25
    engine._guard_request(125, 125)
    with pytest.raises(BudgetExceeded, match='Insufficient estimated cost budget'):
        engine._guard_request(125, 126)
    assert engine.model.tasks == []


def test_action_unknown_question_references_are_filtered_before_research(rig, monkeypatch):
    action = Action(action='research', rationale='Target questions', queries=['test query'],
                    question_ids=['Q1', 'nonexistent', 'Q1'])
    engine, state = rig(ScriptedModel(action=action), RecordingWeb(empty=True), Limits(max_iterations=1))
    state.plan = plan()
    seen = []
    original = engine._research

    def capture(proposal):
        seen.extend(proposal.question_ids)
        return original(proposal)

    monkeypatch.setattr(engine, '_research', capture)
    engine.run(state)
    assert seen == ['Q1']
