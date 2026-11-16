"""Deterministic offline adapters, not a model or a real research service.

Every source and outcome below is fictional demonstration data. example.org
URLs are identifiers only: these adapters never resolve or fetch them. Token
counters are tiny simulated units for exercising the engine, not billed usage.
"""
import hashlib
from typing import Type

from pydantic import BaseModel

from .llm import ModelError
from .models import (Action, Evidence, Extraction, Finding, Hypothesis, Plan,
                     Question, Report, Review, SearchResult, Source)
from .web import WebError

DEMO_TOPIC = 'Do four-day workweeks improve productivity?'
NOTICE = 'Synthetic, fictional demonstration data; not real research.'
_OUTLINE = ['Synthetic demonstration only', 'Productivity signals',
            'Counterchecks and competing explanations', 'Limitations']
# The first two fixtures support the hypothesis; the other two challenge it.
_FIXTURES = (
    ('pilot', 'Fictional demonstration: office pilot',
     'In this fictional office pilot, weekly output rose from 100 to 108 units after adopting a four-day week.',
     'The synthetic office pilot reports higher weekly output.'),
    ('focus', 'Fictional demonstration: focus-time diary',
     'In this fictional team diary, fewer meetings accompanied unchanged weekly output in four days rather than five.',
     'The synthetic diary reports unchanged weekly output with fewer workdays.'),
    ('coverage', 'Fictional demonstration: coverage counterexample',
     'In this fictional service-team trial, weekly output fell from 100 to 94 units when a four-day schedule left coverage gaps.',
     'The synthetic service trial reports lower output under coverage constraints.'),
    ('selection', 'Fictional demonstration: methodological limitations',
     'This fictional audit found that volunteer teams also changed their workflow, so the schedule alone could not explain their output gains.',
     'The synthetic audit identifies selection and workflow changes as competing explanations.'),
)
_URLS = tuple('https://example.org/fictional-demo/' + row[0] for row in _FIXTURES)


class DemoWeb:
    """An in-memory fixture lookup with the WebClient search/read/close API."""

    def __init__(self):
        self._closed = False

    def _check_open(self):
        if self._closed:
            raise WebError('Demo web adapter is closed.')

    def search(self, query: str, limit: int = 5):
        self._check_open()
        contrary = any(word in query.casefold() for word in
                       ('contrary', 'limitations', 'conflicting', 'countercheck'))
        indices = (2, 3) if contrary else (0, 1)
        return [SearchResult(url=_URLS[i], title=_FIXTURES[i][1],
                             snippet=NOTICE + ' ' + _FIXTURES[i][2])
                for i in indices][:max(0, limit)]

    def read(self, url: str, source_id: str):
        self._check_open()
        if url not in _URLS:
            raise WebError('Unknown fictional demo URL; no network request was made.')
        row = _FIXTURES[_URLS.index(url)]
        text = NOTICE + '\n' + row[1] + '\n' + row[2] + '\n'
        return Source(id=source_id, url=url, title=row[1], text=text,
                      author='Fictional demonstration fixture',
                      retrieved_at='2000-01-01T00:00:00+00:00',
                      content_hash=hashlib.sha256(text.encode('utf-8')).hexdigest())

    def close(self):
        self._closed = True


def _evidence_groups(context):
    """Use supplied IDs, never assume that the engine assigned E1, E2, etc."""
    groups = []
    for rows in (_FIXTURES[:2], _FIXTURES[2:]):
        quotes = {row[2] for row in rows}
        groups.append(list(dict.fromkeys(
            e['id'] for e in context.get('evidence', [])
            if e.get('id') and e.get('quote') in quotes)))
    return groups


class DemoModel:
    """Scripted task responses; repeatable from context, including after resume."""

    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.before_request = None
        self._closed = False

    def close(self):
        self._closed = True

    def complete(self, task: str, context: dict, schema: Type[BaseModel],
                 max_tokens: int = 4000) -> BaseModel:
        tasks = {'plan': Plan, 'action': Action, 'extract': Extraction,
                 'review': Review, 'report': Report}
        if self._closed:
            raise ModelError('Demo model is closed.')
        if task not in tasks or schema is not tasks[task] or not isinstance(context, dict):
            raise ModelError('Unknown demo task, context, or schema.')
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens <= 0:
            raise ModelError('max_tokens must be a positive integer.')
        # Simulated accounting only. Honor the same pre-request hook as ModelClient.
        if self.before_request is not None:
            self.before_request(1, 1)
        self.total_input_tokens += 1
        self.total_output_tokens += 1
        result = getattr(self, '_' + task)(context)
        return schema.model_validate(result.model_dump())

    def _plan(self, context):
        return Plan(
            scope=NOTICE + ' Fixed example topic: ' + DEMO_TOPIC,
            assumptions=[NOTICE, 'Productivity means weekly output, not just output per hour.'],
            success_criteria=['Trace positive and contrary synthetic observations to exact quotes.',
                              'Complete two research passes and a countercheck; do not infer real-world effects.'],
            questions=[Question(id='Q1', text='What productivity benefits appear in the fictional examples?'),
                       Question(id='Q2', text='What fictional counterexamples and limitations challenge those benefits?')],
            outline=list(_OUTLINE))

    def _action(self, context):
        contrary = context.get('iteration', 1) >= 2
        questions = (context.get('plan') or {}).get('questions', [])
        return Action(action='research', disconfirming=contrary,
                      rationale=('Countercheck the fictional gains with different synthetic sources.' if contrary
                                 else 'First gather confirmatory fictional productivity observations.'),
                      queries=[DEMO_TOPIC + (' contrary evidence limitations' if contrary else ' productivity benefits')],
                      question_ids=[q['id'] for q in questions])

    def _extract(self, context):
        source = context.get('source', {})
        used = set(context.get('existing_evidence_ids', []))
        evidence = []
        for url, row in zip(_URLS, _FIXTURES):
            if (source.get('url') != url or not source.get('id')
                    or row[2] not in source.get('text', '')):
                continue
            number = 1
            while 'E%d' % number in used:
                number += 1
            eid = 'E%d' % number
            used.add(eid)
            evidence.append(Evidence(id=eid, source_id=source['id'], claim=row[3],
                                     quote=row[2], kind='reported_fact', limitations=NOTICE))
        return Extraction(evidence=evidence)

    def _review(self, context):
        support, opposition = _evidence_groups(context)
        ready = (context.get('iteration', 0) >= 2 and context.get('counterchecks', 0) > 0
                 and len(support) >= 2 and len(opposition) >= 2)
        questions = []
        for index, original in enumerate((context.get('plan') or {}).get('questions', [])):
            ids = support if index == 0 else opposition
            questions.append(Question(id=original['id'], text=original['text'],
                                      status='answered' if len(ids) >= 2 else ('partial' if ids else 'open'),
                                      evidence_ids=ids))
        previous = context.get('hypotheses', [])
        hypothesis_id = previous[0]['id'] if previous else 'H1'
        return Review(
            questions=questions,
            hypotheses=[Hypothesis(
                id=hypothesis_id,
                statement='In the fictional examples, a four-day week can preserve or improve weekly output, but not universally.',
                supporting_evidence=support, opposing_evidence=opposition,
                status='mixed' if support and opposition else 'unresolved',
                test='For real research, compare matched teams with stable workflows and coverage, measuring weekly output and hours.')],
            outline=list(_OUTLINE),
            gaps=[NOTICE] + ([] if ready else ['Need both positive fixtures and contrary fixtures before completion.']),
            next_priority=('Conclude the synthetic demonstration without making real-world claims.' if ready
                           else 'Seek contrary evidence and limitations from different fictional sources.'),
            ready=bool(ready and len(questions) == 2))

    def _report(self, context):
        support, opposition = _evidence_groups(context)
        findings = []
        if support:
            findings.append(Finding(
                heading='Fictional positive observations', confidence='low', evidence_ids=support,
                text='The synthetic office and diary examples illustrate possible output benefits, not evidence of real effects.'))
        if opposition:
            findings.append(Finding(
                heading='Fictional counterchecks', confidence='low', evidence_ids=opposition,
                text='The synthetic coverage and audit examples illustrate why gains need not generalize or be caused by the schedule.'))
        limitations = [NOTICE, 'All example.org URLs identify local fixtures; no sites were fetched.',
                       'Neither real-world effect sizes nor causal conclusions can be inferred.',
                       'Token and cost accounting is simulated; no provider was billed.']
        if len(support) < 2 or len(opposition) < 2:
            limitations.append('The synthetic demonstration stopped with incomplete evidence coverage.')
        if context.get('status') in ('limited', 'failed', 'paused'):
            limitations.append('Research stopped early: ' + context.get('stop_reason', 'unspecified limit'))
        return Report(
            title='Synthetic demonstration: ' + DEMO_TOPIC,
            executive_summary=NOTICE + ' This offline example demonstrates citation tracking, not an answer from real research. '
                              'The scripted observations suggest conditional benefits alongside counterexamples; '
                              'whether four-day workweeks improve real productivity remains unanswered here.',
            findings=findings,
            competing_explanations=['In the fictional scenario, fewer meetings rather than shorter weeks may explain gains.',
                                    'Volunteer selection and service coverage could change the fictional outcomes.'],
            limitations=limitations,
            open_questions=['What do real, independently verified studies find across sectors and longer follow-up periods?'])
