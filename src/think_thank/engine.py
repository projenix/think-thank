"""Bounded research state machine. All writes belong to this orchestrator."""
import json
import time
import uuid
from typing import Callable, Optional

from .llm import ModelError
from .models import (Action, Decision, Evidence, Extraction, Finding, Limits, Plan,
                     Report, ResearchState, Review)
from .web import WebError


class BudgetExceeded(RuntimeError):
    pass


class ResearchEngine:
    def __init__(self, model, web, store, progress: Optional[Callable[[str], None]] = None,
                 should_stop: Optional[Callable[[], bool]] = None):
        self.model, self.web, self.store = model, web, store
        self.progress = progress or (lambda message: None)
        self.should_stop = should_stop or (lambda: False)
        self.state = None
        self._started = 0.0
        self._elapsed_before = 0.0
        self._accounted_input = model.total_input_tokens
        self._accounted_output = model.total_output_tokens
        # Checked before every HTTP attempt, including repairs and retries.
        self.model.before_request = self._guard_request

    def create(self, topic: str, limits: Optional[Limits] = None, config=None) -> ResearchState:
        topic = topic.strip()
        if not topic or len(topic) > 10000:
            raise ValueError('Topic must contain 1–10,000 characters.')
        state = ResearchState(id=uuid.uuid4().hex[:16], topic=topic,
                              limits=limits or Limits(), config=config or {})
        self.store.save(state)
        return state

    def _sync(self):
        s = self.state
        incoming = self.model.total_input_tokens - self._accounted_input
        outgoing = self.model.total_output_tokens - self._accounted_output
        s.tokens_used += max(0, incoming) + max(0, outgoing)
        s.estimated_cost_usd += (max(0, incoming) * s.limits.input_usd_per_million
                                 + max(0, outgoing) * s.limits.output_usd_per_million) / 1e6
        self._accounted_input = self.model.total_input_tokens
        self._accounted_output = self.model.total_output_tokens
        s.elapsed_seconds = self._elapsed_before + time.monotonic() - self._started

    def _check_pause(self):
        if self.should_stop():
            raise KeyboardInterrupt

    def _guard_request(self, input_reserve: int, output_reserve: int):
        self._check_pause()
        self._sync()
        s, limits = self.state, self.state.limits
        if s.elapsed_seconds >= limits.max_minutes * 60:
            raise BudgetExceeded('Runtime budget reached.')
        if s.tokens_used + input_reserve + output_reserve > limits.max_tokens:
            raise BudgetExceeded('Insufficient token budget for the next model request (conservative reservation).')
        estimated = (input_reserve * limits.input_usd_per_million
                     + output_reserve * limits.output_usd_per_million) / 1e6
        if s.estimated_cost_usd + estimated > limits.max_cost_usd:
            raise BudgetExceeded('Insufficient estimated cost budget for the next model request.')

    def _save(self):
        self._sync()
        self.store.save(self.state)

    def _call(self, task, context, schema):
        # A generic guard also supports injected test/local model adapters.
        self._guard_request(len(json.dumps(context).encode('utf-8')) + 2000, 3000)
        self.progress('Model: ' + task)
        try:
            return self.model.complete(task, context, schema, max_tokens=3000)
        finally:
            self._save()

    def _context(self):
        s = self.state
        # Source bodies only enter the per-source extractor. The planner sees
        # bounded metadata and the evidence ledger, not an ever-growing transcript.
        return {'topic': s.topic, 'plan': s.plan.model_dump() if s.plan else None,
                'evidence': [e.model_dump() for e in s.evidence],
                'sources': [x.model_dump(exclude={'text'}) for x in s.sources],
                'hypotheses': [h.model_dump() for h in s.hypotheses],
                'outline': s.outline, 'gaps': s.gaps,
                'recent_decisions': [d.model_dump() for d in s.decisions[-5:]],
                'searched_queries': s.searched_queries[-30:],
                'iteration': s.iteration, 'counterchecks': s.counterchecks,
                'limits': s.limits.model_dump(), 'tokens_used': s.tokens_used,
                'stop_reason': s.stop_reason, 'status': s.status}

    def _limit_reason(self, include_research=True):
        self._check_pause()
        self._sync()
        s = self.state
        if s.elapsed_seconds >= s.limits.max_minutes * 60:
            return 'Runtime budget reached.'
        if s.tokens_used >= s.limits.max_tokens:
            return 'Token budget reached.'
        if s.estimated_cost_usd >= s.limits.max_cost_usd:
            return 'Estimated cost budget reached.'
        if include_research:
            if s.iteration >= s.limits.max_iterations:
                return 'Iteration limit reached.'
            if len(s.sources) >= s.limits.max_sources:
                return 'Source limit reached.'
            if s.no_progress_iterations >= 3:
                return 'Diminishing returns: three iterations without new evidence.'
        return ''

    def _can_finish(self):
        s = self.state
        from .reporting import valid_evidence
        valid = valid_evidence(s)
        return bool(s.plan and valid and s.counterchecks and s.review_ready
                    and all(q.status == 'answered' and q.evidence_ids
                            and all(e in valid for e in q.evidence_ids) for q in s.plan.questions))

    def _extract_pending(self):
        s = self.state
        for source in s.sources:
            if source.id in s.processed_source_ids:
                continue
            result = self._call('extract', {'topic': s.topic, 'source': source.model_dump(),
                                'questions': [q.model_dump() for q in s.plan.questions],
                                'existing_evidence_ids': [e.id for e in s.evidence]}, Extraction)
            existing = {(e.source_id, e.claim, e.quote) for e in s.evidence}
            for item in result.evidence:
                if item.source_id != source.id or not item.quote.strip() or item.quote not in source.text:
                    s.errors.append('Rejected evidence with an invalid source or non-verbatim quotation.')
                    continue
                key = (item.source_id, item.claim, item.quote)
                if key in existing:
                    continue
                item.id = 'E%d' % (len(s.evidence) + 1)
                s.evidence.append(item)
                existing.add(key)
            s.processed_source_ids.append(source.id)
            self._save()

    def _research(self, action):
        s = self.state
        found_new_source = False
        attempted_queries = False
        known_urls = {x.url for x in s.sources}
        known_hashes = {x.content_hash for x in s.sources if x.content_hash}
        reads = 0
        attempted_urls = set()
        for query in action.queries:
            query = query.strip()[:500]
            if not query or query.casefold() in {q.casefold() for q in s.searched_queries}:
                continue
            if self._limit_reason(False):
                raise BudgetExceeded(self._limit_reason(False))
            attempted_queries = True
            self.progress('Search: ' + query)
            try:
                results = self.web.search(query, limit=5)
            except WebError as exc:
                s.errors.append('Search failed: ' + str(exc))
                self._save()
                continue
            s.searched_queries.append(query)
            self._save()
            for result in results:
                if reads >= s.limits.max_reads_per_iteration or len(s.sources) >= s.limits.max_sources:
                    break
                if result.url in known_urls or result.url in attempted_urls:
                    continue
                if self._limit_reason(False):
                    raise BudgetExceeded(self._limit_reason(False))
                reads += 1
                attempted_urls.add(result.url)
                self.progress('Read: ' + result.url)
                try:
                    source = self.web.read(result.url, 'S%d' % (len(s.sources) + 1))
                except WebError as exc:
                    s.errors.append('Read failed: ' + str(exc))
                    self._save()
                    continue
                if source.url in known_urls or source.content_hash and source.content_hash in known_hashes:
                    continue
                known_urls.add(source.url)
                if source.content_hash:
                    known_hashes.add(source.content_hash)
                s.sources.append(source)
                found_new_source = True
                self._save()
                self._extract_pending()
            if reads >= s.limits.max_reads_per_iteration:
                break
        # This records an executed disconfirming search/read, not proof of independence.
        if action.disconfirming and attempted_queries and found_new_source:
            s.counterchecks += 1
        self._save()

    def _review(self):
        s = self.state
        result = self._call('review', self._context(), Review)
        valid = {e.id for e in s.evidence}
        proposed = {q.id: q for q in result.questions}
        for question in s.plan.questions:
            if question.id not in proposed:
                continue
            update = proposed[question.id]
            question.evidence_ids = list(dict.fromkeys(e for e in update.evidence_ids if e in valid))
            question.status = update.status
            if question.status == 'answered' and not question.evidence_ids:
                question.status = 'partial'
        hypotheses = {h.id: h for h in s.hypotheses}
        for hypothesis in result.hypotheses:
            hypothesis.supporting_evidence = list(dict.fromkeys(e for e in hypothesis.supporting_evidence if e in valid))
            hypothesis.opposing_evidence = list(dict.fromkeys(e for e in hypothesis.opposing_evidence if e in valid))
            support, opposition = hypothesis.supporting_evidence, hypothesis.opposing_evidence
            if support and opposition:
                hypothesis.status = 'mixed'
            elif (hypothesis.status == 'supported' and not support
                  or hypothesis.status == 'weakened' and not opposition
                  or hypothesis.status == 'mixed' and not (support and opposition)):
                hypothesis.status = 'unresolved'
            hypotheses[hypothesis.id] = hypothesis
        s.hypotheses = list(hypotheses.values())
        s.outline = result.outline
        s.gaps = result.gaps
        s.review_ready = result.ready
        s.decisions.append(Decision(iteration=s.iteration, action='review', rationale=result.next_priority))
        self._save()

    def _fallback_report(self):
        s = self.state
        return Report(title=s.topic, executive_summary='Research stopped before a final model synthesis could be produced. '
                      'The entries below are source-attributed observations, not independently verified conclusions.',
                      findings=[Finding(heading=e.id, text=e.claim, evidence_ids=[e.id], confidence='low')
                                for e in s.evidence],
                      competing_explanations=[h.statement for h in s.hypotheses],
                      limitations=[s.stop_reason, 'Exact quotation checks verify provenance, not truth or entailment.'],
                      open_questions=[q.text for q in s.plan.questions if q.status != 'answered'] if s.plan else [s.topic])

    def _finalize(self):
        s = self.state
        if self._limit_reason(False):
            s.report = self._fallback_report()
        else:
            try:
                report = self._call('report', self._context(), Report)
                valid = {e.id for e in s.evidence}
                for finding in report.findings:
                    original = list(finding.evidence_ids)
                    finding.evidence_ids = list(dict.fromkeys(e for e in original if e in valid))
                    if not finding.evidence_ids or len(finding.evidence_ids) != len(set(original)):
                        finding.confidence = 'low'
                        report.limitations.append('A finding lacked valid supporting citations: ' + finding.heading)
                report.limitations.append('Provenance checks do not establish truth, entailment, or source independence.')
                s.report = report
            except (ModelError, BudgetExceeded) as exc:
                s.errors.append(str(exc))
                s.report = self._fallback_report()
        self._save()

    def run(self, state: ResearchState) -> ResearchState:
        if state.status == 'completed':
            return state
        self.state = state
        self._started = time.monotonic()
        self._elapsed_before = state.elapsed_seconds
        state.status = 'running'
        state.stop_reason = ''
        state.report = None
        self._save()
        try:
            if not state.plan:
                state.plan = self._call('plan', {'topic': state.topic}, Plan)
                # IDs are controlled by the application, not free-form model paths.
                for index, question in enumerate(state.plan.questions, 1):
                    question.id = 'Q%d' % index
                    question.status, question.evidence_ids = 'open', []
                state.outline = state.plan.outline
                self._save()
            self._extract_pending()
            while True:
                if self._can_finish():
                    state.status, state.stop_reason = 'completed', 'Coverage and countercheck completion criteria met.'
                    break
                reason = self._limit_reason()
                if reason:
                    state.status, state.stop_reason = 'limited', reason
                    break
                state.iteration += 1
                self._save()
                action = self._call('action', self._context(), Action)
                if action.action == 'finish':
                    # A model cannot self-certify completion without evidence.
                    action = Action(action='research', rationale='Completion gate not met; check missing evidence and alternatives.',
                                    queries=[state.topic[:400] + ' conflicting evidence limitations'], disconfirming=True)
                if not state.counterchecks and state.iteration % 2 == 0:
                    # Ensure the loop actually attempts disconfirmation, even when
                    # the model keeps proposing confirmatory searches.
                    action.disconfirming = True
                    action.queries = [state.topic[:350] + ' contrary evidence limitations alternative explanations'] + action.queries[:2]
                valid_questions = {q.id for q in state.plan.questions}
                action.question_ids = list(dict.fromkeys(q for q in action.question_ids if q in valid_questions))
                state.decisions.append(Decision(iteration=state.iteration, action=action.action, rationale=action.rationale))
                self._save()
                before = len(state.evidence)
                self._research(action)
                self._review()
                state.no_progress_iterations = 0 if len(state.evidence) > before else state.no_progress_iterations + 1
                self._save()
        except BudgetExceeded as exc:
            state.status, state.stop_reason = 'limited', str(exc)
        except ModelError as exc:
            state.status, state.stop_reason = 'failed', str(exc)
            state.errors.append(str(exc))
        except KeyboardInterrupt:
            state.status, state.stop_reason = 'paused', 'Interrupted by user; resume this session to continue.'
            self._save()
            return state
        # A second Ctrl-C during synthesis still leaves a resumable checkpoint.
        try:
            self._finalize()
        except KeyboardInterrupt:
            state.status, state.stop_reason = 'paused', 'Interrupted during report synthesis.'
            self._save()
        return state
