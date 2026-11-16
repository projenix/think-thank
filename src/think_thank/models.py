"""Validated, serializable research artifacts. No hidden reasoning is stored."""
from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class Limits(BaseModel):
    max_iterations: int = Field(default=12, ge=1, le=200)
    max_minutes: float = Field(default=15, gt=0)
    max_tokens: int = Field(default=100000, ge=1000)
    max_cost_usd: float = Field(default=3.0, gt=0)
    max_sources: int = Field(default=30, ge=1, le=500)
    max_reads_per_iteration: int = Field(default=3, ge=1, le=10)
    input_usd_per_million: float = Field(default=1.0, ge=0)
    output_usd_per_million: float = Field(default=4.0, ge=0)


class Question(BaseModel):
    id: str
    text: str
    status: Literal['open', 'partial', 'answered'] = 'open'
    evidence_ids: List[str] = Field(default_factory=list)


class Source(BaseModel):
    id: str
    url: str
    title: str
    text: str
    retrieved_at: str = Field(default_factory=utcnow)
    author: Optional[str] = None
    published_at: Optional[str] = None
    content_hash: str = ''
    truncated: bool = False


class SearchResult(BaseModel):
    url: str
    title: str
    snippet: str = ''


class Evidence(BaseModel):
    id: str
    claim: str
    source_id: str
    quote: str
    kind: Literal['reported_fact', 'interpretation', 'hypothesis'] = 'reported_fact'
    limitations: str = ''


class Hypothesis(BaseModel):
    id: str
    statement: str
    supporting_evidence: List[str] = Field(default_factory=list)
    opposing_evidence: List[str] = Field(default_factory=list)
    status: Literal['untested', 'supported', 'mixed', 'weakened', 'unresolved'] = 'untested'
    test: str


class Decision(BaseModel):
    iteration: int
    action: str
    rationale: str
    at: str = Field(default_factory=utcnow)


class Plan(BaseModel):
    scope: str
    assumptions: List[str] = Field(default_factory=list)
    success_criteria: List[str]
    questions: List[Question] = Field(min_length=1)
    outline: List[str]


class Action(BaseModel):
    action: Literal['research', 'finish']
    rationale: str
    queries: List[str] = Field(default_factory=list, max_length=3)
    question_ids: List[str] = Field(default_factory=list)
    disconfirming: bool = False


class Extraction(BaseModel):
    evidence: List[Evidence] = Field(default_factory=list)


class Review(BaseModel):
    questions: List[Question]
    hypotheses: List[Hypothesis]
    outline: List[str]
    gaps: List[str]
    next_priority: str
    ready: bool = False


class Finding(BaseModel):
    heading: str
    text: str
    evidence_ids: List[str] = Field(default_factory=list)
    confidence: Literal['low', 'moderate', 'high'] = 'low'


class Report(BaseModel):
    title: str
    executive_summary: str
    findings: List[Finding]
    competing_explanations: List[str]
    limitations: List[str]
    open_questions: List[str]


class ResearchState(BaseModel):
    id: str
    topic: str
    created_at: str = Field(default_factory=utcnow)
    updated_at: str = Field(default_factory=utcnow)
    status: Literal['running', 'paused', 'completed', 'limited', 'failed'] = 'running'
    limits: Limits = Field(default_factory=Limits)
    plan: Optional[Plan] = None
    sources: List[Source] = Field(default_factory=list)
    evidence: List[Evidence] = Field(default_factory=list)
    hypotheses: List[Hypothesis] = Field(default_factory=list)
    decisions: List[Decision] = Field(default_factory=list)
    gaps: List[str] = Field(default_factory=list)
    outline: List[str] = Field(default_factory=list)
    searched_queries: List[str] = Field(default_factory=list)
    iteration: int = 0
    tokens_used: int = 0
    estimated_cost_usd: float = 0
    elapsed_seconds: float = 0
    no_progress_iterations: int = 0
    counterchecks: int = 0
    processed_source_ids: List[str] = Field(default_factory=list)
    review_ready: bool = False
    stop_reason: str = ''
    report: Optional[Report] = None
    errors: List[str] = Field(default_factory=list)
    config: Dict[str, str] = Field(default_factory=dict)
