"""Deterministic report tests; no network or model calls."""
import pytest

from think_thank.models import (Evidence, Finding, Hypothesis, Limits, Plan,
                               Question, Report, ResearchState, Source)
from think_thank.reporting import render_report, valid_evidence


@pytest.fixture
def state():
    return ResearchState(
        id="run1", topic="Test topic", status="limited", stop_reason="token budget",
        limits=Limits(input_usd_per_million=2, output_usd_per_million=8),
        tokens_used=1234, estimated_cost_usd=0.0234, elapsed_seconds=12.5,
        iteration=3, counterchecks=2, searched_queries=["query"],
        sources=[Source(id="s1", title="A source", url="https://example.org/paper",
                        text="First line.\nExact quotation.\nLast line.", author="A. Author",
                        published_at="2025-01-01", retrieved_at="2026-11-16T12:00:00Z")],
        evidence=[Evidence(id="e1", source_id="s1", claim="A reported claim", quote="Exact quotation.")],
        hypotheses=[Hypothesis(id="h1", statement="A hypothesis", supporting_evidence=["e1"],
                               opposing_evidence=["missing"], status="supported", test="Find contrary evidence")],
        plan=Plan(scope="Topic", success_criteria=[], outline=[],
                  questions=[Question(id="q1", text="Unanswered question")]),
        gaps=["Sampling gap"],
        report=Report(title="A generated title", executive_summary="Model summary",
                      findings=[Finding(heading="Finding one", text="A model claim", evidence_ids=["e1"], confidence="high")],
                      competing_explanations=["Alternative account"], limitations=["Small sample"],
                      open_questions=["Next question"]),
    )


def test_readable_provenance_and_methodology(state):
    report = render_report(state)
    for content in ("# Test topic", "Stopped at a limit", "token budget", "Executive summary",
                    "Model summary", "Finding one", "[e1 → s1]", "Exact quotation.",
                    "2025\\-01\\-01", "2026\\-11\\-16T12:00:00Z", "A\\. Author",
                    "Supporting evidence", "Opposing evidence", "No valid", "Competing explanations",
                    "Alternative account", "Small sample", "Sampling gap", "Unanswered question",
                    "Next question", "Tokens used: 1234", "Counterchecks: 2", "Estimated cost",
                    "$0.023400", "User-configured", "input $2.000000", "output $8.000000",
                    "Not an actual invoice", "not independently verified"):
        # The citation fallback deliberately starts with lower-case no.
        assert content.lower() in report.lower()
    assert "# A generated title" not in report
    assert "````" not in report


def test_no_evidence_cannot_be_presented_as_verified(state):
    state.evidence = []
    output = render_report(state)
    assert "No verified research" in output
    assert "Unverified / unsupported model claim" in output
    assert "the summary has no claim-level citations" in output
    assert "[e1 → s1]" not in output


@pytest.mark.parametrize("change", ["missing-source", "empty", "whitespace", "fabricated", "duplicate-source", "duplicate-evidence"])
def test_invalid_evidence_never_cited(state, change):
    if change == "missing-source":
        state.evidence[0].source_id = "unknown"
    elif change == "empty":
        state.evidence[0].quote = ""
    elif change == "whitespace":
        state.evidence[0].quote = "\n"
    elif change == "fabricated":
        state.evidence[0].quote = "Never present in the original"
    elif change == "duplicate-source":
        state.sources.append(state.sources[0].model_copy())
    else:
        state.evidence.append(state.evidence[0].model_copy())
    assert valid_evidence(state) == {}
    output = render_report(state)
    assert "No verified research" in output
    assert "[e1 → s1]" not in output
    assert "excluded" in output


def test_mixed_valid_and_invalid_citations(state):
    state.report.findings[0].evidence_ids = ["e1", "unknown", "e1"]
    output = render_report(state)
    assert "Some requested citations are invalid" in output
    finding = output.split("## Evidence and provenance ledger")[0]
    assert finding.count("[e1 → s1]") == 1
    assert "[unknown" not in output


def test_metadata_and_generated_text_are_literal_markdown(state):
    attack = '<img src=x onerror=alert(1)> [click](javascript:alert(1))\n# injected'
    state.topic = attack
    state.sources[0].title = attack
    state.sources[0].author = attack
    state.sources[0].url = 'javascript:alert(1)'
    state.report.executive_summary = attack
    state.report.findings[0].text = attack
    output = render_report(state)
    assert '<img' not in output
    assert '&lt;img' in output
    assert '[click](javascript:' not in output
    assert '\n# injected' not in output
    assert '\\[click\\]' in output
    assert 'javascript:alert\\(1\\)' in output


def test_exact_quote_fence_cannot_break_out(state):
    quote = 'Line one\n```\n<script>alert(1)</script>\n`````\n  indented & <literal>'
    state.sources[0].text = quote
    state.evidence[0].quote = quote
    output = render_report(state)
    assert '``````text\n' + quote + '\n``````' in output
    # The raw HTML occurs only inside the six-backtick code fence.
    before, rest = output.split('``````text\n')
    quoted, after = rest.split('\n``````', 1)
    assert quoted == quote
    assert '<script>' not in before + after


@pytest.mark.parametrize("status, expected", [("completed", "Completed"), ("running", "In progress"), ("paused", "Paused"), ("failed", "Failed")])
def test_incomplete_reports_have_honest_defaults(status, expected):
    output = render_report(ResearchState(id="r", topic="Empty", status=status))
    assert '**Status:** ' + expected in output
    assert "No verified research" in output
    assert "No final report has been generated" in output
    assert "Not recorded" in output
    assert "Unknown" not in output  # No fabricated sources or metadata.


def test_unknown_metadata_and_interpretations_are_labeled(state):
    state.sources[0].author = None
    state.sources[0].published_at = None
    state.sources[0].truncated = True
    state.evidence[0].kind = "interpretation"
    output = render_report(state)
    assert "Author: Unknown" in output
    assert "Publication date: Unknown" in output
    assert "Source text truncated: Yes" in output
    assert "interpretation; source-reported, not independently verified" in output
