"""Plain Markdown reports; source text and model output are always untrusted.

A citation validates provenance (an exact excerpt exists), not the truth of a
claim or whether the excerpt entails a model's interpretation. No HTML is emitted.
"""
import html
import re
from collections import Counter
from typing import Dict, List

from .models import Evidence, ResearchState


def _text(value: object) -> str:
    """Render untrusted content as literal, single-line Markdown text."""
    text = " ".join(str(value).splitlines())
    text = "".join(c for c in text if ord(c) >= 32 or c == "\t")
    text = html.escape(text, quote=True)
    return re.sub(r"([\\`*_{}\[\]()#+.!|~\-])", r"\\\1", text)


def _quote(value: str) -> str:
    # A fence longer than any run in the excerpt prevents fence breakout, and
    # code blocks preserve exact quotation characters without interpreting HTML.
    longest = max((len(m.group()) for m in re.finditer(r"`+", value)), default=0)
    fence = "`" * max(3, longest + 1)
    return fence + "text\n" + value + "\n" + fence


def valid_evidence(state: ResearchState) -> Dict[str, Evidence]:
    """Exclude dangling/ambiguous IDs and quotations absent from stored text."""
    source_counts = Counter(s.id for s in state.sources)
    evidence_counts = Counter(e.id for e in state.evidence)
    sources = {s.id: s for s in state.sources if source_counts[s.id] == 1}
    return {
        e.id: e for e in state.evidence
        if e.id and evidence_counts[e.id] == 1 and e.source_id in sources
        and e.quote.strip() and e.quote in sources[e.source_id].text
    }


def render_report(state: ResearchState) -> str:
    """Render an auditable report even for interrupted or evidence-free sessions."""
    evidence = valid_evidence(state)

    def citations(ids: List[str]) -> str:
        refs = list(dict.fromkeys(ids))
        return "; ".join(
            "[{} → {}]".format(_text(eid), _text(evidence[eid].source_id))
            for eid in refs if eid in evidence
        ) or "None (no valid evidence citations)"

    status = {
        "completed": "Completed", "limited": "Stopped at a limit",
        "running": "In progress", "paused": "Paused", "failed": "Failed",
    }[state.status]
    lines = ["# " + _text(state.topic), "", "**Status:** " + status,
             "**Stop reason:** " + _text(state.stop_reason or "Not recorded"), "",
             "## Executive summary", ""]
    if not evidence:
        lines += ["**No verified research:** no evidence has a nonempty exact quotation "
                  "in a uniquely identified stored source. Claims below are unverified.", ""]
    else:
        lines += ["Citations establish traceable quotations, not independent verification "
                  "of source accuracy or proof of a model's claims.", ""]
    report = state.report
    if report:
        lines += ["**Model-written synthesis (not independently verified; the summary "
                  "has no claim-level citations):**", _text(report.executive_summary), ""]
    else:
        lines += ["No final report has been generated.", ""]
    lines += ["## Findings", ""]
    if report and report.findings:
        for finding in report.findings:
            good = [eid for eid in finding.evidence_ids if eid in evidence]
            lines += ["### " + _text(finding.heading), "",
                      "**{}**".format("Source-linked model synthesis; not independently verified"
                                     if good else "Unverified / unsupported model claim"),
                      _text(finding.text),
                      "- Model-assessed confidence: " + _text(finding.confidence),
                      "- Evidence → source: " + citations(finding.evidence_ids)]
            if any(eid not in evidence for eid in finding.evidence_ids):
                lines += ["- Some requested citations are invalid and were excluded; "
                          "the remaining citations do not establish support for every claim."]
            lines += [""]
    else:
        lines += ["No findings recorded.", ""]
    lines += ["## Evidence and provenance ledger", ""]
    sources = {s.id: s for s in state.sources}
    for eid, item in evidence.items():
        source = sources[item.source_id]
        lines += ["### " + _text(eid) + " → " + _text(source.id), "",
                  "- Claim ({}; source-reported, not independently verified): {}".format(
                      _text(item.kind), _text(item.claim)),
                  "- Source title: " + _text(source.title),
                  "- URL (literal text): " + _text(source.url),
                  "- Author: " + _text(source.author or "Unknown"),
                  "- Publication date: " + _text(source.published_at or "Unknown"),
                  "- Retrieved at: " + _text(source.retrieved_at),
                  "- Content hash: " + _text(source.content_hash or "Not recorded"),
                  "- Source text truncated: " + ("Yes" if source.truncated else "No"),
                  "- Evidence limitations: " + _text(item.limitations or "None recorded"),
                  "", "Exact quotation from stored source text:", "", _quote(item.quote), ""]
    if not evidence:
        lines += ["No valid evidence citations.", ""]
    invalid = len(state.evidence) - len(evidence)
    if invalid:
        lines += ["{} evidence item(s) excluded: missing/ambiguous identifiers, empty "
                  "quotations, or quotations not found exactly in source text.".format(invalid), ""]
    lines += ["## Hypotheses: support and opposition", ""]
    for hypothesis in state.hypotheses:
        lines += ["### " + _text(hypothesis.id), "", _text(hypothesis.statement),
                  "- Model-assessed status (not independently verified): " + _text(hypothesis.status),
                  "- Supporting evidence → source: " + citations(hypothesis.supporting_evidence),
                  "- Opposing evidence → source: " + citations(hypothesis.opposing_evidence),
                  "- Proposed test: " + _text(hypothesis.test), ""]
    if not state.hypotheses:
        lines += ["No hypotheses recorded.", ""]
    for title, items in [
        ("Competing explanations (unverified alternatives)", report.competing_explanations if report else []),
        ("Limitations", (report.limitations if report else []) + state.gaps),
        ("Open questions", (report.open_questions if report else []) +
         ([q.text for q in state.plan.questions if q.status != "answered"] if state.plan else [])),
    ]:
        lines += ["## " + title, ""]
        lines += ["- " + _text(item) for item in dict.fromkeys(items)] or ["None recorded (not proof that none exist)."]
        lines += [""]
    limits = state.limits
    lines += ["## Methodology and resource usage", "",
              "Evidence was checked for exact quotation presence in stored text. "
              "Search snippets are not evidence. Citation checks do not assess entailment, "
              "independence, bias, or factual accuracy. Model confidence is not calibrated.", "",
              "- Iterations: {} / {}".format(state.iteration, limits.max_iterations),
              "- Sources retained: {} / {}".format(len(state.sources), limits.max_sources),
              "- Valid evidence items: {} / {}".format(len(evidence), len(state.evidence)),
              "- Search queries: {}".format(len(state.searched_queries)),
              "- Counterchecks: {}".format(state.counterchecks),
              "- Tokens used: {} / {}".format(state.tokens_used, limits.max_tokens),
              "- Elapsed seconds: {:.2f} (time limit: {} minutes)".format(state.elapsed_seconds, limits.max_minutes),
              "- Estimated cost (USD): ${:.6f}; budget: ${:.6f}. Not an actual invoice.".format(
                  state.estimated_cost_usd, limits.max_cost_usd),
              "- User-configured estimation rates (USD per million tokens): input ${:.6f}; output ${:.6f}.".format(
                  limits.input_usd_per_million, limits.output_usd_per_million),
              "- Cost estimation uses input/output token counts and these configured rates; "
              "provider billing and search fees may differ.",
              "- Stop reason: " + _text(state.stop_reason or "Not recorded"), ""]
    return "\n".join(lines)
