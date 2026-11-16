# Think Thank

An autonomous, evidence-driven research agent. Give it a topic; it plans an investigation, searches and reads public web pages, extracts attributable evidence, revises hypotheses, seeks counter-evidence, and produces a structured report. Research state is persisted throughout, not just at the end.

**Included:** a local web app and Python CLI, OpenAI-compatible model API, DuckDuckGo or Tavily search, SQLite checkpoints, Markdown/JSON reports, and an entirely offline demonstration.

## Web app

```bash
source .venv/bin/activate
think-thank serve
```

Open **http://127.0.0.1:8000**. Start a demo without credentials, or choose live research once the server is configured. The app includes:

- Session history and a research-topic form with provider and budget controls.
- Live activity, real usage counters, saved questions, outline, and decision summaries.
- Evidence/source inspection, exact quotations, hypothesis support/opposition, and proposed tests.
- Structured final reports, clickable evidence references, and Markdown/JSON downloads.
- Cooperative pause and resume, with editable cumulative budgets.

Progress refreshes every second while research is active. Pause takes effect at the next safe boundary; it does not forcibly interrupt an in-flight HTTP request. Demo runs use scripted fictional four-day-workweek fixtures regardless of the topic you type; they demonstrate the interface, not real research.

For live research, set `OPENAI_API_KEY` (and optionally `TAVILY_API_KEY`, `OPENAI_BASE_URL`, `THINK_THANK_MODEL`) in the terminal **before starting the server**. Restart the server after changing its environment. Credentials are never entered in the browser. Provider endpoint URLs are server-controlled. The UI displays missing-credential guidance.

```bash
export OPENAI_API_KEY='your-key'
think-thank serve --port 8000
# Different storage or port:
think-thank --data-dir /path/to/research serve --port 8001
```

The web server binds only to loopback. This is a **single-user local app**, not an authenticated public service: do not expose it through a tunnel or reverse proxy. Run one web server process per data directory and avoid using CLI operations on sessions actively controlled by the web app. Up to three sessions can run concurrently. Research checkpoints survive restarts; the recent activity timeline is process-local. Interrupted/stale sessions are resumable, not automatically restarted. Stop the foreground server with Ctrl-C.

## Quick start

Requires Python **3.9+** (3.11+ recommended).

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e '.[dev]'

# No credentials or network needed. All demo evidence is explicitly fictional.
think-thank demo
```

For real research:

```bash
export OPENAI_API_KEY='your-key'
think-thank research 'What evidence supports a four-day workweek without reducing productivity?'
```

The default model is `gpt-4.1-mini`. Override it for your provider:

```bash
export THINK_THANK_MODEL='your-model-name'
export OPENAI_BASE_URL='https://your-provider.example/v1'
think-thank research 'Your research question' --max-iterations 16 --max-minutes 20
```

The endpoint must support OpenAI-compatible chat completions and `response_format: {"type": "json_object"}`. Model availability and prices depend on your provider. For a local compatible server, HTTP is allowed only on loopback; an API key is optional there.

### Search providers

DuckDuckGo HTML search is the default and needs no key. It can throttle requests or present a challenge; such failures are recorded, not bypassed. Tavily is an optional alternative:

```bash
export TAVILY_API_KEY='your-key'
think-thank research 'Your question' --search tavily
```

`THINK_THANK_SEARCH` can set the default. Search snippets guide discovery only: evidence must come from a successfully retrieved source body.

### Budgets

```bash
think-thank research 'Your question' \
  --max-iterations 12 \
  --max-minutes 15 \
  --max-tokens 100000 \
  --max-cost 3 \
  --max-sources 30 \
  --reads-per-iteration 3 \
  --input-price 1 \
  --output-price 4
```

These are the defaults. **Prices are configurable estimates in USD per million tokens, not current provider price claims.** Set them to your provider's actual rates. Search API fees, caching discounts, special reasoning-token charges, and other provider-specific billing are not calculated. A provider-side spending limit is the billing backstop.

The engine conservatively reserves input/output capacity before each model HTTP attempt, including retries and JSON repairs. Missing usage is charged conservatively; byte-based reservations can stop research before the nominal token limit. Runtime is checked between operations, not a hard process deadline: an in-flight network operation, retry delay, or PDF parse can finish after the time limit. A bounded, local evidence report is generated if there is insufficient budget for final model synthesis.

### Inspect and resume

```bash
think-thank list
think-thank show SESSION_ID
think-thank show SESSION_ID --json
think-thank resume SESSION_ID

# Limits are cumulative session totals, not an additional allowance.
think-thank resume SESSION_ID --max-iterations 24 --max-minutes 30 --max-tokens 200000
```

Press **Ctrl-C** to pause at a checkpoint. Already retrieved sources remain available; pending extraction is retried on resume. An interrupted search/action can be planned again; this is resumable, not exactly-once execution. A hard process kill during a paid request may leave its usage unrecorded. Completed sessions are immutable through `resume`; start a new session to investigate further.

To change storage, put the global option **before** the command:

```bash
think-thank --data-dir /path/to/research demo
```

CLI exit codes: `0` completed or useful limit-stopped result, `1` failure, `130` interrupted. Inspect the reported status to distinguish completion from a limit stop.

## Research loop

1. **Frame:** create scope, assumptions, success criteria, questions, and outline.
2. **Choose:** target the next important knowledge gap with up to three queries.
3. **Retrieve:** read bounded public-web source bodies and retain their provenance.
4. **Extract:** accept only evidence with a valid source ID and an exact, nonempty quotation present in that body.
5. **Review:** update question coverage, hypotheses, opposing/supporting evidence, outline, and remaining gaps.
6. **Countercheck:** request alternative explanations and contrary evidence; force an attempted countercheck if the model keeps skipping it.
7. **Stop or repeat:** completion requires cited answers to all planned questions, a positive review, evidence, and an executed counterchecking search/read. Resource limits and three evidence-free iterations also stop research, but with a `limited` status.
8. **Report:** synthesize findings, confidence labels, competing explanations, limitations, open questions, hypothesis status, and an auditable evidence/source ledger.

A `completed` status means these operational gates passed, **not** that a hypothesis has been scientifically proven. Exact-quote checks establish provenance, not claim entailment, source independence, or truth. Source-quality judgments, hypothesis tests, confidence labels, and assessment of success criteria are model-assisted. Countercheck counts show that a counter-oriented search retrieved a new source, not that an independent refutation was found.

The agent stores brief decision justifications and research artifacts, not private chain-of-thought. It can assess published support for a hypothesis; it does not run external experiments.

## Output and persistence

```text
.think-thank/
  research.sqlite3                 # canonical state, transactional saves
  sessions/<session-id>/
    state.json                     # inspectable state snapshot
    plan.json
    evidence.json
    hypotheses.json
    decisions.json
    outline.json
    sources/S1.txt                 # extracted source text, not original HTML
    checkpoints/0001/…             # latest artifacts for each iteration
    report.md                      # human-readable report with provenance ledger
    report.json                    # structured synthesis (IDs join to evidence.json)
```

SQLite is the source of truth. Exports are individually atomic; if an export fails after a database commit, a subsequent save rebuilds them. Checkpoint folders are updated within an iteration, not an immutable event log. Do not run two processes on the **same session** concurrently. Storage is intended for a trusted local filesystem, not a shared multi-tenant service.

## Architecture

```text
cli.py        commands, environment configuration, progress
server.py     local FastAPI server, background workers, session API
static/       responsive web UI, safe DOM rendering, progress polling
engine.py     research state machine, validation, budgets, checkpoints
models.py     Pydantic artifact schemas
llm.py        OpenAI-compatible API, prompts, JSON validation, bounded repair/retry
web.py        search, read-only retrieval, extraction, URL restrictions
store.py      SQLite state + filesystem artifacts
reporting.py  escaped Markdown report and validated citation ledger
demo.py       deterministic fictional adapters for offline testing
```

One orchestrator owns all state. Planner, extractor, critic/reviewer, and synthesizer are focused model tasks rather than competing agents writing shared files. Model and web adapters are replaceable, making local testing straightforward.

## Safety and privacy

- Retrieved pages and all research context are untrusted data, not instructions. The model cannot execute code, purchase access, submit forms, or invoke arbitrary tools. Search API POSTs only perform discovery.
- Public-source retrieval allows HTTP(S) only, blocks private/local/nonpublic IP targets and credential-bearing URLs, validates redirects, ignores proxy environment variables, caps response sizes, and never executes JavaScript.
- **Not a hardened security sandbox:** DNS validation and connection are separate, so DNS rebinding remains possible. For hostile sources, enforce public-only egress at the network layer and run in an isolated process/container. Prompt-injection instructions mitigate attacks; they cannot guarantee model resistance.
- Topics, source excerpts, and research artifacts are sent to your configured model provider. Search queries go to the selected search provider. Do not submit confidential material without appropriate provider policies.
- API credentials are read from environment variables and not written to session config. Do not paste secrets into research topics or free-form content. Local source/evidence files may themselves be sensitive.
- Source text is capped at 30,000 characters and downloads at 5 MiB. Truncation is recorded. HTML and plain text work by default; optional PDF extraction requires `python3 -m pip install -e '.[pdf]'` and has no OCR. Compressed responses are rejected; requests ask for identity encoding.
- JavaScript-only pages, login walls, paywalls, and anti-bot challenges are not bypassed. Failed sources are recorded. Evidence deduplication detects identical extracted text, not every syndicated or derivative source.
- The final executive summary is model synthesis; traceable claim citations live in findings and the evidence ledger. Review important conclusions before acting on them.

## Development

```bash
python3 -m pip install -e '.[dev]'
python3 -m pytest -q
```

Tests are offline and include mocked HTTP failures, SSRF/redirect validation, model output repair, exact-quote checks, citation integrity, budgets, interruption/resume, storage, report escaping, and the end-to-end demo. Live provider compatibility, search availability, and real-world research quality require separate evaluation using your credentials; the offline demo is not a research-quality benchmark.
