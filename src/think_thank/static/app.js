'use strict';

(() => {
  const $ = id => document.getElementById(id);
  const app = { config: null, sessions: [], selected: null, state: null, tab: 'overview', timer: null, polling: false, refreshPending: false, busy: false, signatures: {}, listSignature: '', failures: 0, lastAnnouncement: '' };
  const limitFields = [
    ['max_iterations', 'Research iterations', 1, 200, 1],
    ['max_minutes', 'Active time · minutes', 0.01, null, 'any'],
    ['max_tokens', 'Model tokens', 1000, null, 1],
    ['max_cost_usd', 'Estimated cost · USD', 0.001, null, 'any'],
    ['max_sources', 'Stored sources', 1, 500, 1],
    ['max_reads_per_iteration', 'Reads per iteration', 1, 10, 1],
    ['input_usd_per_million', 'Input · USD / 1M tokens', 0, null, 'any'],
    ['output_usd_per_million', 'Output · USD / 1M tokens', 0, null, 'any']
  ];
  const number = value => Number(value || 0).toLocaleString();
  const money = value => '$' + Number(value || 0).toFixed(4);
  const human = value => String(value || '').replaceAll('_', ' ');
  const date = value => { const d = new Date(value); return Number.isNaN(d.getTime()) ? 'Date unavailable' : d.toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }); };
  function node(tag, className, text) {
    const el = document.createElement(tag);
    if (className) el.className = className;
    if (text !== undefined && text !== null) el.textContent = String(text);
    return el;
  }
  function clear(el) { el.replaceChildren(); return el; }
  function announce(text) {
    if (app.lastAnnouncement !== text) { $('announcement').textContent = text; app.lastAnnouncement = text; }
  }
  function actionError(error) { $('action-error').textContent = error ? error.message || String(error) : ''; $('action-error').hidden = !error; }
  async function api(path, options = {}) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 20000);
    try {
      const response = await fetch('/api' + path, { ...options, signal: controller.signal, cache: 'no-store', headers: { 'X-Think-Thank': '1', ...(options.body ? { 'Content-Type': 'application/json' } : {}) } });
      if (!response.ok) {
        let detail;
        try { detail = (await response.json()).detail; } catch { /* Use a safe status message. */ }
        let message = typeof detail === 'string' ? detail : Array.isArray(detail) ? detail.map(e => `${(e.loc || []).filter(x => x !== 'body').join(' → ')}: ${e.msg || 'Invalid value'}`).join('; ') : `The server returned an error (${response.status}).`;
        if (response.status === 422) message += ' Check the topic, model, provider configuration, and numeric budget ranges. Live research needs server-side credentials.';
        const error = new Error(message); error.status = response.status; throw error;
      }
      return options.download ? await response.blob() : await response.json();
    } catch (error) {
      if (error.name === 'AbortError') throw new Error('The local server took too long to respond. Retrying automatically.');
      if (error instanceof TypeError) throw new Error('Cannot reach the local server. Check that it is running; your saved research is not cleared.');
      throw error;
    } finally { clearTimeout(timeout); }
  }
  function buildLimits(container, prefix, values) {
    clear(container);
    for (const [key, label, min, max, step] of limitFields) {
      const wrapper = node('div');
      const input = node('input'); input.type = 'number'; input.id = prefix + key; input.name = key; input.min = String(min); input.step = String(step); input.required = true;
      if (max !== null) input.max = String(max);
      input.value = values[key];
      const labelEl = node('label', '', label); labelEl.htmlFor = input.id;
      wrapper.append(labelEl, input); container.append(wrapper);
    }
  }
  function readLimits(prefix) {
    return Object.fromEntries(limitFields.map(([key]) => {
      const value = Number($(prefix + key).value);
      if (!Number.isFinite(value)) throw new Error('All research budgets must be finite numbers.');
      return [key, value];
    }));
  }
  function needsModelKey(config) {
    if (config?.has_api_key) return false;
    try {
      const url = new URL(config?.base_url);
      if (['localhost', '127.0.0.1', '[::1]'].includes(url.hostname)) return false;
    } catch { /* Invalid configuration requires server setup. */ }
    return true;
  }
  function readiness() {
    const demo = document.querySelector('input[name="mode"]:checked').value === 'demo';
    $('new-demo-warning').hidden = !demo;
    $('model').disabled = demo; $('search').disabled = demo;
    const c = app.config;
    if (!c) return;
    const missing = [];
    if (needsModelKey(c)) missing.push('OPENAI_API_KEY');
    if ($('search').value === 'tavily' && !c.has_tavily_key) missing.push('TAVILY_API_KEY');
    const box = clear($('readiness'));
    if (missing.length) {
      box.className = 'notice ' + (demo ? 'neutral' : 'warning');
      box.append(node('strong', '', demo ? 'Demo is ready. Live research needs server setup.' : 'Live research is not ready.'));
      box.append(node('p', '', `Set ${missing.join(' and ')} in the environment of the terminal or service running Think Thank, then restart the server. Do not paste keys into this page. Use demo mode without credentials.`));
    } else {
      box.className = 'notice neutral';
      box.append(node('strong', '', 'Server credentials detected.'));
      box.append(node('p', '', 'Live provider access is checked when a run starts. Credentials and provider URLs stay on the server.'));
    }
    box.append(node('small', '', `Server model: ${c.model} · Search: ${c.search} · Endpoint: ${c.base_url}`));
    $('start').disabled = app.busy || (!demo && missing.length > 0);
  }
  function requestRefresh() {
    clearTimeout(app.timer);
    if (app.polling) app.refreshPending = true;
    else void poll();
  }
  async function poll() {
    if (app.polling) { app.refreshPending = true; return; }
    app.polling = true;
    const selected = app.selected;
    try {
      const [config, sessions, state] = await Promise.all([
        api('/config'), api('/sessions'), selected ? api('/sessions/' + encodeURIComponent(selected)) : Promise.resolve(null)
      ]);
      const firstConfig = !app.config;
      const configChanged = JSON.stringify(config) !== JSON.stringify(app.config);
      app.config = config; app.sessions = sessions;
      if (firstConfig) {
        $('model').value = config.model; $('search').value = config.search;
        buildLimits($('new-limits'), 'new-', config.defaults);
      }
      if (firstConfig || configChanged) readiness();
      renderSessionList();
      if (selected === app.selected && state) { app.state = state; renderSession(state); }
      app.failures = 0; $('connection-error').hidden = true;
      $('connection').textContent = 'Local server connected'; $('connection').className = 'connection connected';
    } catch (error) {
      app.failures += 1;
      $('connection').textContent = 'Connection needs attention'; $('connection').className = 'connection disconnected';
      $('connection-error-text').textContent = error.message + ' Retrying shortly.'; $('connection-error').hidden = false;
      if (error.status === 404 && selected === app.selected && selected) {
        $('session-loading').textContent = 'This session was not found. Select another session or start a new research question.';
        $('session-loading').hidden = false; $('session-content').hidden = true;
        app.selected = null; app.state = null;
      }
    } finally {
      app.polling = false;
      const active = app.state?.active || app.sessions.some(s => s.active);
      const delay = app.refreshPending ? 0 : app.failures ? Math.min(15000, 1000 * 2 ** Math.min(app.failures, 4)) : active ? 1000 : 4000;
      app.refreshPending = false;
      app.timer = setTimeout(poll, delay);
    }
  }
  function renderSessionList() {
    const signature = JSON.stringify([app.sessions, app.selected]);
    if (signature === app.listSignature) return;
    app.listSignature = signature;
    const focusedSession = $('session-list').contains(document.activeElement) ? document.activeElement.dataset.sessionId : null;
    const list = clear($('session-list'));
    $('session-count').textContent = app.sessions.length;
    if (!app.sessions.length) { list.append(node('p', 'sidebar-empty', 'Your questions will live here. Start a research session to build your workspace.')); return; }
    app.sessions.forEach(s => {
      const button = node('button', 'session-item' + (s.id === app.selected ? ' selected' : '')); button.type = 'button'; button.dataset.sessionId = s.id;
      if (s.id === app.selected) button.setAttribute('aria-current', 'page');
      button.append(node('span', 'session-item-title', s.topic));
      const meta = node('span', 'session-item-meta');
      meta.append(node('span', s.active ? 'history-active' : '', s.active ? '● Active' : human(s.status)), node('span', '', date(s.updated_at || s.created_at)));
      button.append(meta); button.addEventListener('click', () => selectSession(s.id)); list.append(button);
      if (s.id === focusedSession) button.focus({ preventScroll: true });
    });
  }
  function selectSession(id) {
    app.selected = id; app.state = null; app.signatures = {}; app.tab = 'overview';
    actionError(null); $('new-workspace').hidden = true; $('session-workspace').hidden = false;
    $('session-content').hidden = true; $('session-loading').hidden = false; $('session-loading').textContent = 'Loading research session…';
    $('resume-panel').hidden = true; $('show-resume').setAttribute('aria-expanded', 'false');
    switchTab('overview'); renderSessionList(); requestRefresh(); $('main').focus();
  }
  function switchTab(name, focus = false) {
    app.tab = name;
    document.querySelectorAll('[data-tab]').forEach(button => {
      const selected = button.dataset.tab === name;
      button.setAttribute('aria-selected', String(selected)); button.tabIndex = selected ? 0 : -1;
      $('panel-' + button.dataset.tab).hidden = !selected;
      if (selected && focus) button.focus();
    });
  }
  function renderSession(s) {
    $('session-loading').hidden = true; $('session-content').hidden = false;
    const demo = s.config?.mode === 'demo';
    const status = s.active ? (s.pause_requested ? 'Pausing' : 'Researching') : s.status === 'running' ? 'Inactive checkpoint' : human(s.status);
    $('session-title').textContent = s.topic; $('session-mode').textContent = demo ? 'DEMO RESEARCH' : 'LIVE RESEARCH';
    $('session-status').textContent = status; $('session-status').className = 'badge ' + (s.active ? 'running' : s.status === 'completed' ? 'completed' : '');
    $('session-meta').textContent = `Created ${date(s.created_at)} · ${demo ? 'Scripted demo providers' : (s.config?.model || 'Server model') + ' / ' + (s.config?.search || 'Server search')} · ${s.id}`;
    $('session-demo-warning').hidden = !demo;
    const missingCredentials = [];
    if (needsModelKey(app.config)) missingCredentials.push('OPENAI_API_KEY');
    if (s.config?.search === 'tavily' && !app.config?.has_tavily_key) missingCredentials.push('TAVILY_API_KEY');
    $('session-readiness').hidden = demo;
    $('session-readiness').className = 'notice ' + (missingCredentials.length ? 'warning' : 'neutral');
    $('session-readiness').textContent = missingCredentials.length
      ? `Live server setup needed: set ${missingCredentials.join(' and ')} in the server process environment and restart the server before resuming. Never enter keys in the browser.`
      : 'Live server credentials detected. Provider access is validated when research starts or resumes; keys remain on the server.';
    const stop = s.stop_reason || (!s.active && s.status === 'running' ? 'No worker is active for this saved checkpoint. Resume to continue with current server configuration.' : '');
    $('stop-reason').textContent = stop; $('stop-reason').hidden = !stop;
    $('pause').hidden = !s.active; $('pause').disabled = app.busy || s.pause_requested;
    $('pause').textContent = s.pause_requested ? 'Pausing at checkpoint…' : 'Pause research';
    $('show-resume').hidden = s.active || s.status === 'completed'; $('show-resume').disabled = app.busy;
    if (s.active || s.status === 'completed') { $('resume-panel').hidden = true; $('show-resume').setAttribute('aria-expanded', 'false'); }
    const limits = s.limits;
    const stats = clear($('stats'));
    for (const [label, value, sub] of [
      ['Iterations', number(s.iteration), `of ${number(limits.max_iterations)} allowed`],
      ['Sources', number(s.sources.length), `${s.evidence.length} evidence items`],
      ['Tokens', number(s.tokens_used), `of ${number(limits.max_tokens)} allowed`],
      ['Est. model cost', money(s.estimated_cost_usd), `of ${money(limits.max_cost_usd)} budget`],
      ['Active time', (s.elapsed_seconds / 60).toFixed(1) + ' min', `of ${limits.max_minutes} min allowed`]
    ]) { const stat = node('div', 'stat'); stat.append(node('span', 'stat-label', label), node('strong', '', value), node('small', '', sub)); stats.append(stat); }
    $('activity-state').textContent = s.active ? s.pause_requested ? 'Pause requested' : 'Worker active' : 'Worker inactive';
    $('activity-dot').className = 'activity-dot' + (s.active ? ' active' : '');
    const events = s.events || []; const latest = events[events.length - 1];
    const phases = { plan: 'Planning', action: 'Choosing next research action', extract: 'Extracting evidence', review: 'Reviewing evidence', report: 'Writing synthesis' };
    let phase = 'Worker active';
    if (latest?.message.startsWith('Model: ')) phase = phases[latest.message.slice(7)] || 'Model request';
    else if (latest?.message.startsWith('Search: ')) phase = 'Searching the web';
    else if (latest?.message.startsWith('Read: ')) phase = 'Reading a source';
    if (s.active && !s.pause_requested) $('activity-state').textContent = phase;
    $('latest-event').textContent = latest?.message || (s.active ? 'Starting research. Waiting for the first saved activity…' : 'No recent events on this server. Explore the saved research below.');
    $('activity-time').textContent = latest ? date(latest.at) : `Updated ${date(s.updated_at)}`;
    $('event-count').textContent = `(${events.length})`;
    updateSection('timeline', events, () => {
      const list = clear($('timeline'));
      if (!events.length) list.append(node('li', 'muted', 'No activity events available.'));
      [...events].reverse().forEach(event => { const li = node('li'); const time = node('time', '', date(event.at)); time.dateTime = event.at; li.append(time, node('span', 'preserve', event.message)); list.append(li); });
    });
    const context = evidenceContext(s);
    updateSection('overview', [s.plan, s.outline, s.gaps, s.decisions, s.searched_queries, s.errors, s.counterchecks], () => renderOverview(s, context));
    updateSection('evidence', [s.evidence, s.sources], () => renderEvidence(s, context));
    updateSection('hypotheses', [s.hypotheses, s.evidence, s.sources], () => renderHypotheses(s, context));
    updateSection('report', [s.report, s.evidence, s.sources, s.gaps], () => renderReport(s, context));
    announce(`${status}. ${s.iteration} iterations, ${s.sources.length} sources, ${s.evidence.length} evidence items.`);
  }
  function updateSection(key, value, render) {
    const signature = JSON.stringify(value);
    if (app.signatures[key] !== signature) {
      const root = $(key === 'timeline' ? key : 'panel-' + key);
      const open = new Set([...root.querySelectorAll('details[open]')].map(el => el.dataset.key));
      const focused = root.contains(document.activeElement) ? document.activeElement.dataset.focusKey : null;
      render();
      root.querySelectorAll('details').forEach(el => { if (open.has(el.dataset.key)) el.open = true; });
      if (focused) [...root.querySelectorAll('[data-focus-key]')].find(el => el.dataset.focusKey === focused)?.focus({ preventScroll: true });
      app.signatures[key] = signature;
    }
  }
  function empty(parent, title, text) { const box = node('div', 'empty-state'); box.append(node('h3', '', title), node('p', '', text)); parent.append(box); }
  function section(parent, title, className = 'card') { const box = node('section', className); box.append(node('h2', '', title)); parent.append(box); return box; }
  function listSection(parent, title, items, fallback) {
    const box = section(parent, title);
    if (!items?.length) { box.append(node('p', 'muted', fallback || 'None recorded. This does not establish that none exist.')); return box; }
    const list = node('ul', 'text-list'); items.forEach(item => list.append(node('li', 'preserve', item))); box.append(list); return box;
  }
  function evidenceContext(s) {
    const sourceCounts = new Map(); const evidenceCounts = new Map();
    s.sources.forEach(x => sourceCounts.set(x.id, (sourceCounts.get(x.id) || 0) + 1));
    s.evidence.forEach(x => evidenceCounts.set(x.id, (evidenceCounts.get(x.id) || 0) + 1));
    const sources = new Map(s.sources.filter(x => sourceCounts.get(x.id) === 1).map(x => [x.id, x]));
    const valid = new Map(); const positions = new Map();
    s.evidence.forEach((e, index) => {
      positions.set(e.id, index);
      if (e.id && evidenceCounts.get(e.id) === 1 && sources.has(e.source_id) && e.quote.trim() && sources.get(e.source_id).text.includes(e.quote)) valid.set(e.id, e);
    });
    return { sources, valid, positions };
  }
  function citations(parent, ids, context, label = 'Evidence') {
    const row = node('div', 'citations'); row.append(node('span', 'citation-label', label + ':'));
    const unique = [...new Set(ids || [])];
    if (!unique.length) row.append(node('span', 'muted', 'None recorded'));
    unique.forEach(id => {
      if (!context.valid.has(id)) { row.append(node('span', 'invalid-citation', `${id} · unavailable / unverified`)); return; }
      const a = node('a', 'citation', id + ' ↗');
      const target = 'evidence-' + context.positions.get(id); a.href = '#' + target; a.dataset.focusKey = label + '-' + id;
      a.setAttribute('aria-label', `${label}: view evidence ${id}`);
      a.addEventListener('click', event => { event.preventDefault(); switchTab('evidence'); const card = $(target); if (card) { card.focus({ preventScroll: true }); card.scrollIntoView({ block: 'center', behavior: 'auto' }); } });
      row.append(a);
    });
    parent.append(row);
  }
  function safeLink(url, title) {
    try {
      const parsed = new URL(url);
      if (!['http:', 'https:'].includes(parsed.protocol) || parsed.username || parsed.password || !parsed.hostname) throw new Error('Unsafe URL');
      const a = node('a', 'source-link', title || parsed.hostname); a.href = parsed.href; a.target = '_blank'; a.rel = 'noopener noreferrer'; a.referrerPolicy = 'no-referrer';
      a.append(node('span', 'external-symbol', ' ↗')); return a;
    } catch { return node('span', 'muted', (title || 'Source') + ' · external URL unavailable or unsafe'); }
  }
  function renderOverview(s, context) {
    const panel = clear($('panel-overview'));
    if (!s.plan) { empty(panel, 'A thoughtful answer starts with a plan.', 'The research scope and guiding questions will appear after the planning checkpoint.'); return; }
    const scope = section(panel, 'Research scope'); scope.append(node('p', 'lead preserve', s.plan.scope));
    const grid = node('div', 'content-grid'); panel.append(grid);
    const questions = section(grid, 'Guiding questions');
    s.plan.questions.forEach(q => {
      const item = node('div', 'question'); const heading = node('div', 'question-heading'); heading.append(node('span', 'badge', human(q.status)), node('span', 'micro', q.id));
      item.append(heading, node('p', 'preserve', q.text)); citations(item, q.evidence_ids, context); questions.append(item);
    });
    const right = node('div', 'stack'); grid.append(right);
    listSection(right, 'Working outline', s.outline.length ? s.outline : s.plan.outline, 'The outline has not been recorded yet.');
    listSection(right, 'Evidence gaps', s.gaps, 'No gaps recorded yet; absence of recorded gaps is not proof of completeness.');
    const criteria = node('div', 'content-grid'); panel.append(criteria);
    listSection(criteria, 'Success criteria', s.plan.success_criteria); listSection(criteria, 'Assumptions', s.plan.assumptions);
    const decisions = section(panel, 'Research decisions'); decisions.append(node('p', 'hint', `${s.counterchecks} counterchecks recorded. Decisions are recorded actions and rationales, not hidden model reasoning.`));
    if (!s.decisions.length) decisions.append(node('p', 'muted', 'No decisions recorded yet.'));
    [...s.decisions].reverse().forEach(d => { const item = node('div', 'decision'); item.append(node('span', 'micro', `ITERATION ${d.iteration} · ${date(d.at)}`), node('h3', '', human(d.action)), node('p', 'preserve', d.rationale)); decisions.append(item); });
    listSection(panel, 'Search queries', s.searched_queries, 'No searches recorded yet.');
    if (s.errors.length) listSection(panel, 'Recorded issues', s.errors);
  }
  function renderEvidence(s, context) {
    const panel = clear($('panel-evidence'));
    const heading = node('div', 'panel-heading'); heading.append(node('h2', '', 'The evidence ledger'), node('span', 'muted', `${context.valid.size} traceable / ${s.evidence.length} recorded`)); panel.append(heading);
    panel.append(node('p', 'notice neutral', 'Quotation checks establish exact text presence in a uniquely identified stored source, not source accuracy or proof that a quotation supports a claim. Model interpretations remain unverified.'));
    if (!s.evidence.length) empty(panel, 'No evidence recorded yet.', 'Search snippets are not evidence. Quotations appear here after sources are read and saved.');
    s.evidence.forEach((e, index) => {
      const card = node('article', 'card evidence-card'); card.id = 'evidence-' + index; card.tabIndex = -1;
      const verified = context.valid.has(e.id); const source = context.sources.get(e.source_id);
      const head = node('div', 'card-kicker'); head.append(node('span', 'micro', e.id + ' → ' + e.source_id), node('span', 'badge' + (verified ? ' completed' : ' caution'), verified ? 'Exact quote matched' : 'Unverified quotation'));
      card.append(head, node('h3', 'preserve', e.claim), node('p', 'hint', human(e.kind) + ' · source-reported / model-extracted, not independently verified'));
      card.append(node('blockquote', 'exact-quote', e.quote));
      if (!verified) card.append(node('p', 'warning-text', 'Missing or ambiguous identifiers, empty quotation, or quotation absent from stored source text. Excluded from valid finding citations.'));
      if (source) card.append(safeLink(source.url, source.title));
      else card.append(node('p', 'muted', 'Stored source is missing or ambiguous.'));
      card.append(node('p', 'hint preserve', 'Evidence limitations: ' + (e.limitations || 'None recorded (not proof that none exist).')));
      panel.append(card);
    });
    const sources = section(panel, 'Stored sources');
    if (!s.sources.length) sources.append(node('p', 'muted', 'No sources retained yet.'));
    s.sources.forEach((source, index) => {
      const details = node('details', 'source-details'); details.dataset.key = 'source-' + index;
      const summary = node('summary', '', source.id + ' · ' + source.title); summary.dataset.focusKey = 'source-summary-' + index;
      details.append(summary); const body = node('div', 'source-body');
      body.append(safeLink(source.url, source.url));
      body.append(node('p', 'hint', `Author: ${source.author || 'Unknown'} · Published: ${source.published_at || 'Unknown'} · Retrieved: ${date(source.retrieved_at)}`));
      body.append(node('p', 'hint wrap', `Content hash: ${source.content_hash || 'Not recorded'} · Truncated: ${source.truncated ? 'Yes' : 'No'}`));
      body.append(node('pre', 'source-text', source.text)); details.append(body); sources.append(details);
    });
  }
  function renderHypotheses(s, context) {
    const panel = clear($('panel-hypotheses'));
    const heading = node('div', 'panel-heading'); heading.append(node('h2', '', 'Ideas worth testing')); panel.append(heading);
    panel.append(node('p', 'muted', 'Model-assessed hypotheses, not established conclusions. Examine both supporting and opposing evidence.'));
    if (!s.hypotheses.length) { empty(panel, 'No hypotheses recorded yet.', 'Hypotheses and proposed tests appear as the research is reviewed.'); return; }
    s.hypotheses.forEach(h => {
      const card = node('article', 'card hypothesis-card'); const head = node('div', 'card-kicker'); head.append(node('span', 'micro', h.id), node('span', 'badge', human(h.status)));
      card.append(head, node('h3', 'preserve', h.statement));
      citations(card, h.supporting_evidence, context, 'Supporting evidence'); citations(card, h.opposing_evidence, context, 'Opposing evidence');
      const test = node('div', 'test-box'); test.append(node('strong', '', 'Proposed test'), node('p', 'preserve', h.test)); card.append(test); panel.append(card);
    });
  }
  function renderReport(s, context) {
    const panel = clear($('panel-report'));
    if (!s.report) { empty(panel, 'The report is not ready yet.', 'The saved synthesis will appear here when generated. You can already inspect the plan, evidence, and hypotheses in the other tabs.'); return; }
    const tools = node('div', 'panel-heading'); tools.append(node('span', 'eyebrow teal', 'RESEARCH SYNTHESIS'));
    const downloads = node('div', 'button-row');
    for (const [format, label] of [['md', 'Download Markdown'], ['json', 'Download JSON']]) {
      const button = node('button', 'button secondary compact', label); button.type = 'button'; button.dataset.focusKey = 'download-' + format;
      button.addEventListener('click', () => downloadReport(s.id, format, button)); downloads.append(button);
    }
    tools.append(downloads); panel.append(tools);
    const report = section(panel, s.report.title, 'card report-document');
    report.append(node('p', 'notice ' + (context.valid.size ? 'neutral' : 'warning'), context.valid.size ? 'Model-written synthesis, not independently verified. The executive summary has no claim-level citations. Confidence is model-assessed, not calibrated.' : 'No verified research: no evidence has a nonempty exact quotation in a uniquely identified stored source. Claims below are unverified.'));
    report.append(node('h3', 'report-section-title', 'Executive summary'), node('p', 'lead preserve', s.report.executive_summary));
    report.append(node('h3', 'report-section-title', 'Findings'));
    if (!s.report.findings.length) report.append(node('p', 'muted', 'No findings recorded.'));
    s.report.findings.forEach((finding, index) => {
      const card = node('section', 'finding'); const good = finding.evidence_ids.filter(id => context.valid.has(id));
      card.append(node('span', 'micro', `FINDING ${String(index + 1).padStart(2, '0')} · ${human(finding.confidence).toUpperCase()} MODEL CONFIDENCE`), node('h4', '', finding.heading), node('p', 'preserve', finding.text));
      card.append(node('p', good.length ? 'hint' : 'warning-text', good.length ? 'Source-linked model synthesis; not independently verified.' : 'Unverified / unsupported model claim.'));
      citations(card, finding.evidence_ids, context); report.append(card);
    });
    listSection(report, 'Competing explanations', s.report.competing_explanations, 'No competing explanations recorded.');
    listSection(report, 'Limitations', [...new Set([...s.report.limitations, ...s.gaps])]);
    listSection(report, 'Open questions', s.report.open_questions);
    report.append(node('p', 'hint', 'Method: exact quotation presence is checked against stored source text. This does not establish entailment, independence, lack of bias, or factual accuracy. Cost figures are estimates, not invoices.'));
  }
  async function downloadReport(id, format, button) {
    button.disabled = true; actionError(null);
    try {
      const blob = await api('/sessions/' + encodeURIComponent(id) + '/report?format=' + format, { download: true });
      const url = URL.createObjectURL(blob); const a = node('a'); a.href = url; a.download = id + '-report.' + format;
      document.body.append(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(url), 10000);
    } catch (error) { actionError(error); } finally { button.disabled = false; }
  }
  async function mutate(operation) {
    if (app.busy) return;
    app.busy = true; actionError(null); readiness(); $('pause').disabled = true; $('show-resume').disabled = true; $('resume-submit').disabled = true;
    try { await operation(); } catch (error) { actionError(error); }
    finally { app.busy = false; readiness(); $('resume-submit').disabled = false; if (app.state) renderSession(app.state); requestRefresh(); }
  }
  $('research-form').addEventListener('submit', event => {
    event.preventDefault(); if (!app.config) return;
    const topic = $('topic').value.trim();
    if (!topic) { actionError(new Error('Enter a research question before starting.')); $('topic').focus(); return; }
    void mutate(async () => {
      const mode = document.querySelector('input[name="mode"]:checked').value;
      const body = { topic, mode, limits: readLimits('new-') };
      if (mode === 'live') { if ($('model').value.trim()) body.model = $('model').value.trim(); body.search = $('search').value; }
      const result = await api('/sessions', { method: 'POST', body: JSON.stringify(body) }); selectSession(result.id); announce('Research session started.');
    });
  });
  $('pause').addEventListener('click', () => {
    const id = app.selected;
    void mutate(async () => { await api('/sessions/' + encodeURIComponent(id) + '/pause', { method: 'POST', body: '{}' }); if (app.selected === id && app.state) app.state.pause_requested = true; announce('Pause requested. Waiting for the research worker to stop safely.'); });
  });
  $('show-resume').addEventListener('click', () => {
    if (!app.state) return;
    buildLimits($('resume-limits'), 'resume-', app.state.limits); $('resume-panel').hidden = false; $('show-resume').setAttribute('aria-expanded', 'true'); $('resume-max_iterations').focus();
  });
  $('cancel-resume').addEventListener('click', () => { $('resume-panel').hidden = true; $('show-resume').setAttribute('aria-expanded', 'false'); $('show-resume').focus(); });
  $('resume-panel').addEventListener('submit', event => {
    event.preventDefault(); const id = app.selected;
    void mutate(async () => {
      await api('/sessions/' + encodeURIComponent(id) + '/resume', { method: 'POST', body: JSON.stringify({ limits: readLimits('resume-') }) });
      if (app.selected === id) { $('resume-panel').hidden = true; $('show-resume').setAttribute('aria-expanded', 'false'); }
      announce('Research resumed with updated total limits.');
    });
  });
  $('new-session').addEventListener('click', () => {
    app.selected = null; app.state = null; actionError(null); $('session-workspace').hidden = true; $('new-workspace').hidden = false; renderSessionList(); readiness(); $('topic').focus(); requestRefresh();
  });
  document.querySelectorAll('input[name="mode"]').forEach(input => input.addEventListener('change', readiness));
  $('search').addEventListener('change', readiness); $('retry').addEventListener('click', requestRefresh);
  const tabs = [...document.querySelectorAll('[data-tab]')];
  tabs.forEach((button, index) => {
    button.addEventListener('click', () => switchTab(button.dataset.tab));
    button.addEventListener('keydown', event => {
      let target;
      if (event.key === 'ArrowRight') target = (index + 1) % tabs.length;
      if (event.key === 'ArrowLeft') target = (index + tabs.length - 1) % tabs.length;
      if (event.key === 'Home') target = 0;
      if (event.key === 'End') target = tabs.length - 1;
      if (target !== undefined) { event.preventDefault(); switchTab(tabs[target].dataset.tab, true); }
    });
  });
  window.addEventListener('online', requestRefresh);
  document.addEventListener('visibilitychange', () => { if (!document.hidden) requestRefresh(); });
  void poll();
})();
