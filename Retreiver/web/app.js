const $ = (id) => document.getElementById(id);
const setText = (id, value) => { const el = $(id); if (el) el.textContent = value; };
let state = { operational: {}, files: [], options: {}, sourceCatalog: {datasets: [], groundtruth: []} };
let benchmarkOptions = {};
let globalSourceContextState = {officialGroundtruthId: ''};
let candidateOperationsState = {adapters: null, portfolio: null, details: null, busy: false};

const esc = (v) => String(v ?? '').replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
const num = (v) => Number.parseFloat(v || 0) || 0;
const uniq = (rows, key) => [...new Set(rows.map(r => r[key]).filter(Boolean))].sort();
const fmt = (v, d = 3) => Number.isFinite(num(v)) ? num(v).toFixed(d) : '—';
const fmtInt = (v) => Number.isFinite(Number(v)) ? Number(v).toLocaleString() : '0';
const costLabel = (r) => /openai|amazon/i.test(`${r.embedding || ''} ${r.reranker || ''}`) ? 'commercial key/cost' : 'open-source/VM cost';

function sourceStateModule() {
  return globalThis.DashboardSourceState || null;
}

function canonicalRerankerName(value) {
  if (sourceStateModule()) return sourceStateModule().canonicalRerankerName(value);
  const raw = String(value || '').trim();
  if (!raw) return 'none';
  const normalized = raw.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '');
  if (['none', 'no_reranker', 'baseline'].includes(normalized)) return 'none';
  if (['qwen', 'qwen3', 'qwen3_4b', 'qwen3_4b_rerank', 'qwen3_4b_reranker', 'qwen3_reranker_4b_seq_cls'].includes(normalized)) return 'Qwen3:4B Rerank';
  if (['bge', 'bge_reranker', 'bge_reranker_base'].includes(normalized)) return 'bge-reranker-base';
  if (['amazon', 'amazon_rerank', 'amazon_rerank_v1', 'amazon_rerank_v1_0', 'amazon_bedrock_rerank', 'amazon_reranker'].includes(normalized)) return 'Amazon Rerank v1';
  if (normalized.includes('amazon') && normalized.includes('rerank')) return 'Amazon Rerank v1';
  if (normalized.includes('qwen') && normalized.includes('rerank')) return 'Qwen3:4B Rerank';
  if (normalized.includes('bge') && normalized.includes('rerank')) return 'bge-reranker-base';
  return raw;
}

function canonicalMetricRow(row, extra = {}) {
  if (sourceStateModule()) return sourceStateModule().canonicalMetricRow(row, extra);
  return {...row, ...extra, reranker: canonicalRerankerName(row?.reranker || row?.reranking_model || 'none')};
}

function metricRowKey(row) {
  if (sourceStateModule()) return sourceStateModule().metricRowKey(row);
  return `${row.sheet || ''}|${row.embedding || ''}|${row.store || ''}|${canonicalRerankerName(row.reranker || 'none')}`;
}

function dedupeMetricRows(rows) {
  const map = new Map();
  rows.map(r => canonicalMetricRow(r)).forEach(r => {
    const key = metricRowKey(r);
    if (!map.has(key)) map.set(key, r);
  });
  return [...map.values()];
}

function amazonStatusBlocksRun() {
  const status = String(state.operational?.amazon_status || '').toLowerCase();
  if (!status) return false;
  if (/ok|ready|complete|available|granted/.test(status)) return false;
  return /pending|block|denied|credential|permission|expired|not configured/.test(status);
}

function missingCoverageBlocked(reranker, store) {
  return canonicalRerankerName(reranker) === 'Amazon Rerank v1' && String(store || '').toLowerCase() === 'faiss' && amazonStatusBlocksRun();
}

function positivePageNumber(value) {
  const raw = String(value ?? '').trim();
  if (!raw || raw === '—') return '';
  const match = raw.match(/\d+/);
  if (!match) return '';
  const page = Number.parseInt(match[0], 10);
  return Number.isFinite(page) && page > 0 ? page : '';
}

function pageNumberFromHit(hit = {}) {
  const metadata = hit.metadata || hit.meta || {};
  const candidates = [
    hit.page_number,
    hit.page_num,
    hit.page,
    hit.pdf_page,
    hit.page_start,
    hit.start_page,
    metadata.page_number,
    metadata.page_num,
    metadata.page,
    metadata.pdf_page,
  ];
  for (const candidate of candidates) {
    const page = positivePageNumber(candidate);
    if (page) return page;
  }
  return '';
}

function pdfOpenUrl(name, page = '') {
  const raw = String(name || '').trim();
  if (!raw || raw === '—' || !raw.toLowerCase().endsWith('.pdf')) return '';
  const clean = raw.split(/[\\/]/).pop();
  const pageNumber = positivePageNumber(page);
  return `/api/pdf?name=${encodeURIComponent(clean)}${pageNumber ? `#page=${pageNumber}` : ''}`;
}

function pdfLink(name, label = '', page = '') {
  const pageNumber = positivePageNumber(page);
  const url = pdfOpenUrl(name, pageNumber);
  const text = label || name || '—';
  const display = pageNumber ? `${text} · page ${pageNumber}` : text;
  if (!url) return esc(display);
  return `<a class="pdf-link" href="${url}" target="_blank" rel="noopener">${esc(display)}</a>`;
}

async function api(path, options = {}) {
  const res = await fetch(path, {cache: 'no-store', ...options});
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

function table(el, rows, cols) {
  if (!el) return;
  if (!rows || rows.length === 0) {
    el.innerHTML = '<tr><td class="empty-cell">No live artifact data yet</td></tr>';
    return;
  }
  el.innerHTML = `<thead><tr>${cols.map(c => `<th>${esc(c.label)}</th>`).join('')}</tr></thead><tbody>` +
    rows.map((r, i) => `<tr>${cols.map(c => `<td data-label="${esc(c.label)}">${c.render ? c.render(r, i) : esc(r[c.key] ?? '')}</td>`).join('')}</tr>`).join('') + '</tbody>';
}

function fillSelect(id, values, label) {
  const el = $(id); if (!el) return;
  const cur = el.value || 'all';
  el.innerHTML = `<option value="all">${esc(label)}</option>` + values.map(v => `<option value="${esc(v)}">${esc(v)}</option>`).join('');
  el.value = values.includes(cur) ? cur : 'all';
}

function fillRunSelect(id, values, allLabel = 'All') {
  const el = $(id); if (!el) return;
  const cur = el.value || 'all';
  el.innerHTML = `<option value="all">${esc(allLabel)}</option>` + values.map(v => `<option value="${esc(v)}">${esc(v)}</option>`).join('');
  el.value = cur === 'all' || values.includes(cur) ? cur : 'all';
}

function fillRunMultiSelect(id, values, allLabel = 'All') {
  const el = $(id); if (!el) return;
  const previous = selectedValues(id);
  const options = ['all', ...values];
  el.innerHTML = `<option value="all">${esc(allLabel)}</option>` + values.map(v => `<option value="${esc(v)}">${esc(v)}</option>`).join('');
  const keep = previous.filter(v => options.includes(v));
  const active = keep.length ? keep : ['all'];
  [...el.options].forEach(opt => { opt.selected = active.includes(opt.value); });
}

function selectedValues(id) {
  const el = $(id); if (!el) return ['all'];
  const values = [...(el.selectedOptions || [])].map(o => o.value).filter(Boolean);
  return values.length ? values : ['all'];
}

function selectOnly(id, value) {
  const el = $(id); if (!el) return;
  [...el.options].forEach(opt => { opt.selected = opt.value === value; });
}

function expandedSelection(id, values) {
  const picked = selectedValues(id);
  if (picked.includes('all')) return values || [];
  return picked;
}

function activeDatasetSheets() {
  return benchmarkOptions.chunkers || [];
}

function candidateLane() {
  return benchmarkOptions?.candidate_lane || {};
}

function selectedRetrievalMethod() {
  return $('runRetrievalMethod')?.value || 'Cosine Similarity';
}

function isCandidateRetrievalMethod() {
  return (candidateLane().retrieval_methods || []).includes(selectedRetrievalMethod());
}

function selectedMatrixComboCount() {
  const opts = benchmarkOptions || {};
  const chunkers = expandedSelection('runSheet', activeDatasetSheets());
  if (isCandidateRetrievalMethod()) {
    const candidateEmbeddings = candidateLane().embeddings || [];
    const embeddings = expandedSelection('runEmbedding', candidateEmbeddings);
    const candidateRerankers = candidateLane().rerankers || [];
    const picked = selectedValues('runRerankerMain');
    const rerankers = picked.includes('all') ? candidateRerankers : picked;
    return Math.max(chunkers.length, 0) * Math.max(embeddings.length, 0) * Math.max(rerankers.length, 0);
  }
  const embeddings = expandedSelection('runEmbedding', opts.embeddings || []);
  const stores = expandedSelection('runStore', opts.vector_stores || []);
  const officialRerankers = [...new Set((opts.rerankers || []).map(canonicalRerankerName))];
  const pickedRerankers = selectedValues('runRerankerMain');
  const rerankers = pickedRerankers.includes('all') ? officialRerankers : pickedRerankers;
  return Math.max(chunkers.length, 0) * Math.max(embeddings.length, 0) * Math.max(stores.length, 0) * Math.max(rerankers.length, 0);
}

function updateSelectedMatrixCount() {
  const count = selectedMatrixComboCount();
  const el = $('selectedMatrixCount');
  if (el) el.textContent = `${fmtInt(count)} selected combinations`;
  return count;
}

function latestRows(rows) {
  const ok = rows.filter(r => r.status === 'ok' && r.store && r.embedding && r.sheet);
  const map = new Map();
  ok.forEach(r => map.set(`${r.sheet}|${r.embedding}|${r.store}`, r));
  return [...map.values()].sort((a,b) => String(b.created_at || b.run_id || '').localeCompare(String(a.created_at || a.run_id || '')));
}

function renderServices(health) {
  const el = $('serviceCards');
  if (!health.length) {
    el.innerHTML = '<div class="empty-state">No service snapshot yet. Run collect_vm_dashboard_artifacts.py on the VM.</div>';
    return;
  }
  el.innerHTML = health.map(h => {
    const ok = !!h.ok;
    const error = h.error ? `<small>${esc(h.error).slice(0, 90)}</small>` : `<small>${esc(h.url || '')}</small>`;
    return `<div class="service-card ${ok ? 'ok' : 'warn'}">
      <div><strong>${esc(h.name || 'service')}</strong>${error}</div>
      <span>${ok ? 'Healthy' : 'Check'}</span>
      <code>${fmt(h.latency_ms, 1)} ms</code>
    </div>`;
  }).join('');
}

function matrixOptions() {
  const opt = benchmarkOptions || state.options || {};
  return {
    chunkers: opt.chunkers || [],
    embeddings: opt.embeddings || [],
    stores: opt.vector_stores || [],
    rerankers: opt.rerankers || [],
  };
}

function setRunSelection(sheet, embedding, store, reranker = 'all') {
  showPage('run');
  selectOnly('runSheet', sheet || 'all');
  selectOnly('runEmbedding', embedding || 'all');
  selectOnly('runStore', store || 'all');
  selectOnly('runRerankerMain', reranker || 'all');
  updateSelectedMatrixCount();
  $('runStatus').textContent = 'Selected missing benchmark option';
  $('runOutput').textContent = `Selected: chunker=${sheet || 'all'}, embedding=${embedding || 'all'}, vector DB=${store || 'all'}, reranker=${reranker || 'all'}\nClick Preflight first, then Run selected pipeline.`;
}

function metricCoverageRows() {
  const evaluation = state.operational?.evaluation || {};
  const sourceState = sourceStateModule();
  const expectedKeys = evaluation.benchmark_reference?.report?.expected_keys;
  const byKey = new Map();
  [
    ...(evaluation.summary || []),
    ...(evaluation.reranked?.summary || []),
    ...(evaluation.benchmark_reference?.summary || []),
  ].filter(r => r.sheet && r.embedding && r.store).forEach(raw => {
    const normalized = canonicalMetricRow(raw);
    const admission = sourceState?.officialRowAdmission(normalized, {expectedKeys, reranked: true})
      || {eligible: false, missing_metrics: []};
    const eligible = admission.eligible;
    const next = canonicalMetricRow(normalized, {
      status: eligible ? 'metrics' : (normalized.status === 'completed' ? 'metrics_incomplete' : (normalized.status || 'unknown')),
      missing_metrics: admission.missing_metrics,
      admission_reasons: admission.reasons || [],
    });
    const key = metricRowKey(next);
    const current = byKey.get(key);
    const priority = row => row?.status === 'metrics' ? 3 : row?.status === 'metrics_incomplete' ? 2 : 1;
    if (!current || priority(next) > priority(current)) byKey.set(key, next);
  });
  return [...byKey.values()];
}

function coverageEvidenceRows(rows) {
  const map = new Map();
  latestRows(rows).map(r => canonicalMetricRow(r, {
    reranker: r.reranker || 'none',
    operational_seconds: r.total_store_seconds || '',
  })).forEach(r => map.set(metricRowKey(r), r));
  metricCoverageRows().forEach(r => {
    const key = metricRowKey(r);
    const operational = map.get(key) || {};
    map.set(key, {...operational, ...r, operational_seconds: operational.operational_seconds || ''});
  });
  return [...map.values()];
}

function renderCoverage(rows) {
  const latest = coverageEvidenceRows(rows);
  const opts = matrixOptions();
  const stores = opts.stores.length ? opts.stores : uniq(latest, 'store');
  const chunkers = opts.chunkers.length ? opts.chunkers : uniq(latest, 'sheet');
  const embeddings = opts.embeddings.length ? opts.embeddings : uniq(latest, 'embedding');
  const rerankers = opts.rerankers.length ? [...new Set(opts.rerankers.map(canonicalRerankerName))] : uniq(latest, 'reranker');
  const expected = chunkers.length * embeddings.length * stores.length * Math.max(rerankers.length, 1);
  const latestMap = new Map(latest.map(r => [metricRowKey(r), r]));
  let covered = 0;
  const coverageRows = [];
  chunkers.forEach(sheet => embeddings.forEach(embedding => rerankers.forEach(reranker => {
    const storeMap = {};
    stores.forEach(store => {
      const value = latestMap.get(`${sheet}|${embedding}|${store}|${reranker}`);
      if (value) covered += 1;
      storeMap[store] = value;
    });
    coverageRows.push({ sheet, embedding, reranker, stores: storeMap });
  })));
  $('coverageHint').textContent = `${covered}/${expected || covered} chunker × embedding × vector DB × reranker combos covered`;
  const cols = [
    {key:'sheet', label:'Chunker'},
    {key:'embedding', label:'Embedding'},
    {key:'reranker', label:'Reranker'},
    ...stores.map(st => ({key:st, label:st, render:r => {
      const v = r.stores[st];
      if (v?.status === 'metrics') {
        const combo = pipelineLabel(v);
        return `<button class="coverage-metric-btn mini-run-btn" type="button" data-combo="${esc(combo)}" title="Open this completed combination in Metrics">Done · open Metrics</button>`;
      }
      if (v?.status === 'metrics_incomplete') {
        const missing = (v.missing_metrics || []).join(', ') || 'required metrics';
        return `<span class="badge warn" title="Missing: ${esc(missing)}">metrics incomplete</span>`;
      }
      if (v) {
        const timing = v.operational_seconds ? ` <code title="Vector-store ingestion time">ingest ${fmt(v.operational_seconds, 2)}s</code>` : '';
        return `<span class="coverage-ok">ingested</span>${timing}`;
      }
      if (missingCoverageBlocked(r.reranker, st)) {
        return `<span class="badge warn" title="${esc(state.operational?.amazon_status || 'AWS Bedrock rerank blocked')}">blocked</span>`;
      }
      return `<button class="mini-run-btn" data-sheet="${esc(r.sheet)}" data-embedding="${esc(r.embedding)}" data-store="${esc(st)}" data-reranker="${esc(r.reranker)}">Run</button>`;
    }})),
    {key:'total', label:'DBs ready', render:r => `${Object.values(r.stores).filter(Boolean).length}/${stores.length}`}
  ];
  table($('coverageTable'), coverageRows, cols);
}

function buildEvidenceRows(retrievalSmokes, rerankerSmokes) {
  const base = (retrievalSmokes || []).map(r => ({...r, reranker: 'none', evidence_type: r.evidence_type || 'initial topK retrieval'}));
  const reranked = (rerankerSmokes || []).map(r => ({...r, evidence_type: r.evidence_type || 'reranked final top3'}));
  return [...reranked, ...base].map(r => {
    const rawHits = r.hits || [];
    const hits = rawHits.map((hit, i) => ({
      rank: hit.rank || i + 1,
      pdf_name: hit.pdf_name || hit.pdf || hit.source || '—',
      page_number: pageNumberFromHit(hit),
      source_type: hit.source_type || hit.type || hit.metadata?.source_type || '',
      score: hit.score ?? hit.relevance_score ?? hit.relevanceScore ?? hit.similarity ?? '',
      text: evidenceTextFromHit(hit),
    }));
    return {
      ...r,
      reranker: canonicalRerankerName(r.reranker || 'none'),
      top_pdf: (hits[0] || {}).pdf_name || '—',
      top_page_number: (hits[0] || {}).page_number || '',
      evidence_snippet: hits[0]?.text || '',
      evidence_hits: hits,
      initial_topk_count: r.retrieved_count || hits.length || rawHits.length || '—',
      final_top3: hits.slice(0, 3),
    };
  });
}

function evidenceTextFromHit(hit) {
  return String(hit?.paragraph || hit?.text || hit?.chunk || hit?.content || hit?.page_content || '').replace(/\s+/g, ' ').trim();
}

function shortEvidenceLabel(row) {
  const hit = (row.evidence_hits || [])[0] || {};
  const text = hit.text || 'Open retrieved evidence';
  const pdf = hit.pdf_name && hit.pdf_name !== '—' ? `${hit.pdf_name}: ` : '';
  return `${pdf}${text}`.slice(0, 110);
}

function evidenceStageLabel(type) {
  const normalized = String(type || '').toLowerCase();
  if (normalized.includes('benchmark final')) return 'Benchmark final top 5 evidence';
  if (normalized.includes('reranked')) return 'Smoke reranked final top 3';
  if (normalized.includes('initial')) return 'Initial vector top K retrieval';
  return type || 'Evidence row';
}

function renderEvidenceDetails(row) {
  const limit = row.evidence_type === 'reranked final top3' ? 3 : 5;
  const hits = (row.evidence_hits || []).filter(h => h.text).slice(0, limit);
  if (!hits.length) return '<span class="muted-text">No paragraph returned in artifact</span>';
  const stage = evidenceStageLabel(row.evidence_type);
  const body = hits.map((h, i) => `<article class="evidence-hit ${i < 3 ? 'final-hit' : ''}">
    <div><strong>${esc(stage)} · Rank ${esc(h.rank)}</strong><span>${pdfLink(h.pdf_name || '—', '', h.page_number)}${h.score !== '' ? ` · score ${esc(fmt(h.score, 4))}` : ''}</span></div>
    <p>${esc(h.text)}</p>
  </article>`).join('');
  return `<div class="evidence-stage-note">${esc(stage)}. Open the PDF page link to inspect the source page behind each retrieved chunk.</div><div class="evidence-hit-list">${body}</div>`;
}

function filteredRetrieval(rows) {
  return filterBySelect(rows, {
    retrievalChunkerFilter: 'sheet',
    retrievalEmbeddingFilter: 'embedding',
    retrievalDbFilter: 'store',
    retrievalRerankerFilter: 'reranker',
  });
}

function renderRetrieval(retrievalSmokes, rerankerSmokes = []) {
  const evidence = buildEvidenceRows(retrievalSmokes, rerankerSmokes);
  const opts = matrixOptions();
  fillSelect('retrievalChunkerFilter', opts.chunkers.length ? opts.chunkers : uniq(evidence, 'sheet'), 'All chunkers');
  fillSelect('retrievalEmbeddingFilter', opts.embeddings.length ? opts.embeddings : uniq(evidence, 'embedding'), 'All embeddings');
  fillSelect('retrievalDbFilter', opts.stores.length ? opts.stores : uniq(evidence, 'store'), 'All DBs');
  fillSelect('retrievalRerankerFilter', uniq(evidence, 'reranker'), 'All rerankers');
  const rows = filteredRetrieval(evidence).slice(0, 80);
  table($('retrievalTable'), rows, [
    {key:'query', label:'Query', render:r=>esc((r.query || '').slice(0, 58))},
    {key:'sheet', label:'Chunker'},
    {key:'embedding', label:'Embedding'},
    {key:'store', label:'Vector DB'},
    {key:'reranker', label:'Reranker'},
    {key:'top_pdf', label:'Top PDF/source', render:r=>pdfLink(r.top_pdf, (r.top_pdf || '').slice(0, 58) || '—', r.top_page_number)},
    {key:'initial_topk_count', label:'Hits shown'},
    {key:'evidence_type', label:'Evidence source', render:r=>esc(evidenceStageLabel(r.evidence_type))},
    {key:'evidence_snippet', label:'Retrieved evidence excerpts', render:r=>`<details class="snippet"><summary>${esc(shortEvidenceLabel(r))}</summary>${renderEvidenceDetails(r)}</details>`},
    {key:'retrieval_seconds', label:'Retrieval sec', render:r=>r.retrieval_seconds !== undefined ? `${fmt(r.retrieval_seconds, 3)}s` : '—'},
    {key:'rerank_seconds', label:'Rerank sec', render:r=>r.rerank_seconds !== undefined ? `${fmt(r.rerank_seconds, 3)}s` : '—'},
  ]);
}

function renderHallucination(audit) {
  const summary = audit?.summary || [];
  const details = audit?.details || [];
  const report = audit?.report || {};
  const hint = $('hallucinationHint');
  if (!hint) return;
  hint.textContent = summary.length ? `${report.evaluated_rows || details.length} rows · judge ${report.judge || 'unknown'}` : 'Waiting for audit';
  const cards = $('hallucinationSummary');
  if (cards) {
    if (!summary.length) {
      cards.innerHTML = '<div class="empty-state">Run Evaluate grounding risk or LLM grounding audit after retrieval/reranker results exist.</div>';
    } else {
      cards.innerHTML = summary.slice(0, 6).map(r => `<article class="pipeline-card">
        <div class="rank-label">${esc(r.reranker || 'none')}</div>
        <h3>${esc(pipelineLabel(r))}</h3>
        ${metricBar('Grounded rate', num(r.grounded_rate), 1, `${(num(r.grounded_rate)*100).toFixed(1)}%`)}
        ${metricBar('Hallucination risk', 1 - num(r.hallucination_risk_rate), 1, `${(num(r.hallucination_risk_rate)*100).toFixed(1)}% high risk`)}
        ${metricBar('Support score', num(r.avg_support_score), 1, fmt(r.avg_support_score,3))}
      </article>`).join('');
    }
  }
  table($('hallucinationTable'), details.slice(0, 80), [
    {key:'query', label:'Query', render:r=>esc((r.query || '').slice(0, 72))},
    {key:'sheet', label:'Chunker'},
    {key:'embedding', label:'Embedding'},
    {key:'store', label:'Vector DB'},
    {key:'reranker', label:'Reranker'},
    {key:'verdict', label:'Verdict', render:r=>`<span class="risk ${esc(r.risk || '')}">${esc(r.verdict || '')}</span>`},
    {key:'support_score', label:'Support'},
    {key:'risk', label:'Risk'},
    {key:'reason', label:'Reason', render:r=>esc((r.reason || '').slice(0, 180))},
  ]);
}

function nvidiaResponseBody(smoke) {
  return smoke?.response?.body || {};
}

function nvidiaResultItems(body) {
  const candidates = [
    body?.results,
    body?.documents,
    body?.chunks,
    body?.passages,
    body?.citations,
    body?.data?.results,
    body?.data?.documents,
    body?.data?.chunks,
    body?.response?.results,
  ];
  return candidates.find(v => Array.isArray(v)) || [];
}

function nvidiaAnswerPreview(body) {
  const choice = Array.isArray(body?.choices) ? body.choices[0] : null;
  const value = choice?.message?.content || choice?.text || body?.answer || body?.generated_text || body?.response || body?.text || '';
  return String(value || '').replace(/\s+/g, ' ').trim();
}

function nvidiaSource(hit) {
  return hit?.document_name || hit?.source || hit?.filename || hit?.title || hit?.id || hit?.metadata?.source || hit?.metadata?.filename || hit?.metadata?.pdf_name || '—';
}

function nvidiaSnippet(hit) {
  const raw = hit?.text || hit?.content || hit?.chunk || hit?.paragraph || hit?.page_content || hit?.document?.text || hit?.metadata?.text || '';
  if (raw) return String(raw).replace(/\s+/g, ' ').trim();
  try { return JSON.stringify(hit).slice(0, 240); } catch { return ''; }
}

function renderNvidiaRag(nvidia) {
  const status = $('nvidiaStatus');
  const cards = $('nvidiaCards');
  const artifactHint = $('nvidiaArtifactHint');
  if (!status || !cards) return;
  const health = nvidia?.health || {};
  const smoke = nvidia?.smoke || {};
  const ingestion = nvidia?.ingestion || {};
  const benchmark = nvidia?.benchmark || {};
  const benchmarkMode = $('nvidiaBenchmarkMode')?.value === 'reranked' ? 'reranked' : 'baseline';
  const benchmarkModeLabel = benchmarkMode === 'reranked' ? 'Reranked' : 'Baseline';
  const hasIndependentBenchmarkLanes = ['baseline', 'reranked'].some(mode => {
    const lane = benchmark[mode] || {};
    return Object.keys(lane.report || {}).length || (lane.summary || []).length || (lane.details || []).length;
  });
  const legacyReport = benchmark.report || {};
  const legacySummary = benchmark.summary || [];
  const legacyReranker = legacyReport.reranker_enabled ?? legacyReport.use_reranker ?? legacySummary[0]?.reranker_enabled;
  const legacyMode = legacyReranker === true ? 'reranked' : 'baseline';
  const benchmarkLane = hasIndependentBenchmarkLanes
    ? (benchmark[benchmarkMode] || {})
    : (benchmarkMode === legacyMode ? benchmark : {});
  const benchmarkReport = benchmarkLane.report || {};
  const benchmarkRows = benchmarkLane.summary || [];
  const benchmarkDetails = benchmarkLane.details || [];
  const bestBenchmark = benchmarkReport.ok ? (benchmarkRows[0] || benchmarkReport.best || null) : null;
  const evidenceValue = (...values) => values.find(value => value !== undefined && value !== null && String(value).trim() !== '');
  const numericEvidence = (value) => {
    const parsed = Number.parseFloat(value);
    return evidenceValue(value) === undefined || !Number.isFinite(parsed) ? undefined : parsed;
  };
  const metricValue = (value, digits = 3) => numericEvidence(value)?.toFixed(digits) ?? '—';
  const recallValue = (value) => numericEvidence(value) === undefined ? '—' : `${(numericEvidence(value) * 100).toFixed(1)}%`;
  const evaluatedQueries = evidenceValue(bestBenchmark?.evaluated_queries, benchmarkReport.evaluated_rows);
  const successfulQueries = evidenceValue(bestBenchmark?.successful_queries, benchmarkReport.successful_queries);
  const files = nvidia?.files || [];
  const checks = health.checks || {};
  const checkRows = Object.entries(checks).map(([name, check]) => ({name, ok: !!check.ok, url: check.url || '', status: check.status || '', error: check.error || '', latency_ms: check.latency_ms || ''}));
  const healthyCount = checkRows.filter(r=>r.ok).length;
  const body = nvidiaResponseBody(smoke);
  const resultItems = nvidiaResultItems(body);
  const answerPreview = nvidiaAnswerPreview(body);
  const batches = ingestion.batches || [];
  const uploadedBatches = batches.filter(b => b.upload?.ok).length;
  const smokeError = smoke.response?.error || body?.message || smoke.response?.body?.message || '';
  const benchmarkError = benchmarkReport.first_error || benchmarkReport.error || '';
  const collection = ingestion.collection || smoke.collection || benchmarkReport.collection || 'multimodal_data';
  status.textContent = benchmarkReport.ok ? 'Benchmarked' : health.ok ? 'Healthy' : (health.created_at ? 'Needs attention' : 'Not checked');
  const nextStep = !health.ok ? 'Run health check first.' : !ingestion.ok ? 'Run Ingest current data before smoke/benchmark.' : !smoke.ok ? 'Run Test one query to verify retrieval evidence.' : !benchmarkReport.ok ? `Run NVIDIA ${benchmarkModeLabel.toLowerCase()} benchmark on groundtruth_500.csv.` : `${benchmarkModeLabel} benchmark artifact is ready.`;
  cards.innerHTML = `
    <article class="nvidia-stage ${health.ok ? 'ok' : 'warn'}"><span>01</span><strong>Services</strong><p>${healthyCount}/${checkRows.length || 3} healthy · rag-server, ingestor, frontend</p></article>
    <article class="nvidia-stage ${ingestion.ok ? 'ok' : 'warn'}"><span>02</span><strong>Current data</strong><p>${ingestion.files ? `${ingestion.files.length} files selected · ${uploadedBatches}/${batches.length || 0} batches uploaded` : `Collection ${esc(collection)} has no ingestion artifact yet.`}</p></article>
    <article class="nvidia-stage ${benchmarkReport.ok ? 'ok' : 'warn'}"><span>03</span><strong>${benchmarkModeLabel} benchmark</strong><p>${bestBenchmark ? `R@5 ${recallValue(bestBenchmark.recall_at_5)} · MRR ${metricValue(bestBenchmark.mrr)} · ${metricValue(bestBenchmark.avg_latency_seconds)}s/query` : (benchmarkError || `No NVIDIA ${benchmarkModeLabel.toLowerCase()} benchmark artifact yet.`)}</p></article>
    <article class="nvidia-stage action"><span>Next</span><strong>${esc(nextStep)}</strong><p>${smokeError && !smoke.ok ? esc(smokeError).slice(0, 160) : 'Keep the benchmark path evidence-led: ingest current PDFs, test one query, then run a limited benchmark.'}</p></article>`;

  const serviceRows = checkRows.map(r => ({kind:'health', name:r.name, status:r.ok ? 'ok' : 'check', detail:r.error || r.url, latency:r.latency_ms ? `${fmt(r.latency_ms,1)} ms` : '—'}));
  const artifactRows = files.map(f => ({kind:'artifact', name:f.split('/').pop(), status:'present', detail:f, latency:'—'}));
  const rows = [
    ...serviceRows,
    ...(smoke.created_at ? [{kind:'smoke', name:smoke.mode || 'search', status:smoke.ok ? 'ok' : 'check', detail:smoke.query || smokeError || '', latency:smoke.response?.latency_seconds ? `${fmt(smoke.response.latency_seconds,3)}s` : '—'}] : []),
    ...(ingestion.created_at ? [{kind:'ingestion', name:collection, status:ingestion.ok ? 'ok' : 'check', detail:`${(ingestion.files || []).length} files · ${batches.length} batches`, latency:'—'}] : []),
    ...(benchmarkReport.created_at || Object.keys(benchmarkReport).length ? [{kind:`benchmark ${benchmarkModeLabel.toLowerCase()}`, name:collection, status:benchmarkReport.ok ? 'ok' : 'check', detail:`${evaluatedQueries ?? '—'} evaluated · ${successfulQueries ?? '—'} successful · ${benchmarkReport.groundtruth || ''}`, latency:benchmarkReport.total_seconds ? `${fmt(benchmarkReport.total_seconds,2)}s total` : '—'}] : []),
    ...artifactRows,
  ];
  artifactHint.textContent = `${files.length} NVIDIA JSON artifacts`;
  table($('nvidiaTable'), rows, [
    {key:'kind', label:'Type'},
    {key:'name', label:'Name'},
    {key:'status', label:'Status'},
    {key:'latency', label:'Latency'},
    {key:'detail', label:'Detail', render:r=>esc((r.detail || '').slice(0, 180))},
  ]);

  const smokeRows = smoke.created_at ? (resultItems.length ? resultItems.slice(0, 12).map((hit, i) => ({rank:i+1, source:nvidiaSource(hit), score:hit.score ?? hit.relevance_score ?? hit.similarity ?? '', snippet:nvidiaSnippet(hit)})) : [{rank:1, source:smoke.mode || 'response', score:smoke.response?.status || '', snippet:answerPreview || smokeError || 'No result items found in response body.'}]) : [];
  $('nvidiaSmokeHint') && ($('nvidiaSmokeHint').textContent = smoke.created_at ? `${smoke.mode || 'search'} · ${smoke.ok ? 'ok' : 'check'} · ${resultItems.length || (answerPreview ? 1 : 0)} rows` : 'No smoke artifact yet');
  table($('nvidiaSmokeTable'), smokeRows, [
    {key:'rank', label:'Rank'},
    {key:'source', label:'Source', render:r=>esc((r.source || '').slice(0, 70))},
    {key:'score', label:'Score/status', render:r=>esc(String(r.score || '—')).slice(0, 24)},
    {key:'snippet', label:'Evidence / answer', render:r=>esc((r.snippet || '').slice(0, 260))},
  ]);

  const batchRows = batches.map(b => {
    const taskBody = b.upload?.body || {};
    const task = taskBody.task_id || taskBody.id || taskBody.taskId || '—';
    return {batch:b.batch, files:(b.files || []).join(', '), status:b.upload?.ok ? 'uploaded' : 'check', task, detail:b.upload?.error || b.status?.error || JSON.stringify(b.status?.body || taskBody || {}).slice(0, 180)};
  });
  $('nvidiaIngestHint') && ($('nvidiaIngestHint').textContent = ingestion.created_at ? `${uploadedBatches}/${batches.length} batches uploaded` : 'No ingestion artifact yet');
  table($('nvidiaIngestTable'), batchRows, [
    {key:'batch', label:'Batch'},
    {key:'status', label:'Status'},
    {key:'task', label:'Task'},
    {key:'files', label:'Files', render:r=>esc((r.files || '').slice(0, 120))},
    {key:'detail', label:'Detail', render:r=>esc((r.detail || '').slice(0, 180))},
  ]);

  const benchmarkCards = $('nvidiaBenchmarkCards');
  if (benchmarkCards) {
    if (bestBenchmark) {
      benchmarkCards.innerHTML = `<article class="run-result-card ok"><span>${benchmarkModeLabel} Recall@5</span><strong>${recallValue(bestBenchmark.recall_at_5)}</strong><small>${esc(evaluatedQueries ?? '—')} evaluated queries</small></article>
        <article class="run-result-card"><span>MRR</span><strong>${metricValue(bestBenchmark.mrr)}</strong><small>First relevant rank strength</small></article>
        <article class="run-result-card"><span>Avg sec/query</span><strong>${metricValue(bestBenchmark.avg_latency_seconds)}s</strong><small>${esc(successfulQueries ?? '—')} successful service calls</small></article>`;
    } else {
      benchmarkCards.innerHTML = `<div class="empty-state">${esc(benchmarkError || `Run NVIDIA ${benchmarkModeLabel.toLowerCase()} benchmark after current PDFs are ingested into the collection.`)}</div>`;
    }
  }
  $('nvidiaBenchmarkHint') && ($('nvidiaBenchmarkHint').textContent = `${benchmarkModeLabel} · ${evaluatedQueries ?? '—'} evaluated · ${successfulQueries ?? '—'} successful · ${Object.keys(benchmarkReport).length ? (benchmarkReport.ok ? 'ok' : 'check') : 'not recorded'}`);
  table($('nvidiaBenchmarkTable'), benchmarkRows.length ? benchmarkRows : benchmarkDetails.slice(0, 20), benchmarkRows.length ? [
    {key:'pipeline', label:'Pipeline', render:r=>esc(r.pipeline || 'NVIDIA RAG Blueprint')},
    {key:'evaluated_queries', label:'Queries'},
    {key:'successful_queries', label:'Success'},
    {key:'recall_at_1', label:'R@1'},
    {key:'recall_at_5', label:'R@5'},
    {key:'mrr', label:'MRR'},
    {key:'ndcg_at_5', label:'nDCG@5'},
    {key:'no_hit_queries', label:'No hit'},
    {key:'avg_latency_seconds', label:'Avg sec/query'},
    {key:'winner_score', label:'Score'},
  ] : [
    {key:'query', label:'Query', render:r=>esc((r.query || '').slice(0, 80))},
    {key:'status', label:'Status'},
    {key:'retrieved_count', label:'Hits'},
    {key:'first_relevant_rank', label:'First rank'},
    {key:'latency_seconds', label:'Seconds'},
    {key:'top_snippet', label:'Top snippet', render:r=>esc((r.top_snippet || r.error || '').slice(0, 220))},
  ]);
}

function renderRunResult(kind, payload) {
  const cards = $('runResultCards');
  const tbl = $('runResultTable');
  const output = payload.output || '';
  if (!cards || !tbl) return;
  const ok = payload.exit_code === 0;
  const lines = output.split('\n').filter(Boolean);
  const retrievalLines = lines.filter(l => l.includes('query=') && l.includes('hits='));
  const extracted = retrievalLines.map(line => {
    const get = (rx) => (line.match(rx) || [,''])[1];
    return {combo:get(/^([^q]+?)\s+query=/), query:get(/query="([^"]+)"/), hits:get(/hits=(\d+)/), seconds:get(/seconds=([0-9.]+)/), raw:line};
  });
  cards.innerHTML = `<article class="run-result-card ${ok ? 'ok' : 'warn'}"><span>Status</span><strong>${ok ? 'Passed' : 'Needs attention'}</strong><small>${esc(kind)} · exit code ${esc(payload.exit_code ?? '—')}</small></article>
    <article class="run-result-card"><span>Result rows</span><strong>${extracted.length || lines.length}</strong><small>${extracted.length ? 'retrieval checks parsed' : 'output lines'}</small></article>
    <article class="run-result-card"><span>Next useful view</span><strong>${kind.includes('hallucination') ? 'Grounding audit' : kind.includes('eval') ? 'Quality decision' : 'Retrieval evidence'}</strong><small>Refresh already completed</small></article>`;
  if (extracted.length) {
    table(tbl, extracted, [
      {key:'query', label:'Query'},
      {key:'combo', label:'Pipeline'},
      {key:'hits', label:'Hits'},
      {key:'seconds', label:'Seconds'},
    ]);
  } else {
    table(tbl, lines.slice(0, 20).map((line, i) => ({i:i+1, line})), [
      {key:'i', label:'#'},
      {key:'line', label:'Output', render:r=>esc(r.line).slice(0, 220)},
    ]);
  }
  $('runOutput').textContent = output.slice(0, 8000);
}

function pipelineLabel(r) {
  return `${r.sheet || '—'} · ${r.embedding || '—'} · ${r.store || '—'} · ${r.reranker || 'none'}`;
}

function filterBySelect(rows, mapping) {
  return rows.filter(r => Object.entries(mapping).every(([id, key]) => {
    const v = $(id)?.value || 'all';
    if (v === 'all') return true;
    if (key === '__combo') return pipelineLabel(r) === v;
    return String(r[key] || (key === 'reranker' ? 'none' : '')) === v;
  }));
}

function metricScore(r) {
  if (sourceStateModule()) return sourceStateModule().metricScore(r);
  return num(r.winner_score) || ((0.45*num(r.recall_at_5)) + (0.30*num(r.mrr)) + (0.20*num(r.ndcg_at_5)) + (0.05*(1/(1+num(r.avg_latency_seconds)))));
}

function displayScore(r) {
  return fmt(metricScore(r), 3);
}

function renderEvaluation(evaluation, syncRecommendations = true) {
  const rows = evaluation?.summary || [];
  const section = $('qualitySection');
  if (!section) return;
  const report = evaluation?.report || {};
  const reference = evaluation?.benchmark_reference || {};
  const displayRows = evaluatedRows(evaluation);
  fillSelect('qualityComboFilter', [...new Set(displayRows.map(pipelineLabel))].sort(), 'All pipeline combinations');
  fillSelect('qualityChunkerFilter', uniq(displayRows, 'sheet'), 'All chunkers');
  fillSelect('qualityEmbeddingFilter', uniq(displayRows, 'embedding'), 'All embeddings');
  fillSelect('qualityDbFilter', uniq(displayRows, 'store'), 'All DBs');
  fillSelect('qualityRerankerFilter', uniq(displayRows, 'reranker'), 'All rerankers');
  const filteredRows = filterBySelect(displayRows, {
    qualityComboFilter: '__combo',
    qualityChunkerFilter: 'sheet',
    qualityEmbeddingFilter: 'embedding',
    qualityDbFilter: 'store',
    qualityRerankerFilter: 'reranker',
  });
  const bestRow = filteredRows[0];
  section.classList.toggle('hidden', displayRows.length === 0);
  const evidenceRows = report.groundtruth_rows || evaluation?.reranked?.report?.groundtruth_rows || reference?.report?.query_count || '—';
  const referenceCount = dedupeMetricRows(reference.summary || []).length;
  const referenceNote = referenceCount ? ` · ${referenceCount} benchmark artifact configs` : '';
  $('qualityHint').textContent = filteredRows.length ? `${filteredRows.length}/${displayRows.length} configs · ${evidenceRows} rows${referenceNote}` : 'No matching configs';
  $('qualityInsight').innerHTML = bestRow ? `<strong>Current winner:</strong> ${esc(pipelineLabel(bestRow))} <span>Score ${displayScore(bestRow)} · R@5 ${(num(bestRow.recall_at_5)*100).toFixed(1)}% · MRR ${fmt(bestRow.mrr,3)} · ${fmt(bestRow.avg_latency_seconds,3)}s avg/query across ${esc(bestRow.evaluated_queries || '—')} queries</span>` : '<span>No evaluated combination matches these filters.</span>';
  table($('qualityTable'), filteredRows, [
    {key:'serial', label:'S/No.', render:(r,i)=>String(i + 1)},
    {key:'winner_score', label:'Score', render:displayScore},
    {key:'sheet', label:'Chunker'},
    {key:'embedding', label:'Embedding'},
    {key:'store', label:'Vector DB'},
    {key:'reranker', label:'Reranker', render:r=>esc(r.reranker || 'none')},
    {key:'source', label:'Source', render:r=>esc(r.source || 'groundtruth_eval')},
    {key:'evaluated_queries', label:'Queries'},
    {key:'recall_at_1', label:'R@1', render:r=>fmt(r.recall_at_1,3)},
    {key:'recall_at_3', label:'R@3', render:r=>fmt(r.recall_at_3,3)},
    {key:'recall_at_5', label:'R@5', render:r=>fmt(r.recall_at_5,3)},
    {key:'recall_at_10', label:'R@10', render:r=>fmt(r.recall_at_10,3)},
    {key:'mrr', label:'MRR', render:r=>fmt(r.mrr,3)},
    {key:'precision_at_5', label:'P@5', render:r=>fmt(r.precision_at_5,3)},
    {key:'ndcg_at_5', label:'nDCG@5', render:r=>fmt(r.ndcg_at_5,3)},
    {key:'avg_first_relevant_rank', label:'Avg first relevant rank', render:r=>fmt(r.avg_first_relevant_rank,3)},
    {key:'no_hit_queries', label:'No-hit queries', render:r=>fmtInt(r.no_hit_queries)},
    {key:'avg_latency_seconds', label:'Avg latency/query', render:r=>fmt(r.avg_latency_seconds,3)},
    {key:'cost', label:'Cost lane', render:costLabel},
  ]);
  renderBestMethods(filteredRows.length ? filteredRows : displayRows);
  if (syncRecommendations) {
    renderRecommendationSource({
      source_type: 'official',
      scoring_mode: 'retrieval_labels',
      result_set: 'official',
      configured: displayRows.length,
      evaluated: displayRows.length,
      rows: displayRows,
    });
  }
}

function renderMetricsSource(payload) {
  const context = payload?.source_type === 'uploaded_project'
    ? `Viewing: ${payload.project_label || payload.project_id} · ${payload.groundtruth_label || (payload.scoring_mode === 'retrieval_labels' ? 'Selected ground truth' : 'None (evidence-only)')} · ${payload.timestamp_label || payload.run_id}`
    : payload?.result_set === 'baseline'
      ? `Viewing: Official WNS corpus · Official benchmark ground truth · No-reranker baseline`
      : `Viewing: Official WNS corpus · Official benchmark ground truth · Official reranked matrix`;
  setText('metricsSourceContext', context);
  const incomplete = Array.isArray(payload?.incomplete_rows) ? payload.incomplete_rows : [];
  const incompletePanel = $('incompleteMetricsPanel');
  incompletePanel?.classList.toggle('hidden', incomplete.length === 0);
  setText('incompleteMetricsSummary', `${incomplete.length} incomplete combination${incomplete.length === 1 ? '' : 's'} · unranked`);
  table($('incompleteMetricsTable'), incomplete, [
    {key:'sheet', label:'Chunker'},
    {key:'embedding', label:'Embedding'},
    {key:'store', label:'Vector DB'},
    {key:'reranker', label:'Reranker'},
    {key:'status', label:'Artifact status', render:r=>esc(r.status || 'recorded')},
    {key:'missing_metrics', label:'Missing metrics', render:r=>esc((r.missing_metrics || []).join(', ') || 'unknown')},
  ]);
  if (payload?.scoring_mode !== 'retrieval_labels') {
    const section = $('qualitySection');
    section?.classList.remove('hidden');
    setText('qualityHint', 'Evidence-only · scored metrics unavailable');
    if ($('qualityInsight')) $('qualityInsight').textContent = 'This result has no selected ground truth. Recall, MRR, nDCG, score, and winner claims are intentionally suppressed.';
    table($('qualityTable'), [], []);
    renderBestMethods([]);
    return;
  }
  renderEvaluation({
    summary: Array.isArray(payload?.rows) ? payload.rows : [],
    report: {groundtruth_rows: payload?.evaluated_queries || payload?.query_count || '—'},
  }, false);
  if (payload?.source_type === 'official' && payload?.result_set === 'official') {
    setText(
      'qualityHint',
      `${fmtInt(payload.evaluated ?? payload.rows?.length ?? 0)} ranked · ${fmtInt(incomplete.length)} incomplete/unranked · ${fmtInt(payload.configured ?? 180)} configured`,
    );
  }
}

function groupWinner(rows, key) {
  const grouped = new Map();
  rows.forEach(r => {
    const name = r[key] || '—';
    if (!grouped.has(name)) grouped.set(name, []);
    grouped.get(name).push(r);
  });
  const scored = [...grouped.entries()].map(([name, rs]) => ({
    name,
    configs: rs.length,
    score: rs.reduce((a,r)=>a+metricScore(r),0)/rs.length,
    r5: rs.reduce((a,r)=>a+num(r.recall_at_5),0)/rs.length,
    mrr: rs.reduce((a,r)=>a+num(r.mrr),0)/rs.length,
    ndcg: rs.reduce((a,r)=>a+num(r.ndcg_at_5),0)/rs.length,
    latency: rs.reduce((a,r)=>a+num(r.avg_latency_seconds),0)/rs.length,
  })).sort((a,b)=>b.score-a.score || b.r5-a.r5 || a.latency-b.latency);
  return scored[0];
}

function renderBestMethods(rows) {
  const section = $('bestMethodSection');
  if (!section) return;
  section.classList.toggle('hidden', !rows.length);
  if (!rows.length) return;
  const overall = [...rows].sort((a,b)=>metricScore(b)-metricScore(a) || num(b.recall_at_5)-num(a.recall_at_5))[0];
  const fastest = [...rows].filter(r => num(r.avg_latency_seconds) > 0).sort((a,b)=>num(a.avg_latency_seconds)-num(b.avg_latency_seconds))[0] || overall;
  const bestR1 = [...rows].sort((a,b)=>num(b.recall_at_1)-num(a.recall_at_1) || num(b.mrr)-num(a.mrr))[0];
  const cards = [
    {step:'Overall best combination', value:`${overall.sheet} · ${overall.embedding} · ${overall.store}${overall.reranker && overall.reranker !== 'none' ? ' · ' + overall.reranker : ''}`, note:`Score ${displayScore(overall)} · R@5 ${(num(overall.recall_at_5)*100).toFixed(1)}% · ${fmt(overall.avg_latency_seconds,3)}s`, accent:'gold'},
    {step:'Best chunking method', value:groupWinner(rows,'sheet').name, note:`Average score ${fmt(groupWinner(rows,'sheet').score,3)} across ${groupWinner(rows,'sheet').configs} configs`, accent:'cyan'},
    {step:'Best embedding model', value:groupWinner(rows,'embedding').name, note:`R@5 ${(groupWinner(rows,'embedding').r5*100).toFixed(1)}% · MRR ${fmt(groupWinner(rows,'embedding').mrr,3)}`, accent:'mint'},
    {step:'Best vector DB component', value:groupWinner(rows,'store').name, note:`Avg latency ${fmt(groupWinner(rows,'store').latency,3)}s · score ${fmt(groupWinner(rows,'store').score,3)}`, accent:'rose'},
    {step:'Best reranker', value:groupWinner(rows,'reranker').name || 'none', note:`Average score ${fmt(groupWinner(rows,'reranker').score,3)} across ${groupWinner(rows,'reranker').configs} configs`, accent:'rank'},
    {step:'Lowest avg latency/query', value:`${fastest.store}`, note:`${fastest.sheet} · ${fastest.embedding}${fastest.reranker && fastest.reranker !== 'none' ? ' · ' + fastest.reranker : ''} · ${fmt(fastest.avg_latency_seconds,3)}s`, accent:'speed'},
    {step:'Best first-answer accuracy', value:`${bestR1.sheet}`, note:`${bestR1.embedding} · ${bestR1.store} · R@1 ${(num(bestR1.recall_at_1)*100).toFixed(1)}%`, accent:'rank'},
  ];
  $('bestMethodHint').textContent = `${rows.length} configs ranked`;
  $('bestMethodGrid').innerHTML = cards.map(c => `<article class="best-method-card ${c.accent}"><span>${esc(c.step)}</span><strong>${esc(c.value)}</strong><small>${esc(c.note)}</small></article>`).join('');
}

function evaluatedRows(evaluation) {
  return dedupeMetricRows([
    ...(evaluation?.summary || []),
    ...(evaluation?.reranked?.summary || []),
    ...(evaluation?.benchmark_reference?.summary || []),
  ])
    .map(r => canonicalMetricRow(r, {source: r.source || 'groundtruth_eval'}))
    .sort((a,b)=>metricScore(b)-metricScore(a) || num(b.recall_at_5)-num(a.recall_at_5));
}

function metricBar(label, value, maxValue, detail) {
  const pct = maxValue > 0 ? Math.max(2, Math.min(100, (value / maxValue) * 100)) : 0;
  return `<div class="bar-row"><div class="bar-meta"><strong>${esc(label)}</strong><span>${esc(detail)}</span></div><div class="bar-track"><i style="width:${pct.toFixed(1)}%"></i></div></div>`;
}

function shortPipeline(r, index) {
  const parts = [r.sheet, r.embedding, r.store, r.reranker || 'none'].filter(Boolean);
  return `P${index + 1} ${parts.join(' · ')}`;
}

function renderTop10Visualizations(rows) {
  const hint = $('top10Hint');
  const scoreEl = $('top10ScoreChart');
  const scatterEl = $('qualityLatencyChart');
  if (!scoreEl || !scatterEl) return;
  const top = rows.slice(0, 10);
  if (!top.length) {
    if (hint) hint.textContent = 'Waiting for evaluation';
    scoreEl.innerHTML = '<div class="empty-state">No top 10 pipeline data yet.</div>';
    scatterEl.innerHTML = '<div class="empty-state">No quality-latency plot yet.</div>';
    return;
  }
  if (hint) hint.textContent = `${top.length} best pipelines`;
  const maxScore = Math.max(...top.map(metricScore), 0.001);
  const maxR5 = Math.max(...top.map(r => num(r.recall_at_5)), 0.001);
  const maxMrr = Math.max(...top.map(r => num(r.mrr)), 0.001);
  const bars = top.map((r, i) => {
    const y = 46 + i * 30;
    const label = esc(shortPipeline(r, i)).slice(0, 78);
    return `<g>
      <text x="12" y="${y + 6}" class="svg-label">${label}</text>
      <rect x="255" y="${y - 10}" width="${(metricScore(r)/maxScore*230).toFixed(1)}" height="8" rx="4" class="svg-score" />
      <rect x="255" y="${y}" width="${(num(r.recall_at_5)/maxR5*230).toFixed(1)}" height="8" rx="4" class="svg-r5" />
      <rect x="255" y="${y + 10}" width="${(num(r.mrr)/maxMrr*230).toFixed(1)}" height="8" rx="4" class="svg-mrr" />
      <text x="500" y="${y + 6}" class="svg-value">${displayScore(r)} · R@5 ${(num(r.recall_at_5)*100).toFixed(1)}%</text>
    </g>`;
  }).join('');
  scoreEl.innerHTML = `<svg viewBox="0 0 720 370" role="img" aria-label="Top 10 pipeline metric bars">
    <text x="12" y="24" class="svg-title">Top 10 metric bars</text>
    <text x="255" y="24" class="svg-legend"><tspan class="legend-score">Score</tspan>   <tspan class="legend-r5">Recall@5</tspan>   <tspan class="legend-mrr">MRR</tspan></text>
    ${bars}
  </svg>`;

  const latencies = top.map(r => num(r.avg_latency_seconds));
  const minLat = Math.min(...latencies, 0);
  const maxLat = Math.max(...latencies, 0.001);
  const points = top.map((r, i) => {
    const x = 62 + ((num(r.avg_latency_seconds) - minLat) / (maxLat - minLat || 1)) * 560;
    const y = 304 - (num(r.recall_at_5) / maxR5) * 235;
    const radius = 8 + (metricScore(r) / maxScore) * 12;
    return `<g>
      <circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${radius.toFixed(1)}" class="svg-dot" />
      <text x="${(x + 15).toFixed(1)}" y="${(y + 4).toFixed(1)}" class="svg-value">P${i + 1}</text>
    </g>`;
  }).join('');
  scatterEl.innerHTML = `<svg viewBox="0 0 720 370" role="img" aria-label="Recall versus latency plot">
    <text x="12" y="24" class="svg-title">Quality vs latency</text>
    <line x1="62" y1="304" x2="650" y2="304" class="svg-axis" />
    <line x1="62" y1="58" x2="62" y2="304" class="svg-axis" />
    <text x="560" y="335" class="svg-label">Latency seconds</text>
    <text x="12" y="54" class="svg-label">Recall@5</text>
    ${points}
    <text x="62" y="335" class="svg-value">${fmt(minLat,3)}s</text>
    <text x="610" y="335" class="svg-value">${fmt(maxLat,3)}s</text>
  </svg>`;
}

function renderPipelineComparison(evaluation, scopedRows = null) {
  const rows = Array.isArray(scopedRows)
    ? [...scopedRows].sort((a, b) => metricScore(b) - metricScore(a))
    : evaluatedRows(evaluation);
  const grid = $('comparisonGrid');
  const hint = $('comparisonHint');
  if (!grid) return;
  if (!rows.length) {
    renderTop10Visualizations(rows);
    if (hint) hint.textContent = 'Waiting for evaluation';
    grid.innerHTML = '<div class="empty-state">Run ground-truth evaluation to compare complete pipelines.</div>';
    $('rerankerLiftChart').innerHTML = '<div class="empty-state">No reranker comparison yet.</div>';
    $('stageComparisonChart').innerHTML = '<div class="empty-state">No stage comparison yet.</div>';
    return;
  }
  const top = rows.slice(0, 8);
  renderTop10Visualizations(rows);
  const maxScore = Math.max(...top.map(metricScore));
  const maxR5 = Math.max(...top.map(r => num(r.recall_at_5)));
  const maxMrr = Math.max(...top.map(r => num(r.mrr)));
  const maxLatency = Math.max(...top.map(r => num(r.avg_latency_seconds)));
  if (hint) hint.textContent = `${rows.length} evaluated configs`;
  grid.innerHTML = top.map((r, i) => `
    <article class="pipeline-card">
      <div class="rank-label">Rank ${i + 1}</div>
      <h3>${esc(pipelineLabel(r))}</h3>
      ${metricBar('Score', metricScore(r), maxScore, displayScore(r))}
      ${metricBar('Recall@5', num(r.recall_at_5), maxR5, `${(num(r.recall_at_5)*100).toFixed(1)}%`)}
      ${metricBar('MRR', num(r.mrr), maxMrr, fmt(r.mrr,3))}
      ${metricBar('Avg latency/query', maxLatency - num(r.avg_latency_seconds) + 0.0001, maxLatency, `${fmt(r.avg_latency_seconds,3)}s avg`) + metricBar('Cost lane', /openai|amazon/i.test(`${r.embedding || ''} ${r.reranker || ''}`) ? 0.55 : 1, 1, costLabel(r))}
    </article>`).join('');

  const byReranker = ['none', ...uniq(rows, 'reranker').filter(r => r !== 'none')].filter(Boolean).map(name => {
    const rs = rows.filter(r => (r.reranker || 'none') === name);
    return {name, score: rs.reduce((a,r)=>a+metricScore(r),0)/(rs.length||1), r5: rs.reduce((a,r)=>a+num(r.recall_at_5),0)/(rs.length||1), latency: rs.reduce((a,r)=>a+num(r.avg_latency_seconds),0)/(rs.length||1), configs: rs.length};
  }).filter(r => r.configs).sort((a,b)=>b.score-a.score);
  const maxLift = Math.max(...byReranker.map(r => r.score), 0);
  $('rerankerLiftChart').innerHTML = byReranker.map(r => metricBar(r.name, r.score, maxLift, `Score ${fmt(r.score,3)} · R@5 ${(r.r5*100).toFixed(1)}% · ${fmt(r.latency,3)}s · ${r.configs} configs`)).join('') || '<div class="empty-state">No reranker comparison yet.</div>';

  const stageWinners = [
    ['Chunker', groupWinner(rows, 'sheet')],
    ['Embedding', groupWinner(rows, 'embedding')],
    ['Vector DB component', groupWinner(rows, 'store')],
    ['Reranker', groupWinner(rows, 'reranker')],
  ].filter(([,v]) => v);
  const maxStage = Math.max(...stageWinners.map(([,v]) => v.score), 0);
  $('stageComparisonChart').innerHTML = stageWinners.map(([label, v]) => metricBar(`${label}: ${v.name}`, v.score, maxStage, `Avg score ${fmt(v.score,3)} · R@5 ${(v.r5*100).toFixed(1)}% · ${v.configs} configs`)).join('');
}

const recommendationState = {
  sourceType: 'official',
  official: {source_type: 'official', configured: 180, evaluated: 0},
  projects: [],
  runs: [],
  active: null,
  generation: 0,
  controller: null,
  evidenceController: null,
  evidenceOffset: 0,
  evidenceLoadedCount: 0,
  evidenceComboId: null,
  tablePage: 0,
  tablePayload: null,
  tableResult: null,
};

function recommendationModule() {
  const module = globalThis.PipelineRecommendations;
  if (!module || typeof module.recommendationsForSource !== 'function') throw new Error('Recommendation logic is unavailable');
  return module;
}

function closeRecommendationEvidence() {
  recommendationState.evidenceController?.abort();
  recommendationState.evidenceController = null;
  recommendationState.evidenceOffset = 0;
  recommendationState.evidenceLoadedCount = 0;
  recommendationState.evidenceComboId = null;
  setText('projectEvidenceStatus', '');
  if ($('projectEvidenceRows')) $('projectEvidenceRows').innerHTML = '';
  if ($('projectEvidenceMore')) $('projectEvidenceMore').hidden = true;
  const dialog = $('projectEvidenceDialog');
  if (dialog?.open && typeof dialog.close === 'function') dialog.close();
}

function clearRecommendationView(message) {
  setText('recommendationStatus', message || 'No result source selected');
  setText('recommendationContext', message || 'Choose a result source');
  if ($('recommendationStrip')) $('recommendationStrip').innerHTML = '<div class="empty-state">No recommendations loaded.</div>';
  if ($('recommendationTable')) $('recommendationTable').innerHTML = '<tr><td class="empty-cell">No result rows loaded</td></tr>';
  if ($('pricingLedger')) $('pricingLedger').innerHTML = '<div class="empty-state">No measured commercial usage loaded.</div>';
  if ($('recommendationFilters')) $('recommendationFilters').innerHTML = '';
  if ($('recommendationSort')) $('recommendationSort').innerHTML = '';
  ['top10ScoreChart','qualityLatencyChart','comparisonGrid','rerankerLiftChart','stageComparisonChart'].forEach(id => {
    if ($(id)) $(id).innerHTML = '<div class="empty-state">No trade-off data for the active source.</div>';
  });
  closeRecommendationEvidence();
}

function beginRecommendationRequest() {
  recommendationState.generation += 1;
  recommendationState.controller?.abort();
  recommendationState.controller = new AbortController();
  clearRecommendationView('Loading selected result source…');
  return {generation: recommendationState.generation, signal: recommendationState.controller.signal};
}

function requestIsCurrent(generation) {
  return generation === recommendationState.generation;
}

function metricCell(value, {percent = false} = {}) {
  if (value === null || value === undefined || value === '') return 'Not recorded';
  const number = Number(value);
  if (!Number.isFinite(number)) return 'Not recorded';
  return percent ? `${(number * 100).toFixed(1)}%` : number.toFixed(3);
}

function recommendationPipelineName(row) {
  if (!row) return 'Not available';
  const parts = [
    row.chunker_id || row.sheet,
    row.embedding_id || row.embedding,
    row.vector_store_id || row.store,
    row.reranker_id || row.reranker || 'none',
  ].filter(Boolean);
  return parts.length ? parts.join(' · ') : (row.combo_id || row.label || 'Not available');
}

function recommendationResultRows(result) {
  if (Array.isArray(result?.rows)) return result.rows;
  const completed = result?.completed_rows || result?.completed || [];
  const failed = result?.failed_rows || result?.failed || [];
  return [...completed, ...failed];
}

function roleRow(role) {
  return role?.row || role?.winner || role || null;
}

function recommendationRoleTitle(key, row) {
  if (!row) return 'Not available';
  if (key === 'fastest' && row.state === 'latency_not_recorded') return 'Latency not recorded';
  if (key === 'run_health') return `${fmtInt(row.succeeded ?? 0)}/${fmtInt(row.total ?? 0)} succeeded`;
  if (key === 'evidence_coverage') {
    return row.evidence_count === null || row.evidence_count === undefined
      ? 'Evidence not recorded'
      : `${fmtInt(row.evidence_count)} evidence rows`;
  }
  return recommendationPipelineName(row);
}

function recommendationPricingState(pricing) {
  return String(pricing?.state || pricing?.status || pricing?.code || '').trim();
}

function recommendationPricingText(pricing) {
  const total = Number(pricing?.total_cost_usd ?? pricing?.cost_usd);
  const stateName = recommendationPricingState(pricing);
  const sharedMeasured = stateName === 'measured_usage' && Array.isArray(pricing?.charges)
    && pricing.charges.some(charge => charge?.usage_scope === 'shared_embedding' && charge?.usage !== null && charge?.usage !== undefined);
  if (Number.isFinite(total) && total >= 0 && stateName === 'measured_usage') {
    return sharedMeasured
      ? `$${total.toFixed(6)} combination API cost · shared embedding in pricing ledger`
      : `$${total.toFixed(6)} measured API cost`;
  }
  if (sharedMeasured) return 'Shared embedding cost · see pricing ledger';
  if (stateName === 'no_api_fee') return 'No external model API fee · VM infrastructure excluded';
  if (stateName === 'pricing_unavailable') return 'Pricing unavailable';
  if (stateName === 'usage_missing') return 'Run usage not recorded';
  return 'Run usage not recorded';
}

function recommendationRoleNote(key, row, mode) {
  if (!row) return 'Not available for this source';
  if (key === 'fastest' && row.state === 'latency_not_recorded') return 'No completed combination recorded latency';
  if (key === 'quality') return `nDCG ${metricCell(row.ndcg_at_k ?? row.ndcg_at_5)} · MRR ${metricCell(row.mrr_at_k ?? row.mrr)}`;
  if (key === 'speed' || key === 'fastest') return `${metricCell(row.avg_query_latency_s ?? row.avg_latency_seconds)} seconds per query`;
  if (key === 'evidence_coverage') {
    return row.query_count === null || row.query_count === undefined
      ? 'Query count not recorded'
      : `${fmtInt(row.query_count)} queries represented`;
  }
  if (key === 'run_health') return `${fmtInt(row.failed ?? 0)} failed · ${fmtInt(row.total ?? 0)} total`;
  if (key === 'value') return recommendationPricingText(recommendationModule().pricingForRow(row));
  return mode === 'labelled' ? 'Ranked from retrieval labels' : 'Operational evidence only';
}

function renderRecommendationStrip(result) {
  const mode = result?.mode === 'labelled' ? 'labelled' : 'evidence_only';
  const definitions = mode === 'labelled'
    ? [['quality','Quality'], ['speed','Speed'], ['value','Value']]
    : [['fastest','Fastest'], ['run_health','Run health'], ['evidence_coverage','Evidence coverage']];
  const roles = result?.roles || {};
  $('recommendationStrip').innerHTML = definitions.map(([key, label]) => {
    const row = roleRow(roles[key]);
    const tone = key === 'fastest' ? 'speed' : key === 'run_health' ? 'quality' : key === 'evidence_coverage' ? 'value' : key;
    return `<article class="recommendation-item ${tone}"><span>${esc(label)}</span><strong>${esc(recommendationRoleTitle(key, row))}</strong><small>${esc(recommendationRoleNote(key, row, mode))}</small></article>`;
  }).join('');
}

function qualityMetricCell(row, value, mode) {
  if (mode !== 'labelled') return 'Not applicable';
  const labelledQueries = row?.labelled_queries ?? row?.labelled_query_count;
  if (row?.quality_applicable === false || Number(labelledQueries) === 0) return 'Not applicable';
  return metricCell(value, {percent: true});
}

function recommendationEvidenceCell(payload, row, failed) {
  if (failed) return 'Not applicable';
  const rawCount = row.evidence_count;
  const count = rawCount === null || rawCount === undefined || rawCount === ''
    ? null
    : Number(rawCount);
  if (!Number.isFinite(count) || count < 0) return 'Not recorded';
  const comboId = String(row.combo_id || '');
  if (payload.source_type === 'uploaded_project' && comboId && count > 0) {
    return `<button class="mini-run-btn view-project-evidence" type="button" data-combo-id="${esc(comboId)}">View ${esc(fmtInt(count))} evidence rows</button>`;
  }
  return esc(fmtInt(count));
}

function recommendationWinnerDetails(row, roles) {
  const badges = recommendationModule().rowBadges(row, roles);
  const labels = {
    quality: 'Quality',
    speed: 'Speed',
    value: 'Value',
    no_api_fee: 'No API fee',
  };
  const reasons = {
    quality: 'highest recorded quality',
    speed: 'lowest recorded latency',
    value: 'best measured value posture',
    no_api_fee: 'no external model API fee',
  };
  return {
    badges: badges.map(key => labels[key]).filter(Boolean),
    why: badges.map(key => reasons[key]).filter(Boolean).join(' · '),
  };
}

function renderRecommendationTable(payload, result, {preservePage = false} = {}) {
  const allRows = recommendationResultRows(result);
  const mode = result?.mode === 'labelled' ? 'labelled' : 'evidence_only';
  const k = Number(payload.metric_k) > 0 ? Number(payload.metric_k) : (payload.source_type === 'official' ? 5 : null);
  const recallLabel = k ? `Recall@${k}` : 'Recall';
  const ndcgLabel = k ? `nDCG@${k}` : 'nDCG';
  recommendationState.tablePayload = payload;
  recommendationState.tableResult = result;
  if (!preservePage) recommendationState.tablePage = 0;
  if (!allRows.length) {
    $('recommendationTable').innerHTML = '<tr><td class="empty-cell">No completed or failed combinations in this source</td></tr>';
    if ($('recommendationSort')) $('recommendationSort').innerHTML = '';
    return;
  }
  const pageSize = 10;
  const pageCount = Math.ceil(allRows.length / pageSize);
  recommendationState.tablePage = Math.max(0, Math.min(recommendationState.tablePage, pageCount - 1));
  const start = recommendationState.tablePage * pageSize;
  const rows = allRows.slice(start, start + pageSize);
  const end = start + rows.length;
  if ($('recommendationSort')) {
    $('recommendationSort').innerHTML = pageCount > 1
      ? `<div class="recommendation-pager" aria-label="Recommendation table pages"><span>${esc(`${start + 1}–${end} of ${allRows.length}`)}</span><button type="button" class="mini-run-btn recommendation-page-btn" data-page-delta="-1" ${start === 0 ? 'disabled' : ''}>Previous 10</button><button type="button" class="mini-run-btn recommendation-page-btn" data-page-delta="1" ${end >= allRows.length ? 'disabled' : ''}>Next 10</button></div>`
      : '';
  }
  const head = mode === 'labelled'
    ? ['Pipeline','Status',recallLabel,'MRR',ndcgLabel,'Avg sec/query','Cost posture','Evidence']
    : ['Pipeline','Status','Queries','Avg sec/query','Cost posture','Evidence'];
  $('recommendationTable').innerHTML = `<thead><tr>${head.map(label => `<th>${esc(label)}</th>`).join('')}</tr></thead><tbody>${rows.map((row, index) => {
    const failed = row.status && row.status !== 'completed';
    const pricing = recommendationModule().pricingForRow(row);
    const comboId = String(row.combo_id || '');
    const evidence = recommendationEvidenceCell(payload, row, failed);
    const classes = [
      roleRow(result?.roles?.quality)?.combo_id === comboId ? 'recommendation-row-quality' : '',
      roleRow(result?.roles?.speed || result?.roles?.fastest)?.combo_id === comboId ? 'recommendation-row-speed' : '',
      roleRow(result?.roles?.value)?.combo_id === comboId ? 'recommendation-row-value' : '',
    ].filter(Boolean).join(' ');
    const winner = recommendationWinnerDetails(row, result?.roles);
    const badges = winner.badges.length
      ? `<div class="recommendation-badges">${winner.badges.map(label => `<span>${esc(label)}</span>`).join('')}</div>`
      : '';
    const why = winner.why ? `<small class="recommendation-why">Why it stands out: ${esc(winner.why)}</small>` : '';
    const pipeline = `<div class="recommendation-pipeline"><span class="recommendation-rank">#${start + index + 1}</span><strong>${esc(recommendationPipelineName(row))}</strong></div>${badges}${why}`;
    const prefix = `<td data-label="Pipeline">${pipeline}</td><td data-label="Status"><span class="badge ${failed ? 'warn' : 'ok'}">${esc(row.status || 'completed')}</span></td>`;
    const operationalCells = `<td data-label="Queries">${esc(metricCell(row.query_count))}</td><td data-label="Avg sec/query">${esc(metricCell(row.avg_query_latency_s ?? row.avg_latency_seconds))}</td><td data-label="Cost posture">${esc(recommendationPricingText(pricing))}</td><td data-label="Evidence">${evidence}</td>`;
    const qualityCells = `<td data-label="${esc(recallLabel)}">${esc(qualityMetricCell(row, row.recall_at_k ?? row.recall_at_5, mode))}</td><td data-label="MRR">${esc(qualityMetricCell(row, row.mrr_at_k ?? row.mrr, mode))}</td><td data-label="${esc(ndcgLabel)}">${esc(qualityMetricCell(row, row.ndcg_at_k ?? row.ndcg_at_5, mode))}</td><td data-label="Avg sec/query">${esc(metricCell(row.avg_query_latency_s ?? row.avg_latency_seconds))}</td><td data-label="Cost posture">${esc(recommendationPricingText(pricing))}</td><td data-label="Evidence">${evidence}</td>`;
    return `<tr class="${classes}">${prefix}${mode === 'labelled' ? qualityCells : operationalCells}</tr>`;
  }).join('')}</tbody>`;
}

function renderPricingLedger(payload) {
  const raw = recommendationModule().pricingLedger(payload);
  const entries = Array.isArray(raw) ? raw : (raw?.entries || []);
  if (!entries.length) {
    $('pricingLedger').innerHTML = '<div class="pricing-entry"><strong>No measured commercial usage</strong><span>No external model API fee · VM infrastructure excluded, or run usage not recorded.</span></div>';
    return;
  }
  $('pricingLedger').innerHTML = entries.map(entry => {
    const model = entry.model_id || entry.commercial_model_id || entry.label || entry.usage_key || 'Commercial model';
    const rate = Number(entry.usd_rate ?? entry.rate_usd);
    const usage = entry.usage ?? entry.quantity ?? entry.input_tokens ?? entry.search_units;
    const rawCost = entry.cost_usd;
    const cost = rawCost === null || rawCost === undefined || rawCost === '' ? NaN : Number(rawCost);
    const rateText = Number.isFinite(rate) ? `$${rate} / ${entry.unit || 'unit'}` : 'Pricing unavailable';
    const usageText = usage === null || usage === undefined ? 'Run usage not recorded' : `${usage} measured`;
    const costText = Number.isFinite(cost) && cost >= 0
      ? `$${cost.toFixed(6)} measured ${entry.usage_scope === 'shared_embedding' ? 'run' : 'combination'} cost`
      : 'Computed cost not recorded';
    return `<div class="pricing-entry"><strong>${esc(model)}</strong><span>${esc(rateText)} · ${esc(usageText)} · ${esc(costText)}</span></div>`;
  }).join('');
}

function clearUploadedTradeoffs() {
  ['top10ScoreChart','qualityLatencyChart','comparisonGrid','rerankerLiftChart','stageComparisonChart'].forEach(id => {
    if ($(id)) $(id).innerHTML = '<div class="empty-state">Uploaded-run metrics are shown in the exact table above; official @5 charts are not reused.</div>';
  });
}

function renderRecommendationSource(payload) {
  const result = recommendationModule().recommendationsForSource(payload);
  recommendationState.active = payload;
  const uploaded = payload.source_type === 'uploaded_project';
  const baseline = !uploaded && payload.result_set === 'baseline';
  recommendationState.sourceType = uploaded ? 'uploaded_project' : 'official';
  setText('recommendationStatus', uploaded ? (payload.run_state || 'Uploaded matrix run') : (baseline ? 'No-reranker baseline' : 'Official benchmark'));
  const context = uploaded
    ? `Viewing: ${payload.project_label || payload.project_id} · ${payload.run_id} · ${payload.scoring_mode === 'retrieval_labels' ? `retrieval labels @${payload.metric_k}` : 'evidence only'}`
    : baseline
      ? `Viewing: No-reranker baseline · ${fmtInt(payload.rows?.length ?? 0)} measured combinations · separate from the official matrix`
      : `Viewing: Official WNS benchmark · ${fmtInt(payload.configured ?? 180)} configured combinations · ${fmtInt(payload.evaluated ?? payload.rows?.length ?? 0)} evaluated`;
  setText('recommendationContext', context);
  setText('recommendationModeNote', result.mode === 'labelled' ? 'Quality, speed, and honest cost posture for the active source.' : 'Operational speed, run health, and evidence coverage. Quality is not claimed without labels.');
  renderRecommendationStrip(result);
  renderRecommendationTable(payload, result);
  renderPricingLedger(payload);
  if (uploaded) clearUploadedTradeoffs();
  return result;
}

function officialRecommendationPayload(resultSet = 'official') {
  const evaluation = state.operational?.evaluation || {};
  const descriptor = recommendationState.official || {};
  const sourceState = sourceStateModule();
  if (!sourceState) throw new Error('Dashboard source-state module is unavailable');
  if (resultSet === 'baseline') {
    return sourceState.baselineResultPayload(evaluation, state.operational?.benchmark_detail_evidence || []);
  }
  return sourceState.officialResultPayload(
    evaluation,
    {
      configured: descriptor.configured ?? 180,
      evaluated: descriptor.evaluated,
      expected_keys: evaluation?.benchmark_reference?.report?.expected_keys,
    },
    state.operational?.benchmark_detail_evidence || [],
  );
}

function setSelectOptionLabel(selectId, value, label) {
  const select = $(selectId);
  const option = Array.from(select?.options || []).find(item => item.value === value);
  if (option) option.textContent = label;
}

function showOfficialRecommendations() {
  recommendationState.sourceType = 'official';
  const resultSet = $('recommendationOfficialSet')?.value === 'baseline' ? 'baseline' : 'official';
  const payload = officialRecommendationPayload(resultSet);
  const baselineCount = officialRecommendationPayload('baseline')?.rows.length || 0;
  const baselineLabel = `No-reranker baseline — ${baselineCount} measured`;
  setSelectOptionLabel('recommendationOfficialSet', 'baseline', baselineLabel);
  setSelectOptionLabel('metricsResultSet', 'baseline', baselineLabel);
  renderCanonicalResultPayload(payload);
}

function renderCanonicalResultPayload(payload) {
  const rows = Array.isArray(payload?.rows) ? payload.rows : [];
  const scored = payload?.scoring_mode === 'retrieval_labels';
  const officialMatrix = payload?.source_type === 'official' && payload?.result_set === 'official';
  const stable = !officialMatrix || payload?.matrix_complete === true;
  const best = scored && stable ? rows[0] || null : null;
  const fastest = scored && stable
    ? rows.filter(row => num(row.avg_latency_seconds) > 0).sort((a, b) => num(a.avg_latency_seconds) - num(b.avg_latency_seconds))[0] || null
    : null;
  setText('bestR5Status', best ? `${(num(best.recall_at_5) * 100).toFixed(1)}%` : '—');
  setText('bestConfigNote', best
    ? pipelineLabel(best)
    : (officialMatrix && !stable
      ? `Final winner pending · ${fmtInt(payload.evaluated ?? rows.length)}/${fmtInt(payload.configured ?? 180)} fully scored`
      : (scored ? 'best accuracy score' : 'quality unavailable without labels')));
  setText('bestLatencyStatus', fastest ? `${fmt(fastest.avg_latency_seconds, 3)}s` : '—');
  setText('bestLatencyNote', fastest
    ? `${pipelineLabel(fastest)} · avg/query`
    : (officialMatrix && !stable ? 'Final latency winner pending full matrix' : 'average query latency'));
  renderRecommendationSource(payload);
  if (!stable) {
    const evaluated = fmtInt(payload.evaluated ?? rows.length);
    const configured = fmtInt(payload.configured ?? 180);
    setText('recommendationStatus', `Provisional recommendations · ${evaluated}/${configured} complete`);
    setText(
      'recommendationModeNote',
      `Rankings update as validated 500-query results arrive. Final winners require ${configured}/${configured}.`,
    );
  }
  recommendationState.active = payload;
  renderMetricsSource(payload);
  syncSourceSelectorControls();
  if (scored) renderPipelineComparison(state.operational?.evaluation || {}, stable ? rows : []);
}

function populateRecommendationProjects() {
  const select = $('recommendationProject');
  if (!select) return;
  select.innerHTML = recommendationState.projects.length
    ? recommendationState.projects.map(project => `<option value="${esc(project.project_id)}">${esc(project.label || project.project_id)}</option>`).join('')
    : '<option value="">No matrix projects available</option>';
}

function populateRecommendationDatasets() {
  const select = $('recommendationDataset');
  if (!select) return;
  const current = select.value;
  select.innerHTML = [
    '<option value="official">Official WNS corpus</option>',
    ...recommendationState.projects.map(project => `<option value="${esc(project.dataset_id || `project:${project.project_id}`)}">${esc(project.label || project.project_id)}</option>`),
  ].join('');
  if (Array.from(select.options || []).some(option => option.value === current)) select.value = current;
}

function populateRecommendationGroundtruths() {
  const select = $('recommendationGroundtruth');
  if (!select) return;
  if (recommendationState.sourceType === 'official') {
    select.innerHTML = '<option value="groundtruth:official">Official benchmark ground truth</option>';
    return;
  }
  const current = select.value;
  const unique = new Map();
  recommendationState.runs.forEach(run => unique.set(
    run.groundtruth_id || 'groundtruth:none',
    run.groundtruth_label || 'None (evidence-only)',
  ));
  select.innerHTML = unique.size
    ? [...unique].map(([id, label]) => `<option value="${esc(id)}">${esc(label)}</option>`).join('')
    : '<option value="">No ground truth with completed runs</option>';
  if (Array.from(select.options || []).some(option => option.value === current)) select.value = current;
}

function matchingRecommendationRuns() {
  const groundtruthId = $('recommendationGroundtruth')?.value;
  return recommendationState.runs.filter(run => !groundtruthId || run.groundtruth_id === groundtruthId);
}

function populateRecommendationRuns() {
  const select = $('recommendationRun');
  if (!select) return;
  const runs = matchingRecommendationRuns();
  select.innerHTML = runs.length
    ? runs.map(run => `<option value="${esc(run.run_id)}">${esc(run.timestamp_label || run.completed_at || run.created_at || run.run_id)}</option>`).join('')
    : '<option value="">No completed run for this dataset and ground truth</option>';
  syncSourceSelectorControls();
}

function copySelectState(sourceId, targetId) {
  const source = $(sourceId);
  const target = $(targetId);
  if (!source || !target) return;
  if (source.options && typeof target.replaceChildren === 'function') {
    const options = Array.from(source.options)
      .filter(option => typeof option.cloneNode === 'function')
      .map(option => option.cloneNode(true));
    target.replaceChildren(...options);
  }
  target.value = source.value;
  target.disabled = source.disabled;
}
function syncSourceSelectorControls() {
  copySelectState('recommendationDataset', 'metricsDataset');
  copySelectState('recommendationGroundtruth', 'metricsGroundtruth');
  if (recommendationState.sourceType === 'official') {
    copySelectState('recommendationOfficialSet', 'metricsResultSet');
  } else {
    copySelectState('recommendationRun', 'metricsResultSet');
  }
}

async function loadResultSources() {
  const {generation, signal} = beginRecommendationRequest();
  try {
    const payload = await api('/api/result-sources', {signal});
    if (!requestIsCurrent(generation)) return;
    recommendationState.official = payload.official || recommendationState.official;
    recommendationState.projects = payload.projects || [];
    populateRecommendationProjects();
    populateRecommendationDatasets();
    populateRecommendationGroundtruths();
    syncSourceSelectorControls();
    if ($('globalDataset') && $('globalGroundtruth')) await applyGlobalSourceContext();
    else if (recommendationState.sourceType === 'official') showOfficialRecommendations();
  } catch (error) {
    if (error?.name === 'AbortError' || !requestIsCurrent(generation)) return;
    clearRecommendationView('Result sources unavailable');
  }
}

async function loadProjectRuns(projectId) {
  const {generation, signal} = beginRecommendationRequest();
  try {
    const payload = await api(`/api/project-runs?project_id=${encodeURIComponent(projectId)}`, {signal});
    if (!requestIsCurrent(generation)) return;
    recommendationState.runs = Array.isArray(payload) ? payload : (payload.runs || []);
    populateRecommendationGroundtruths();
    populateRecommendationRuns();
    const runId = $('recommendationRun')?.value;
    if (runId) await loadProjectRun(projectId, runId);
    else clearRecommendationView('No completed run for this dataset and ground truth');
  } catch (error) {
    if (error?.name === 'AbortError' || !requestIsCurrent(generation)) return;
    recommendationState.runs = [];
    populateRecommendationRuns();
    clearRecommendationView('Project runs unavailable');
  }
}

async function loadProjectRun(projectId, runId) {
  const {generation, signal} = beginRecommendationRequest();
  const path = `/api/project-run-results?project_id=${encodeURIComponent(projectId)}&run_id=${encodeURIComponent(runId)}`;
  try {
    const payload = await api(path, {signal});
    if (!requestIsCurrent(generation)) return;
    const selectedDataset = $('recommendationDataset')?.value;
    const selectedGroundtruth = $('recommendationGroundtruth')?.value;
    if (
      payload?.source_type !== 'uploaded_project'
      || payload?.project_id !== projectId
      || payload?.run_id !== runId
      || (selectedDataset && payload?.dataset_id !== selectedDataset)
      || (selectedGroundtruth && payload?.groundtruth_id !== selectedGroundtruth)
    ) {
      throw new Error('Project run response identity mismatch');
    }
    const normalizedPayload = sourceStateModule()?.normalizeResultPayload(payload) || payload;
    renderCanonicalResultPayload(normalizedPayload);
    syncSourceSelectorControls();
  } catch (error) {
    if (error?.name === 'AbortError' || !requestIsCurrent(generation)) return;
    recommendationState.active = null;
    clearRecommendationView('Selected project run is unavailable');
  }
}

async function loadProjectEvidencePage({append = false} = {}) {
  const active = recommendationState.active;
  const comboId = recommendationState.evidenceComboId;
  if (!active || active.source_type !== 'uploaded_project' || !comboId) return;
  recommendationState.evidenceController?.abort();
  const controller = new AbortController();
  recommendationState.evidenceController = controller;
  const offset = append ? recommendationState.evidenceOffset : 0;
  const ownership = `${active.project_id}|${active.run_id}|${comboId}`;
  setText('projectEvidenceStatus', 'Loading evidence…');
  const path = `/api/project-run-evidence?project_id=${encodeURIComponent(active.project_id)}&run_id=${encodeURIComponent(active.run_id)}&combo_id=${encodeURIComponent(comboId)}&limit=25&offset=${offset}`;
  try {
    const page = await api(path, {signal: controller.signal});
    const current = recommendationState.active;
    if (controller !== recommendationState.evidenceController || `${current?.project_id}|${current?.run_id}|${recommendationState.evidenceComboId}` !== ownership) return;
    const nextOffset = page.next_offset;
    if (
      nextOffset !== null
      && nextOffset !== undefined
      && (!Number.isInteger(nextOffset) || nextOffset <= offset)
    ) {
      throw new Error('Evidence cursor did not advance');
    }
    const html = (page.rows || []).map(row => `<article class="evidence-dialog-row"><strong>${esc(row.query || row.query_id || 'Query')}</strong><p>${esc(row.excerpt || 'Evidence text not recorded')}</p><span>${esc(row.source_name || 'Source not recorded')}${row.page_number ? ` · page ${esc(row.page_number)}` : ''} · rank ${esc(row.rank ?? '—')}</span></article>`).join('');
    $('projectEvidenceRows').innerHTML = append ? $('projectEvidenceRows').innerHTML + html : (html || '<div class="empty-state">No evidence rows for this combination.</div>');
    recommendationState.evidenceOffset = page.next_offset ?? 0;
    const loadedNow = Array.isArray(page.rows) ? page.rows.length : 0;
    recommendationState.evidenceLoadedCount = append
      ? recommendationState.evidenceLoadedCount + loadedNow
      : loadedNow;
    $('projectEvidenceMore').hidden = page.next_offset === null || page.next_offset === undefined;
    setText('projectEvidenceStatus', `${fmtInt(recommendationState.evidenceLoadedCount)} evidence rows loaded`);
  } catch (error) {
    if (error?.name === 'AbortError' || controller !== recommendationState.evidenceController) return;
    setText('projectEvidenceStatus', 'Evidence unavailable for this combination');
    $('projectEvidenceMore').hidden = true;
  }
}

function openProjectEvidence(comboId) {
  closeRecommendationEvidence();
  recommendationState.evidenceComboId = comboId;
  setText('projectEvidenceTitle', `Combination evidence · ${comboId}`);
  const dialog = $('projectEvidenceDialog');
  if (dialog && typeof dialog.showModal === 'function') dialog.showModal();
  loadProjectEvidencePage().catch(console.error);
}

function showPage(page) {
  document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.toggle('active', btn.dataset.page === page));
  document.querySelectorAll('[data-page-panel]').forEach(panel => panel.classList.toggle('active', panel.dataset.pagePanel === page));
  history.replaceState(null, '', `#${page}`);
}

function showMetricsCombination(combo) {
  if (!combo) return;
  showPage('metrics');
  const payload = recommendationState.active || officialRecommendationPayload('official');
  ['qualityChunkerFilter', 'qualityEmbeddingFilter', 'qualityDbFilter', 'qualityRerankerFilter'].forEach(id => {
    if ($(id)) $(id).value = 'all';
  });
  renderMetricsSource(payload);
  if ($('qualityComboFilter')) $('qualityComboFilter').value = combo;
  renderMetricsSource(payload);
  $('qualitySection')?.scrollIntoView?.({behavior: 'smooth', block: 'start'});
}

async function uploadDataset(ev) {
  ev.preventDefault();
  const file = $('datasetFile')?.files?.[0];
  if (!file) {
    $('uploadStatus').textContent = 'Choose file';
    $('uploadOutput').textContent = 'Select a PDF, ZIP, CSV, XLSX, TXT, or MD first.';
    return;
  }
  const uploadType = $('uploadType')?.value || 'dataset';
  const form = new FormData();
  form.append('file', file);
  form.append('label', $('datasetLabel')?.value || file.name.replace(/\.[^.]+$/, ''));
  form.append('upload_type', uploadType);
  $('uploadStatus').textContent = 'Uploading';
  $('uploadOutput').textContent = uploadType === 'groundtruth'
    ? `Validating and registering ${file.name} as reusable ground truth...`
    : `Uploading ${file.name} into isolated project...`;
  const res = await fetch('/api/upload-dataset', {method: 'POST', body: form});
  const payload = await res.json();
  $('uploadStatus').textContent = res.ok ? (uploadType === 'groundtruth' ? 'Ground truth registered' : 'Uploaded project') : 'Error';
  $('uploadOutput').textContent = JSON.stringify(payload, null, 2);
  if (res.ok) {
    await refresh();
  }
}

async function uploadRunQueries() {
  const file = $('runQueryFile')?.files?.[0];
  if (!file) throw new Error('Choose a TXT, CSV, or XLSX query file first.');

  const form = new FormData();
  form.append('file', file);
  form.append('upload_type', 'queries');
  const status = $('runQueryUploadStatus');
  if (status) status.textContent = `Validating ${file.name}...`;

  const res = await fetch('/api/upload-dataset', {method: 'POST', body: form});
  const payload = await res.json();
  if (!res.ok) {
    const message = payload?.error?.message || payload?.message || 'Query upload failed validation.';
    throw new Error(message);
  }
  if (!Array.isArray(payload.queries) || !payload.queries.length) {
    throw new Error('The query file did not contain any validated queries.');
  }

  $('runQueries').value = payload.queries.join('\n');
  if (status) status.textContent = `${payload.queries.length} validated queries loaded from ${file.name}.`;
}


function readinessAuditLabel(repo) {
  const status = repo?.audit_status || 'unknown';
  if (status === 'missing') return 'extraction audit missing';
  if (status === 'empty') return 'extraction audit empty';
  if (status === 'text_only_fallback') return 'text-only fallback; MinerU/layout extraction not verified';
  if (status === 'review_required') return 'extraction audit requires review';
  if (status === 'clean') {
    const parsers = Object.entries(repo?.parser_counts || {}).map(([name, count]) => `${name}: ${count}`).join(', ');
    return parsers ? `clean extraction audit · ${parsers}` : 'clean extraction audit';
  }
  return 'extraction audit state unknown';
}

function documentReadinessSummary(repo) {
  const total = Number(repo?.total || 0);
  const ready = Number(repo?.ready_count || 0);
  const review = Number(repo?.review_count || 0);
  if (repo?.display_source === 'last_successful') {
    const latest = repo?.latest_attempt || {};
    return `${ready}/${total} ready · preserved clean snapshot · latest attempt: ${readinessAuditLabel(latest)}, ${Number(latest.ready_count || 0)}/${Number(latest.total || total)} ready · ${Number(latest.review_count || 0)} review`;
  }
  return `${ready}/${total} ready · ${review} review · ${readinessAuditLabel(repo)}`;
}

function renderDocumentRepository(repo) {
  const rows = repo?.rows || [];
  const hint = $('documentRepositoryHint');
  if (hint) hint.textContent = rows.length ? documentReadinessSummary(repo) : 'No repository scan yet';
  table($('documentRepositoryTable'), rows, [
    {key:'pdf_name', label:'PDF', render:r=>pdfLink(r.pdf_name, (r.pdf_name || '').slice(0, 88) || '—')},
    {key:'status', label:'Readiness'},
    {key:'chunked_rows', label:'Chunk rows'},
    {key:'pages', label:'Pages'},
    {key:'size_mb', label:'Size MB'},
    {key:'parser_method', label:'Parser', render:r=> (r.parser_method === 'PyPDF2_fallback' ? '<span class="badge warn">text-only fallback</span>' : esc(r.parser_method || '—'))},
    {key:'repository_path', label:'Path', render:r=>`<code>${esc(r.repository_path || '')}</code>`},
    {key:'note', label:'Note'},
  ]);
}

function syncStatusDialog() {
  const pairs = [
    ['pdfStatus', 'dialogPdfStatus'],
    ['pdfNote', 'dialogPdfNote'],
    ['comboStatus', 'dialogComboStatus'],
    ['comboNote', 'dialogComboNote'],
    ['retrievalStatus', 'dialogRetrievalStatus'],
    ['retrievalNote', 'dialogRetrievalNote'],
  ];
  pairs.forEach(([from, to]) => setText(to, $(from)?.textContent || '—'));
}

function openStatusDialog() {
  const dialog = $('statusSummaryDialog');
  if (!dialog) return;
  syncStatusDialog();
  if (typeof dialog.showModal === 'function') dialog.showModal();
  else dialog.setAttribute('open', '');
}

function operationalRerankRows() {
  const op = state.operational || {};
  return [...(op.reranker_smokes || []), ...(op.benchmark_detail_evidence || [])];
}

function renderDatasetStatusSummary() {
  const op = state.operational || {};
  const datasetId = $('globalDataset')?.value || 'dataset:wns-default';
  const groundtruthId = $('globalGroundtruth')?.value || 'groundtruth:none';
  const dataset = (state.sourceCatalog?.datasets || []).find(row => row.id === datasetId);

  if (datasetId !== 'dataset:wns-default' && dataset) {
    const documentCount = Number(dataset.document_count || 0);
    const readyCount = dataset.ready ? documentCount : 0;
    const validation = String(dataset.validation || (dataset.ready ? 'ready' : 'not ready')).replaceAll('_', ' ');
    setText('pdfStatus', `${fmtInt(readyCount)}/${fmtInt(documentCount)}`);
    setText('pdfNote', `${dataset.label || 'Selected project'} · ${validation}`);
    setText('groundTruthStatus', groundtruthId === 'groundtruth:none' ? 'Evidence-only' : 'Selected, not evaluated');
  } else {
    const repo = op.document_repository || {};
    const evaluation = op.evaluation || {};
    setText('pdfStatus', `${repo.ready_count || 0}/${repo.total || op.known_pdf_count || 0}`);
    setText('pdfNote', documentReadinessSummary(repo));
    setText('groundTruthStatus', groundtruthId === 'groundtruth:none'
      ? 'Evidence-only'
      : (evaluation?.report?.groundtruth_rows
        ? `${evaluation.report.groundtruth_rows} rows evaluated`
        : ((evaluation?.groundtruth_files || []).length ? 'Loaded, not evaluated' : 'Pending')));
  }
  syncStatusDialog();
}

function renderOperational() {
  const op = state.operational || {};
  const ingestion = op.ingestion || {};
  const rows = ingestion.rows || [];
  const latest = latestRows(rows);
  const retrieval = op.retrieval_smokes || [];
  const rerank = operationalRerankRows();
  const benchmarkEvidence = op.benchmark_detail_evidence || [];
  const health = op.service_health || [];
  const snapshot = op.vm_snapshot || {};
  const optsForStatus = matrixOptions();
  const embeddings = optsForStatus.embeddings.length ? optsForStatus.embeddings : uniq(latest, 'embedding');
  const stores = optsForStatus.stores.length ? optsForStatus.stores : uniq(latest, 'store');
  const officialPayload = recommendationState.sourceType === 'official' && globalThis.PipelineRecommendations
    ? officialRecommendationPayload('official')
    : null;

  $('modeLabel').textContent = 'Live artifacts';
  $('latestRun').textContent = ingestion.latest_run_id || snapshot?.ingestion?.latest_run_id || '—';
  $('snapshotAt').textContent = snapshot.created_at ? new Date(snapshot.created_at).toLocaleString() : '—';
  renderDatasetStatusSummary();
  $('comboStatus').textContent = String(op.known_matrix_count || ingestion.combo_count || latest.length || 0);
  $('comboNote').textContent = op.options_formula || 'chunkers × embeddings × vector DBs × retrieval × rerankers';
  const evidenceRowsLoaded = retrieval.length + rerank.length;
  $('retrievalStatus').textContent = String(evidenceRowsLoaded);
  $('retrievalNote').textContent = `${fmtInt(evidenceRowsLoaded)} evidence display rows loaded. Raw artifacts on disk: ${fmtInt(op.retrieval_smoke_total || 0)} retrieval + ${fmtInt(op.reranker_smoke_total || 0)} reranker. Benchmark-detail rows loaded: ${fmtInt(benchmarkEvidence.length)}. Not total document chunks.`;
  setText('embeddingStatus', embeddings.length ? embeddings.join(' + ') : '—');
  setText('dbStatus', stores.length ? stores.join(' + ') : '—');
  syncStatusDialog();
  $('ingestionHint').textContent = `${latest.length} latest successful component rows`;
  $('rerankHint').textContent = `${rerank.length} artifacts`;
  $('artifactHint').textContent = `${state.files.length} files`;
  $('healthHint').textContent = `${health.filter(h => h.ok).length}/${health.length || 0} healthy`;
  const pdfAudit = op.pdf_audit || {};
  $('pdfAuditHint') && ($('pdfAuditHint').textContent = pdfAudit.total ? `${pdfAudit.ok_count || 0}/${pdfAudit.total} ready · review hidden from KPI` : 'No audit file yet');

  renderCoverage(rows);
  renderServices(health);
  renderRetrieval(retrieval, rerank);
  if (officialPayload) renderCanonicalResultPayload(officialPayload);
  renderHallucination(op.hallucination || {});
  renderNvidiaRag(op.nvidia_rag || {});
  renderDocumentRepository(op.document_repository || {});
  updateSelectedMatrixCount();

  table($('ingestionTable'), latest.slice(0, 18), [
    {key:'run_id', label:'Run'},
    {key:'sheet', label:'Chunker'},
    {key:'embedding', label:'Embedding'},
    {key:'store', label:'Vector DB'},
    {key:'chunk_count', label:'Chunks'},
    {key:'collection_or_table', label:'Collection/table', render:r=>`<code>${esc((r.collection_or_table || '').slice(0, 46))}</code>`},
    {key:'total_store_seconds', label:'Store sec', render:r=>fmt(r.total_store_seconds, 2)},
    {key:'search_hits', label:'Hits'},
    {key:'status', label:'Status', render:r=>`<span class="badge ok">ok</span>`}
  ]);
  table($('rerankerTable'), rerank.slice(0, 10), [
    {key:'reranker', label:'Reranker'},
    {key:'query', label:'Query', render:r=>esc((r.query || '').slice(0, 58))},
    {key:'sheet', label:'Chunker'},
    {key:'embedding', label:'Embedding'},
    {key:'rerank_seconds', label:'Sec', render:r=>fmt(r.rerank_seconds, 3)},
    {key:'retrieved_count', label:'Initial K'},
    {key:'artifact', label:'Artifact', render:r=>`<code>${esc((r.artifact || '').slice(0, 56))}</code>`}
  ]);
  table($('pdfAuditTable'), (pdfAudit.rows || []).slice(0, 80), [
    {key:'pdf_name', label:'PDF', render:r=>esc((r.pdf_name || '').slice(0, 72))},
    {key:'status', label:'Status'},
    {key:'parser_method', label:'Parser', render:r=> (r.parser_method === 'PyPDF2_fallback' ? '<span class="badge warn">text-only fallback</span>' : esc(r.parser_method || '—'))},
    {key:'total_pages', label:'Pages'},
    {key:'text_chars', label:'Text chars'},
    {key:'image_count', label:'Images'},
    {key:'table_count', label:'Tables'},
    {key:'formula_count', label:'Formulas'},
    {key:'row_count', label:'Rows'},
    {key:'error', label:'Error', render:r=>esc((r.error || '').slice(0, 120))},
  ]);
  $('files').innerHTML = state.files.length ? state.files.map(f => `<li><code>${esc(f)}</code></li>`).join('') : '<li>No artifact files listed yet</li>';
}

function datasetChunkSummary(row) {
  const counts = Object.values(row?.chunk_counts_by_strategy || {})
    .map(Number)
    .filter(value => Number.isFinite(value) && value >= 0);
  if (!counts.length) return 'chunk counts by strategy unavailable';
  const low = Math.min(...counts);
  const high = Math.max(...counts);
  const range = low === high ? fmtInt(high) : `${fmtInt(low)}–${fmtInt(high)}`;
  return `${fmtInt(counts.length)} chunk strateg${counts.length === 1 ? 'y' : 'ies'} · ${range} chunks/strategy`;
}

function sourceOption(row, type) {
  const count = type === 'dataset'
    ? `${fmtInt(row.document_count || 0)} documents · ${datasetChunkSummary(row)}`
    : (row.id === 'groundtruth:none' ? 'evidence-only' : `${fmtInt(row.row_count || 0)} queries`);
  const runnable = type === 'dataset' ? row.ready : row.valid;
  return `<option value="${esc(row.id)}"${runnable ? '' : ' disabled'}>${esc(row.label)} · ${esc(count)}${runnable ? '' : ' · unavailable'}</option>`;
}

function fillSourceSelect(id, rows, type, preferredId='') {
  const el = $(id);
  if (!el) return;
  const current = el.value;
  el.innerHTML = (rows || []).map(row => sourceOption(row, type)).join('');
  const validIds = new Set((rows || []).filter(row => type === 'dataset' ? row.ready : row.valid).map(row => row.id));
  if (validIds.has(current)) el.value = current;
  else if (validIds.has(preferredId)) el.value = preferredId;
  else el.value = [...validIds][0] || '';
}

function syncDatasetChunkers() {
  const datasetId = $('runDataset')?.value || 'dataset:wns-default';
  const dataset = (state.sourceCatalog?.datasets || []).find(row => row.id === datasetId);
  const sheets = activeDatasetSheets();
  fillRunMultiSelect('runSheet', sheets, 'All dataset chunkers');
  if ($('runDatasetHint') && dataset) {
    $('runDatasetHint').textContent = `${fmtInt(dataset.document_count || 0)} documents · ${datasetChunkSummary(dataset)}`;
  }
  updateSelectedMatrixCount();
}

function syncRunMode() {
  const evidenceOnly = $('runGroundtruth')?.value === 'groundtruth:none';
  const datasetId = $('runDataset')?.value || 'dataset:wns-default';
  const dataset = (state.sourceCatalog?.datasets || []).find(row => row.id === datasetId);
  const defaultDataset = !dataset || dataset.kind === 'default';
  if ($('runQueryField')) $('runQueryField').hidden = !evidenceOnly;
  if ($('runModeHint')) {
    $('runModeHint').textContent = !defaultDataset
      ? evidenceOnly
        ? 'Project evidence-only mode — the isolated project matrix returns retrieval evidence without recall, MRR, nDCG, accuracy, or winner scores. Optional reranking does not create quality metrics.'
        : 'Project evaluated mode — the isolated project matrix uses the selected immutable ground truth. Legacy stage controls remain disabled because they target default repository artifacts.'
      : evidenceOnly
        ? 'Evidence-only mode — retrieval evidence will be returned without recall, MRR, nDCG, accuracy, or winner scores. Fresh-run archival and scored controls are disabled.'
        : 'Evaluated mode — retrieval metrics will use the selected ground truth.';
  }
  if ($('runCompletePipelineBtn')) {
    $('runCompletePipelineBtn').textContent = evidenceOnly ? 'Run evidence retrieval' : 'Run pipeline and evaluate';
    $('runCompletePipelineBtn').disabled = false;
  }

  const fresh = $('runFresh');
  if (fresh) {
    fresh.disabled = evidenceOnly;
    if (evidenceOnly) fresh.checked = false;
  }
  const plannedActionIds = ['runHallucinationBtn', 'runHallucinationLlmBtn'];
  const scoredActionIds = ['runFullGtBtn', 'runRerankerBtn', 'runRerankedEvalBtn', 'runEvalBtn', ...plannedActionIds];
  const canonicalOnlyIds = ['runParseMineruBtn', 'runIngestBtn', 'runSmokeBtn', ...scoredActionIds];
  canonicalOnlyIds.forEach(id => {
    const button = $(id);
    if (button) button.disabled = plannedActionIds.includes(id) || !defaultDataset || (evidenceOnly && scoredActionIds.includes(id));
  });
  if ($('advancedRunHint')) $('advancedRunHint').textContent = !defaultDataset
    ? 'Advanced stage controls are disabled for uploaded datasets; use the source-aware complete pipeline above.'
    : evidenceOnly
      ? 'Scored evaluation and reranker actions are disabled in evidence-only mode.'
      : 'Use these controls only for targeted troubleshooting of the default WNS pipeline.';

  const nvidiaEvidenceOnly = $('nvidiaGroundtruth')?.value === 'groundtruth:none';
  if ($('runNvidiaBenchmarkBtn')) $('runNvidiaBenchmarkBtn').disabled = nvidiaEvidenceOnly;
  if ($('nvidiaGroundtruthHint')) $('nvidiaGroundtruthHint').textContent = nvidiaEvidenceOnly
    ? 'None selected — use Test one query; scored benchmark is disabled.'
    : 'Validated ground truth selected for NVIDIA benchmark.';
}

function syncNvidiaBenchmarkMode() {
  const baseline = ($('nvidiaBenchmarkMode')?.value || 'baseline') === 'baseline';
  if ($('nvidiaDisableReranker')) $('nvidiaDisableReranker').checked = baseline;
}

function setSourceMirrorValue(id, value) {
  const select = $(id);
  if (!select) return false;
  const exists = Array.from(select.options || []).some(option => option.value === value && !option.disabled);
  if (exists) select.value = value;
  return exists;
}

function clearGlobalResultViews(message) {
  recommendationState.active = null;
  clearRecommendationView(message);
  setText('metricsSourceContext', message);
  setText('qualityInsight', message);
  if ($('qualityTable')) {
    const row = document.createElement('tr');
    const cell = document.createElement('td');
    cell.className = 'empty-cell';
    cell.textContent = 'No completed result for the global data context';
    row.appendChild(cell);
    $('qualityTable').replaceChildren(row);
  }
}

function renderRunSourceContext(datasetLabel, groundtruthLabel) {
  setText(
    'runSourceContext',
    `Running on: ${datasetLabel} · Ground truth: ${groundtruthLabel} · Change using the selectors at the top.`,
  );
}

async function applyGlobalSourceContext({syncResults = true} = {}) {
  const datasetId = $('globalDataset')?.value || 'dataset:wns-default';
  const groundtruthId = $('globalGroundtruth')?.value || 'groundtruth:none';
  ['runDataset', 'nvidiaDataset'].forEach(id => setSourceMirrorValue(id, datasetId));
  ['runGroundtruth', 'nvidiaGroundtruth'].forEach(id => setSourceMirrorValue(id, groundtruthId));
  syncDatasetChunkers();
  syncRunMode();

  const datasetLabel = $('globalDataset')?.selectedOptions?.[0]?.textContent || datasetId;
  const groundtruthLabel = $('globalGroundtruth')?.selectedOptions?.[0]?.textContent || groundtruthId;
  setText('globalSourceContext', `${datasetLabel} · ${groundtruthLabel}`);
  renderRunSourceContext(datasetLabel, groundtruthLabel);
  renderDatasetStatusSummary();
  if (!syncResults) return;

  if (datasetId === 'dataset:wns-default') {
    if (groundtruthId !== globalSourceContextState.officialGroundtruthId) {
      clearGlobalResultViews('No completed official result for the selected global ground truth');
      return;
    }
    setSourceMirrorValue('recommendationDataset', 'official');
    setSourceMirrorValue('recommendationGroundtruth', 'groundtruth:official');
    await selectRecommendationDataset('official');
    return;
  }

  await selectRecommendationDataset(datasetId);
  if (recommendationState.sourceType !== 'uploaded_project') return;
  if (!setSourceMirrorValue('recommendationGroundtruth', groundtruthId)) {
    clearGlobalResultViews('No completed run for the selected dataset and ground truth');
    return;
  }
  await selectRecommendationGroundtruth(groundtruthId);
}

function renderSourceSelectors(catalog) {
  const datasets = catalog?.datasets || [];
  const groundtruth = catalog?.groundtruth || [];
  const preferredGroundtruth = groundtruth.find(row => row.valid && /groundtruth_500/i.test(row.id))?.id
    || groundtruth.find(row => row.valid && row.id !== 'groundtruth:none')?.id
    || 'groundtruth:none';
  globalSourceContextState.officialGroundtruthId = preferredGroundtruth;
  fillSourceSelect('globalDataset', datasets, 'dataset', 'dataset:wns-default');
  fillSourceSelect('globalGroundtruth', groundtruth, 'groundtruth', preferredGroundtruth);
  fillSourceSelect('runDataset', datasets, 'dataset', 'dataset:wns-default');
  fillSourceSelect('nvidiaDataset', datasets, 'dataset', 'dataset:wns-default');
  fillSourceSelect('runGroundtruth', groundtruth, 'groundtruth', preferredGroundtruth);
  fillSourceSelect('nvidiaGroundtruth', groundtruth, 'groundtruth', preferredGroundtruth);
  applyGlobalSourceContext({syncResults: false}).catch(console.error);
}

async function refresh() {
  $('statusPill').textContent = 'Refreshing';
  try {
    const data = await api('/api/results?retrieval_limit=0&reranker_limit=0&detail_evidence_limit=360&detail_evidence_per_combo=1');
    state = {
      operational: data.operational || {},
      files: data.files || [],
      options: data.options || {},
      sourceCatalog: data.source_catalog || {datasets: [], groundtruth: []},
    };
    renderSourceSelectors(state.sourceCatalog);
    renderOperational();
    $('statusPill').textContent = 'Live';
  } catch (e) {
    $('statusPill').textContent = 'Error';
    console.error('Dashboard render failed', e);
    document.body.insertAdjacentHTML('afterbegin', `<pre class="fatal">Dashboard render failed: ${esc(e.stack || e.message || String(e))}</pre>`);
    throw e;
  }
}

async function loadOptions() {
  benchmarkOptions = await api('/api/options');
  fillRunMultiSelect('runSheet', benchmarkOptions.chunkers || [], 'All chunkers');
  fillRunMultiSelect('runEmbedding', benchmarkOptions.embeddings || [], 'All embeddings');
  fillRunMultiSelect('runStore', benchmarkOptions.vector_stores || [], 'All DBs');
  const retrieval = $('runRetrievalMethod');
  if (retrieval) {
    const methods = [...(benchmarkOptions.retrieval_methods || []), ...(candidateLane().retrieval_methods || [])];
    retrieval.innerHTML = methods.map(value => `<option value="${esc(value)}">${esc(value)}</option>`).join('');
    retrieval.value = methods.includes(retrieval.value) ? retrieval.value : 'Cosine Similarity';
  }
  syncRerankerOptionsForMethod();
  updateSelectedMatrixCount();
}

function syncRerankerOptionsForMethod() {
  const candidate = isCandidateRetrievalMethod();
  if (candidate) {
    const lane = candidateLane();
    fillRunMultiSelect('runEmbedding', lane.embeddings || [], 'All candidate embeddings');
    fillRunMultiSelect('runStore', lane.vector_stores || ['FAISS'], 'Candidate vector store');
  } else {
    fillRunMultiSelect('runEmbedding', benchmarkOptions.embeddings || [], 'All embeddings');
    fillRunMultiSelect('runStore', benchmarkOptions.vector_stores || [], 'All DBs');
  }
  const values = candidate
    ? (candidateLane().rerankers || [])
    : [...new Set((benchmarkOptions.rerankers || []).map(canonicalRerankerName)), 'none'];
  fillRunMultiSelect('runRerankerMain', [...new Set(values)], 'All rerankers');
}

function isProjectDatasetSelected() {
  const datasetId = $('runDataset')?.value || 'dataset:wns-default';
  return datasetId.startsWith('project:');
}

function typedRunQueries() {
  return ($('runQueries')?.value || '')
    .split('\n')
    .map(value => value.trim())
    .filter(Boolean);
}

function projectSelectedValues(id, available) {
  return expandedSelection(id, available || []);
}

function projectSelectedRerankers() {
  const available = [...new Set((benchmarkOptions.rerankers || []).map(canonicalRerankerName))];
  return projectSelectedValues('runRerankerMain', available)
    .filter(value => value !== 'none');
}

function projectMatrixPayload(confirmationToken = null) {
  const datasetId = $('runDataset')?.value || 'dataset:wns-default';
  const groundtruthId = $('runGroundtruth')?.value || 'groundtruth:none';
  const evidenceOnly = groundtruthId === 'groundtruth:none';
  return {
    schema_version: 1,
    dataset_id: datasetId,
    groundtruth_id: groundtruthId,
    typed_queries: evidenceOnly ? typedRunQueries() : [],
    top_k: Number($('runRetrievalTopK')?.value || 10),
    selections: {
      chunkers: projectSelectedValues('runSheet', benchmarkOptions.chunkers || []),
      embeddings: projectSelectedValues('runEmbedding', benchmarkOptions.embeddings || []),
      vector_stores: projectSelectedValues('runStore', benchmarkOptions.vector_stores || []),
      rerankers: projectSelectedRerankers(),
    },
    large_matrix_confirmation: confirmationToken,
  };
}

async function postProjectMatrix(path, payload) {
  return api(path, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload),
  });
}

function selectedParams() {
  const p = new URLSearchParams();
  const candidate = isCandidateRetrievalMethod();
  const selectedChunkers = candidate ? selectedValues('runSheet') : expandedSelection('runSheet', activeDatasetSheets());
  selectedChunkers.forEach(v => p.append('sheet', v));
  if (candidate) {
    selectedValues('runEmbedding').forEach(v => p.append('embedding', v));
    p.append('store', 'FAISS');
  } else {
    selectedValues('runEmbedding').forEach(v => p.append('embedding', v));
    selectedValues('runStore').forEach(v => p.append('store', v));
  }
  selectedValues('runRerankerMain').forEach(v => p.append('reranker', v));
  p.set('retrieval_method', selectedRetrievalMethod());
  if (candidate) p.set('candidate_lane', '1');
  p.set('query_limit', $('runQueryLimit')?.value || '0');
  p.set('retrieval_top_k', $('runRetrievalTopK')?.value || '10');
  p.set('reranked_output_k', $('runRerankedOutputK')?.value || '5');
  p.set('dataset_id', $('runDataset')?.value || 'dataset:wns-default');
  p.set('groundtruth_id', $('runGroundtruth')?.value || 'groundtruth:none');
  if (($('runGroundtruth')?.value || 'groundtruth:none') === 'groundtruth:none') {
    ($('runQueries')?.value || '').split('\n').map(value => value.trim()).filter(Boolean).forEach(query => p.append('query', query));
  }
  if ($('runFresh')?.checked) p.set('fresh_run', '1');
  if ($('runAllowPartialExtraction')?.checked) p.set('allow_partial_extraction', '1');
  p.set('max_runs', String(Math.max(updateSelectedMatrixCount(), 1)));
  return p;
}

async function runNvidiaAction(kind) {
  const endpoints = {
    health: '/api/run/nvidia-health',
    smoke: '/api/run/nvidia-smoke',
    ingest: '/api/run/nvidia-ingest',
    benchmark: '/api/run/nvidia-benchmark',
  };
  const p = new URLSearchParams();
  const collection = ($('nvidiaCollection')?.value || 'multimodal_data').trim();
  if (collection) p.set('collection', collection);
  p.set('limit', $('nvidiaLimit')?.value || '25');
  p.set('batch_size', $('nvidiaBatchSize')?.value || '2');
  p.set('top_k', $('nvidiaTopK')?.value || '10');
  p.set('reranker_top_k', $('nvidiaRerankerTopK')?.value || '5');
  p.set('dataset_id', $('nvidiaDataset')?.value || 'dataset:wns-default');
  p.set('groundtruth_id', $('nvidiaGroundtruth')?.value || 'groundtruth:none');
  if ($('nvidiaCreateCollection')?.checked) p.set('create_collection', '1');
  if ($('nvidiaPoll')?.checked) p.set('poll', '1');
  if (kind === 'smoke') {
    p.set('mode', $('nvidiaMode')?.value || 'search');
    const query = ($('nvidiaQuery')?.value || '').trim();
    if (query) p.append('query', query);
    if ($('nvidiaAgentic')?.checked) p.set('agentic', '1');
  }
  const disableReranker = kind === 'benchmark'
    ? ($('nvidiaBenchmarkMode')?.value || 'baseline') === 'baseline'
    : $('nvidiaDisableReranker')?.checked;
  if (disableReranker) p.set('disable_reranker', '1');
  $('nvidiaStatus').textContent = 'Running';
  $('nvidiaOutput').textContent = `POST ${endpoints[kind]}?${p.toString()}\n`;
  const res = await fetch(`${endpoints[kind]}?${p.toString()}`, {method:'POST'});
  const payload = await res.json();
  $('nvidiaOutput').textContent += JSON.stringify(payload, null, 2).slice(0, 8000);
  $('nvidiaStatus').textContent = payload.exit_code === 0 ? 'Done' : 'Check output';
  await refresh();
}

function renderPreflight(payload) {
  const ok = !!payload.ok;
  const missing = payload.missing || [];
  const warnings = payload.warnings || [];
  const checks = payload.service_checks || [];
  const diagnostics = $('runTechnicalDiagnostics');
  const summary = $('runReadinessSummary');
  const evidenceOnly = payload.groundtruth_id === 'groundtruth:none';
  $('runStatus').textContent = ok ? 'Ready' : 'Needs inputs';
  if (summary) {
    summary.textContent = ok
      ? `Ready to run the selected pipeline.${warnings.length ? ` ${warnings.length} warning(s) are available in technical details.` : ''}`
      : `Resolve before running: ${missing.slice(0, 3).join('; ') || 'review technical readiness details'}`;
    summary.classList.toggle('warn', !ok);
  }
  if (diagnostics) diagnostics.open = false;
  $('runResultCards').innerHTML = `<article class="run-result-card ${ok ? 'ok' : 'warn'}"><span>Requirements</span><strong>${ok ? 'Ready' : 'Blocked'}</strong><small>${missing.length ? `${missing.length} item(s) needed` : 'All required inputs found'}</small></article>
    <article class="run-result-card"><span>Pipeline size</span><strong>${esc(payload.combo_count || 0)}</strong><small>chunker · embedding · DB combinations</small></article>
    <article class="run-result-card"><span>Mode</span><strong>${evidenceOnly ? 'Evidence-only' : 'Evaluated'}</strong><small>${payload.query_limit ? `${payload.query_limit} query limit` : 'all queries'}</small></article>`;
  const rows = [
    ...missing.map(x => ({type:'Needed', item:x})),
    ...warnings.map(x => ({type:'Warning', item:x})),
    ...checks.map(x => ({type:x.ok ? 'Service ok' : 'Service check', item:`${x.name}: ${x.note}`})),
  ];
  table($('runResultTable'), rows, [
    {key:'type', label:'Type'},
    {key:'item', label:'Details', render:r=>esc(r.item).slice(0, 220)},
  ]);
  $('runOutput').textContent = JSON.stringify(payload, null, 2);
}

async function runProjectMatrixPreflight() {
  const payload = await postProjectMatrix(
    '/api/run/preflight-project-matrix',
    projectMatrixPayload(),
  );
  renderPreflight({
    ...payload,
    combo_count: payload.combination_count || 0,
    warnings: payload.warning ? [payload.warning] : [],
  });
  return payload;
}

async function runPreflight() {
  if (isProjectDatasetSelected()) return runProjectMatrixPreflight();
  const p = selectedParams();
  const endpoint = isCandidateRetrievalMethod()
    ? '/api/run/preflight-candidate'
    : '/api/run/preflight-complete-pipeline';
  $('runStatus').textContent = 'Checking';
  const payload = await api(`${endpoint}?${p.toString()}`, {method:'POST'});
  renderPreflight(payload);
  return payload;
}

function parseLiveOutputRows(output) {
  const lines = (output || '').split('\n').filter(Boolean);
  const stageRows = lines.filter(l => l.includes('START ') || l.includes('DONE ') || l.includes('FAILED') || l.includes('groundtruth_rows=') || l.includes('OK ')).slice(-80);
  return stageRows.map((line, i) => ({i:i+1, line}));
}

async function pollRunJob(jobId, {candidate = false} = {}) {
  const payload = await api(`/api/run/status?job_id=${encodeURIComponent(jobId)}`);
  const output = payload.output || '';
  const running = !!payload.running;
  $('runStatus').textContent = running ? 'Running' : (payload.exit_code === 0 ? 'Done' : 'Check output');
  $('runOutput').textContent = output;
  $('runResultCards').innerHTML = `<article class="run-result-card ${payload.exit_code === 0 ? 'ok' : running ? '' : 'warn'}"><span>Status</span><strong>${running ? 'Running' : payload.exit_code === 0 ? 'Passed' : 'Needs attention'}</strong><small>job ${esc(jobId)}</small></article>
    <article class="run-result-card"><span>Live log</span><strong>${output.split('\n').filter(Boolean).length}</strong><small>${esc(payload.log_path || '')}</small></article>
    <article class="run-result-card"><span>Result view</span><strong>${candidate ? 'Candidate artifacts' : 'Quality + evidence'}</strong><small>${candidate ? 'Isolated from official dashboard metrics' : 'Tables refresh when stages finish'}</small></article>`;
  table($('runResultTable'), parseLiveOutputRows(output), [
    {key:'i', label:'#'},
    {key:'line', label:'Live stage/output', render:r=>esc(r.line).slice(0, 260)},
  ]);
  if (running) {
    setTimeout(() => pollRunJob(jobId, {candidate}).catch(e => { $('runStatus').textContent='Error'; $('runOutput').textContent=String(e); }), 1500);
  } else {
    await refresh();
  }
}

async function runCandidateBenchmark() {
  const p = selectedParams();
  $('runStatus').textContent = 'Starting candidate lane';
  $('runOutput').textContent = 'Candidate lane: running selected retrieval/reranker combination';
  const payload = await api(`/api/run/selected?${p.toString()}`, {method:'POST'});
  if (!payload.job_id) throw new Error('Candidate benchmark launch did not return a job ID.');
  $('runOutput').textContent = `Candidate lane job ${payload.job_id}\n${payload.output || ''}`;
  await pollRunJob(payload.job_id, {candidate: true});
}

async function runCompletePipeline() {
  const candidate = isCandidateRetrievalMethod();
  const preflight = await runPreflight();
  if (!preflight.ok) return;
  const comboCount = preflight.combination_count || preflight.combo_count || 0;
  if (comboCount > 12) {
    const ok = window.confirm(`This will run ${comboCount} pipeline combinations through ingestion, retrieval, reranking and evaluation. Continue?`);
    if (!ok) {
      $('runStatus').textContent = 'Cancelled';
      return;
    }
  }
  if (candidate) return runCandidateBenchmark();
  if (isProjectDatasetSelected()) {
    $('runStatus').textContent = 'Starting';
    const payload = projectMatrixPayload(preflight.confirmation_token || null);
    const launched = await postProjectMatrix('/api/run/project-matrix', payload);
    const jobId = launched.job?.job_id;
    if (!jobId) throw new Error('Project matrix launch did not return a job ID.');
    $('runOutput').textContent = `Started isolated project run ${launched.run_id}\njob ${jobId}`;
    await pollRunJob(jobId);
    return;
  }
  const p = selectedParams();
  $('runStatus').textContent = 'Starting';
  const payload = await api(`/api/run/complete-pipeline?${p.toString()}`, {method:'POST'});
  $('runOutput').textContent = `Started job ${payload.job_id}\n${payload.output || ''}`;
  await pollRunJob(payload.job_id);
}

function candidateOperatorHeaders() {
  const token = $('portfolioOperatorToken')?.value || '';
  if (!token.trim()) throw new Error('Operator token required for candidate mutations.');
  return {
    'Content-Type': 'application/json',
    Authorization: `Bearer ${token}`,
  };
}

async function candidateMutation(path, payload) {
  const response = await fetch(path, {
    method: 'POST',
    cache: 'no-store',
    headers: candidateOperatorHeaders(),
    body: JSON.stringify(payload),
  });
  const result = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(result?.error?.message || result?.message || 'Candidate action was blocked.');
  }
  return result;
}

function candidateStateLabel(value) {
  const normalized = String(value || 'unknown').replaceAll('_', ' ');
  return normalized.charAt(0).toUpperCase() + normalized.slice(1);
}

function selectedPortfolioBatch() {
  const portfolio = candidateOperationsState.portfolio;
  if (!portfolio) return null;
  return (portfolio.batches || []).find(batch => batch.batch_id === portfolio.selected_batch_id) || null;
}

function renderPortfolioBatches(portfolio) {
  const target = $('portfolioBatchList');
  if (!target) return;
  const batches = portfolio?.batches || [];
  if (!batches.length) {
    target.innerHTML = '<p class="candidate-empty">No immutable portfolio batches are available.</p>';
    return;
  }
  const controlsEnabled = candidateOperationsState.adapters?.control_enabled === true;
  target.innerHTML = batches.map((batch, index) => {
    const selected = batch.selected === true;
    const terminal = batch.state === 'completed';
    const services = (batch.required_service_ids || []).join(', ') || 'No managed services';
    return `<article class="portfolio-batch-row state-${esc(batch.state)} ${selected ? 'selected' : ''}" data-batch-id="${esc(batch.batch_id)}">
      <div class="batch-sequence"><span>${String(index + 1).padStart(2, '0')}</span><small>of ${batches.length}</small></div>
      <div class="batch-identity"><strong>${esc(batch.embedding)}</strong><span>Reranker group ${esc(batch.reranker_group)} · ${fmtInt(batch.combination_count)} combinations</span><small title="${esc(services)}">Requires: ${esc(services)}</small></div>
      <span class="batch-state">${selected && batch.state === 'not_run' ? 'Selected next' : esc(candidateStateLabel(batch.state))}</span>
      <button type="button" data-portfolio-run="${esc(batch.batch_id)}" ${!controlsEnabled || terminal || candidateOperationsState.busy ? 'disabled' : ''}>${batch.state === 'failed' ? 'Retry batch' : 'Run batch'}</button>
    </article>`;
  }).join('');
}

function adapterActionForSlot(slot) {
  if (slot.ok && slot.stoppable) return {action: 'stop', label: 'Stop'};
  if (slot.status === 'unhealthy' && slot.startable) return {action: 'retry', label: 'Retry'};
  if (!slot.ok && slot.startable) return {action: 'start', label: 'Start'};
  return null;
}

function renderAdapterSlots(payload) {
  const target = $('adapterSlotList');
  if (!target) return;
  const slots = payload?.slots || [];
  setText('adapterSlotCount', fmtInt(payload?.slot_count));
  setText('adapterControlMode', payload?.control_enabled ? 'Control enabled' : 'Read only');
  if (!slots.length) {
    target.innerHTML = '<p class="candidate-empty">Adapter status is unavailable.</p>';
    return;
  }
  target.innerHTML = slots.map(slot => {
    const action = adapterActionForSlot(slot);
    const disabled = !payload.control_enabled || candidateOperationsState.busy;
    return `<article class="adapter-slot-row ${slot.ok ? 'healthy' : ''}" data-service-id="${esc(slot.service_id)}">
      <div class="adapter-identity"><strong>${esc(slot.label)}</strong><span>${esc(slot.role)} · ${esc(slot.manage_mode)}</span></div>
      <code>${esc(slot.port)}</code>
      <div class="adapter-status"><span class="adapter-dot" aria-hidden="true"></span><strong>${esc(candidateStateLabel(slot.status))}</strong><small>${esc(slot.message || '')}</small></div>
      ${action ? `<button type="button" data-adapter-action="${esc(action.action)}" data-service-id="${esc(slot.service_id)}" ${disabled ? 'disabled' : ''}>${esc(action.label)}</button>` : '<span class="adapter-managed">Server managed</span>'}
    </article>`;
  }).join('');
}

function renderCandidateOperations() {
  const portfolio = candidateOperationsState.portfolio;
  const adapters = candidateOperationsState.adapters;
  if (portfolio) {
    setText('portfolioConfiguredCount', fmtInt(portfolio.configured_combination_count));
    setText('portfolioExcludedCount', fmtInt(portfolio.excluded_combination_count));
    setText('portfolioBatchCount', fmtInt(portfolio.batch_count));
    setText('portfolioCompletedCount', fmtInt(portfolio.combination_state_counts?.completed));
    setText('portfolioPromotionStatus', `Promotion: ${String(portfolio.promotion_status || 'not_accepted').replaceAll('_', ' ')}`);
    setText('portfolioIdentity', `${portfolio.portfolio_id} · max ${portfolio.max_combinations_per_batch} per batch`);
    renderPortfolioBatches(portfolio);
  }
  if (adapters) renderAdapterSlots(adapters);
  const selected = selectedPortfolioBatch();
  const mutable = adapters?.control_enabled === true && selected && !candidateOperationsState.busy;
  if ($('portfolioStartRequired')) $('portfolioStartRequired').disabled = !mutable;
  if ($('portfolioRunNext')) $('portfolioRunNext').disabled = !mutable;
}

async function loadCandidateOperations() {
  setText('candidateOperationsStatus', 'Refreshing candidate state');
  const [adapters, portfolio] = await Promise.all([
    api('/api/adapters'),
    api('/api/portfolio'),
  ]);
  candidateOperationsState.adapters = adapters;
  candidateOperationsState.portfolio = portfolio;
  renderCandidateOperations();
  setText('candidateOperationsStatus', `Live · ${fmtInt(adapters.slot_count)} slots · ${fmtInt(portfolio.batch_count)} batches`);
}

async function loadPortfolioCombinationDetails() {
  if (!candidateOperationsState.details) {
    setText('portfolioCombinationHint', 'Loading 4,320 portfolio rows…');
    candidateOperationsState.details = await api('/api/portfolio/batches');
  }
  renderPortfolioCombinationRows();
}

function renderPortfolioCombinationRows() {
  const target = $('portfolioCombinationRows');
  const details = candidateOperationsState.details;
  if (!target || !details) return;
  const query = String($('portfolioCombinationFilter')?.value || '').trim().toLowerCase();
  const rows = [
    ...(details.configured_combinations || []),
    ...(details.excluded_combinations || []),
  ];
  const visible = query ? rows.filter(row => JSON.stringify(row).toLowerCase().includes(query)) : rows;
  setText('portfolioCombinationHint', `${fmtInt(visible.length)} of ${fmtInt(rows.length)} rows`);
  target.innerHTML = visible.map(row => `<article class="combination-row state-${esc(row.execution_state || row.state)}">
    <code>${esc(row.combination_id)}</code>
    <strong>${esc(row.chunker)} · ${esc(row.embedding)}</strong>
    <span>${esc(row.vector_store)} / ${esc(row.index_type)} · ${esc(row.retrieval_method)} · ${esc(row.reranker)}</span>
    <small>${esc(candidateStateLabel(row.execution_state || row.state))}${row.reason_code ? ` · ${esc(row.reason_code)}` : ''}</small>
  </article>`).join('') || '<p class="candidate-empty">No combinations match this filter.</p>';
}

async function runCandidateOperation(action) {
  if (candidateOperationsState.busy) return;
  candidateOperationsState.busy = true;
  renderCandidateOperations();
  setText('candidateActionMessage', 'Submitting candidate operation…');
  try {
    const result = await action();
    setText('candidateActionMessage', result?.message || `Candidate operation accepted${result?.job_id ? ` · job ${result.job_id}` : ''}.`);
    await loadCandidateOperations();
  } catch (error) {
    setText('candidateActionMessage', error?.message || String(error));
  } finally {
    candidateOperationsState.busy = false;
    renderCandidateOperations();
  }
}

async function startRequiredAdapters() {
  const batchId = selectedPortfolioBatch()?.batch_id;
  if (!batchId) throw new Error('No incomplete batch is selected.');
  return candidateMutation('/api/adapters/start-required', {batch_id: batchId});
}

async function runPortfolioBatch(batchId) {
  return candidateMutation('/api/portfolio/run-batch', {batch_id: batchId});
}

async function runNextPortfolioBatch() {
  return candidateMutation('/api/portfolio/run-next', {});
}

async function runAction(kind) {
  const endpoints = {
    ingest: '/api/run/ingest-selected',
    parseMineru: '/api/run/parse-pdfs-mineru',
    smoke: '/api/run/retrieval-smoke',
    eval: '/api/run/evaluate-groundtruth',
    full: '/api/run/full-gt-retrieval',
    rerank: '/api/run/reranker-smoke',
    rerankEval: '/api/run/evaluate-reranked-groundtruth',
    hallucination: '/api/run/evaluate-hallucination',
    hallucinationLlm: '/api/run/evaluate-hallucination',
  };
  const p = selectedParams();
  if (kind === 'smoke') {
    p.delete('limit');
    p.set('max_runs', '1');
    ($('runQueries')?.value || '').split('\n').map(s=>s.trim()).filter(Boolean).forEach(q => p.append('query', q));
  }
  if (kind === 'full') {
    p.set('top_k', $('runRetrievalTopK')?.value || '10');
    p.set('max_runs', String(benchmarkOptions.matrix_count || 180));
  }
  if (kind === 'rerank') {
    p.set('top_k', $('runRerankedOutputK')?.value || '5');
    p.set('limit', '0');
    const mainChoices = selectedValues('runRerankerMain');
    let selected = [];
    if (mainChoices.includes('all')) {
      selected = (benchmarkOptions.rerankers || ['bge-reranker-base', 'qwen3_4b_rerank', 'Amazon Rerank v1']).map(canonicalRerankerName);
    } else {
      selected = mainChoices.filter(v => v && v !== 'none');
    }
    selected.forEach(r => p.append('reranker', r));
  }
  if (kind === 'hallucination' || kind === 'hallucinationLlm') {
    p.set('limit', '0');
    if (kind === 'hallucinationLlm') p.set('use_llm', '1');
  }
  $('runStatus').textContent = 'Running';
  $('runOutput').textContent = `POST ${endpoints[kind]}?${p.toString()}\n`;
  const res = await fetch(`${endpoints[kind]}?${p.toString()}`, {method:'POST'});
  const payload = await res.json();
  renderRunResult(kind, payload);
  $('runStatus').textContent = payload.exit_code === 0 ? 'Done' : 'Check output';
  await refresh();
}

['retrievalChunkerFilter','retrievalDbFilter','retrievalEmbeddingFilter','retrievalRerankerFilter'].forEach(id => $(id)?.addEventListener('input', () => renderRetrieval(state.operational?.retrieval_smokes || [], operationalRerankRows())));
['qualityComboFilter','qualityChunkerFilter','qualityEmbeddingFilter','qualityDbFilter','qualityRerankerFilter'].forEach(id => $(id)?.addEventListener('input', () => {
  const payload = recommendationState.active || officialRecommendationPayload('official');
  renderMetricsSource(payload);
}));
['runSheet','runEmbedding','runStore','runRerankerMain'].forEach(id => $(id)?.addEventListener('input', updateSelectedMatrixCount));
$('runRetrievalMethod')?.addEventListener('input', () => {
  syncRerankerOptionsForMethod();
  updateSelectedMatrixCount();
});
$('globalDataset')?.addEventListener('change', () => applyGlobalSourceContext().catch(console.error));
$('globalGroundtruth')?.addEventListener('change', () => applyGlobalSourceContext().catch(console.error));
$('runDataset')?.addEventListener('input', () => {
  syncDatasetChunkers();
  syncRunMode();
});
$('runGroundtruth')?.addEventListener('input', syncRunMode);
$('nvidiaGroundtruth')?.addEventListener('input', syncRunMode);
$('uploadType')?.addEventListener('input', () => {
  const groundtruth = $('uploadType').value === 'groundtruth';
  if ($('datasetFile')) $('datasetFile').accept = groundtruth ? '.csv,.xlsx' : '.pdf,.zip,.csv,.txt,.md';
  if ($('uploadSubmitBtn')) $('uploadSubmitBtn').textContent = groundtruth ? 'Register reusable ground truth' : 'Upload into isolated project';
});
$('nvidiaBenchmarkMode')?.addEventListener('input', () => { syncNvidiaBenchmarkMode(); renderNvidiaRag(state.operational?.nvidia_rag || {}); });
$('refreshBtn').addEventListener('click', () => refresh().catch(e => { $('statusPill').textContent = 'Error'; console.error(e); }));
$('candidateOperationsRefresh')?.addEventListener('click', () => loadCandidateOperations().catch(error => {
  setText('candidateOperationsStatus', 'Candidate state unavailable');
  setText('candidateActionMessage', error?.message || String(error));
}));
$('portfolioStartRequired')?.addEventListener('click', () => runCandidateOperation(startRequiredAdapters));
$('portfolioRunNext')?.addEventListener('click', () => runCandidateOperation(runNextPortfolioBatch));
$('portfolioCombinationDetails')?.addEventListener('toggle', event => {
  if (event.target.open) loadPortfolioCombinationDetails().catch(error => setText('portfolioCombinationHint', error?.message || String(error)));
});
$('portfolioCombinationFilter')?.addEventListener('input', renderPortfolioCombinationRows);
$('runPreflightBtn')?.addEventListener('click', () => runPreflight().catch(e => { $('runStatus').textContent='Error'; $('runOutput').textContent=String(e); }));
$('runCompletePipelineBtn')?.addEventListener('click', () => runCompletePipeline().catch(e => { $('runStatus').textContent='Error'; $('runOutput').textContent=String(e); }));
$('runQueryUploadBtn')?.addEventListener('click', () => uploadRunQueries().catch(e => { if ($('runQueryUploadStatus')) $('runQueryUploadStatus').textContent = String(e); }));
$('runParseMineruBtn')?.addEventListener('click', () => runAction('parseMineru').catch(e => { $('runStatus').textContent='Error'; $('runOutput').textContent=String(e); }));
$('runIngestBtn')?.addEventListener('click', () => runAction('ingest').catch(e => { $('runStatus').textContent='Error'; $('runOutput').textContent=String(e); }));
$('runSmokeBtn')?.addEventListener('click', () => runAction('smoke').catch(e => { $('runStatus').textContent='Error'; $('runOutput').textContent=String(e); }));
$('runFullGtBtn')?.addEventListener('click', () => runAction('full').catch(e => { $('runStatus').textContent='Error'; $('runOutput').textContent=String(e); }));
$('runRerankerBtn')?.addEventListener('click', () => runAction('rerank').catch(e => { $('runStatus').textContent='Error'; $('runOutput').textContent=String(e); }));
$('runRerankedEvalBtn')?.addEventListener('click', () => runAction('rerankEval').catch(e => { $('runStatus').textContent='Error'; $('runOutput').textContent=String(e); }));
$('runEvalBtn')?.addEventListener('click', () => runAction('eval').catch(e => { $('runStatus').textContent='Error'; $('runOutput').textContent=String(e); }));
$('runHallucinationBtn')?.addEventListener('click', () => runAction('hallucination').catch(e => { $('runStatus').textContent='Error'; $('runOutput').textContent=String(e); }));
$('runHallucinationLlmBtn')?.addEventListener('click', () => runAction('hallucinationLlm').catch(e => { $('runStatus').textContent='Error'; $('runOutput').textContent=String(e); }));
$('runNvidiaHealthBtn')?.addEventListener('click', () => runNvidiaAction('health').catch(e => { $('nvidiaStatus').textContent='Error'; $('nvidiaOutput').textContent=String(e); }));
$('runNvidiaSmokeBtn')?.addEventListener('click', () => runNvidiaAction('smoke').catch(e => { $('nvidiaStatus').textContent='Error'; $('nvidiaOutput').textContent=String(e); }));
$('runNvidiaIngestBtn')?.addEventListener('click', () => runNvidiaAction('ingest').catch(e => { $('nvidiaStatus').textContent='Error'; $('nvidiaOutput').textContent=String(e); }));
$('runNvidiaBenchmarkBtn')?.addEventListener('click', () => runNvidiaAction('benchmark').catch(e => { $('nvidiaStatus').textContent='Error'; $('nvidiaOutput').textContent=String(e); }));
document.addEventListener('click', e => {
  const adapterButton = e.target.closest('[data-adapter-action]');
  if (adapterButton) {
    const action = adapterButton.dataset.adapterAction;
    const serviceId = adapterButton.dataset.serviceId;
    runCandidateOperation(() => candidateMutation(`/api/adapters/${action}`, {service_id: serviceId}));
    return;
  }
  const portfolioButton = e.target.closest('[data-portfolio-run]');
  if (portfolioButton) {
    runCandidateOperation(() => runPortfolioBatch(portfolioButton.dataset.portfolioRun));
    return;
  }
  const coverageButton = e.target.closest('.coverage-metric-btn');
  if (coverageButton) {
    showMetricsCombination(coverageButton.dataset.combo || '');
    return;
  }
  const evidenceButton = e.target.closest('.view-project-evidence');
  if (evidenceButton) {
    openProjectEvidence(evidenceButton.dataset.comboId || '');
    return;
  }
  const pageButton = e.target.closest('.recommendation-page-btn');
  if (pageButton && recommendationState.tablePayload && recommendationState.tableResult) {
    const delta = Number(pageButton.dataset.pageDelta);
    if (Number.isInteger(delta) && delta !== 0) {
      recommendationState.tablePage += delta;
      renderRecommendationTable(
        recommendationState.tablePayload,
        recommendationState.tableResult,
        {preservePage: true},
      );
    }
    return;
  }
  const btn = e.target.closest('.mini-run-btn');
  if (btn) setRunSelection(btn.dataset.sheet, btn.dataset.embedding, btn.dataset.store, btn.dataset.reranker || 'all');
});
async function selectRecommendationSource(sourceType) {
  const normalized = sourceType === 'uploaded_project' ? 'uploaded_project' : 'official';
  recommendationState.controller?.abort();
  recommendationState.generation += 1;
  recommendationState.sourceType = normalized;
  recommendationState.active = null;
  $('recommendationOfficialSetField').hidden = normalized !== 'official';
  $('recommendationProjectField').hidden = true;
  $('recommendationRunField').hidden = normalized !== 'uploaded_project';
  populateRecommendationGroundtruths();
  closeRecommendationEvidence();
  if (normalized === 'official') {
    showOfficialRecommendations();
    return;
  }
  if (!recommendationState.projects.length) await loadResultSources();
  if (recommendationState.sourceType !== 'uploaded_project') return;
  const projectId = $('recommendationProject')?.value || recommendationState.projects[0]?.project_id;
  if (projectId) await loadProjectRuns(projectId);
  else if (recommendationState.sourceType === 'uploaded_project') {
    clearRecommendationView('No uploaded projects with matrix results');
  }
}

async function selectRecommendationDataset(datasetId) {
  if (datasetId === 'official') {
    $('recommendationSource').value = 'official';
    await selectRecommendationSource('official');
    return;
  }
  const project = recommendationState.projects.find(item => (item.dataset_id || `project:${item.project_id}`) === datasetId);
  if (!project) {
    recommendationState.runs = [];
    populateRecommendationGroundtruths();
    populateRecommendationRuns();
    clearRecommendationView('No completed results for this dataset');
    return;
  }
  $('recommendationSource').value = 'uploaded_project';
  $('recommendationProject').value = project.project_id;
  await selectRecommendationSource('uploaded_project');
}

async function selectRecommendationGroundtruth(groundtruthId) {
  if ($('recommendationGroundtruth')) $('recommendationGroundtruth').value = groundtruthId;
  if (recommendationState.sourceType !== 'uploaded_project') {
    showOfficialRecommendations();
    return;
  }
  populateRecommendationRuns();
  const projectId = $('recommendationProject')?.value;
  const runId = $('recommendationRun')?.value;
  if (projectId && runId) await loadProjectRun(projectId, runId);
  else clearRecommendationView('No completed run for this dataset and ground truth');
}

async function selectRecommendationResultSet(resultSetId) {
  if (recommendationState.sourceType === 'official') {
    if ($('recommendationOfficialSet')) $('recommendationOfficialSet').value = resultSetId;
    showOfficialRecommendations();
    return;
  }
  if ($('recommendationRun')) $('recommendationRun').value = resultSetId;
  const projectId = $('recommendationProject')?.value;
  if (projectId && resultSetId) await loadProjectRun(projectId, resultSetId);
}

$('recommendationSource')?.addEventListener('change', event => {
  selectRecommendationSource(event.target.value).catch(console.error);
});
$('recommendationDataset')?.addEventListener('change', event => {
  selectRecommendationDataset(event.target.value).catch(console.error);
});
$('recommendationGroundtruth')?.addEventListener('change', event => {
  selectRecommendationGroundtruth(event.target.value).catch(console.error);
});
$('recommendationOfficialSet')?.addEventListener('change', event => {
  selectRecommendationResultSet(event.target.value).catch(console.error);
});
$('recommendationProject')?.addEventListener('change', event => {
  if (event.target.value) loadProjectRuns(event.target.value).catch(console.error);
});
$('recommendationRun')?.addEventListener('change', event => {
  const projectId = $('recommendationProject')?.value;
  if (projectId && event.target.value) loadProjectRun(projectId, event.target.value).catch(console.error);
});
$('metricsDataset')?.addEventListener('input', event => {
  if ($('recommendationDataset')) $('recommendationDataset').value = event.target.value;
  selectRecommendationDataset(event.target.value).catch(console.error);
});
$('metricsGroundtruth')?.addEventListener('input', event => {
  selectRecommendationGroundtruth(event.target.value).catch(console.error);
});
$('metricsResultSet')?.addEventListener('input', event => {
  selectRecommendationResultSet(event.target.value).catch(console.error);
});
$('projectEvidenceMore')?.addEventListener('click', () => loadProjectEvidencePage({append: true}).catch(console.error));
$('projectEvidenceDialog')?.addEventListener('close', closeRecommendationEvidence);
document.querySelectorAll('.tab-btn').forEach(btn => btn.addEventListener('click', () => showPage(btn.dataset.page || 'overview')));
document.querySelectorAll('[data-status-card]').forEach(btn => btn.addEventListener('click', openStatusDialog));
showPage((location.hash || '#overview').slice(1));
$('uploadForm')?.addEventListener('submit', e => uploadDataset(e).catch(err => { $('uploadStatus').textContent='Error'; $('uploadOutput').textContent=String(err); }));
loadOptions().then(refresh).then(loadResultSources).catch(e => { $('statusPill').textContent = 'Error'; document.body.insertAdjacentHTML('beforeend', `<pre class="fatal">${esc(e.message)}</pre>`); });
loadCandidateOperations().catch(error => {
  setText('candidateOperationsStatus', 'Candidate state unavailable');
  setText('candidateActionMessage', error?.message || String(error));
});
