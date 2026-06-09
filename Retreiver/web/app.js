const $ = (id) => document.getElementById(id);
let state = { operational: {}, files: [], options: {} };
let benchmarkOptions = {};

const esc = (v) => String(v ?? '').replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
const num = (v) => Number.parseFloat(v || 0) || 0;
const uniq = (rows, key) => [...new Set(rows.map(r => r[key]).filter(Boolean))].sort();
const fmt = (v, d = 3) => Number.isFinite(num(v)) ? num(v).toFixed(d) : '—';
const costLabel = (r) => /openai|amazon/i.test(`${r.embedding || ''} ${r.reranker || ''}`) ? 'commercial key/cost' : 'open-source/VM cost';

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
    rows.map((r, i) => `<tr>${cols.map(c => `<td>${c.render ? c.render(r, i) : esc(r[c.key] ?? '')}</td>`).join('')}</tr>`).join('') + '</tbody>';
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
  if ($('runSheet')) $('runSheet').value = sheet || 'all';
  if ($('runEmbedding')) $('runEmbedding').value = embedding || 'all';
  if ($('runStore')) $('runStore').value = store || 'all';
  if ($('runRerankerMain')) $('runRerankerMain').value = reranker || 'all';
  $('runStatus').textContent = 'Selected missing benchmark option';
  $('runOutput').textContent = `Selected: chunker=${sheet || 'all'}, embedding=${embedding || 'all'}, vector DB=${store || 'all'}, reranker=${reranker || 'all'}\nClick Preflight first, then Run selected pipeline.`;
}

function renderCoverage(rows) {
  const latest = latestRows(rows);
  const opts = matrixOptions();
  const stores = opts.stores.length ? opts.stores : uniq(latest, 'store');
  const chunkers = opts.chunkers.length ? opts.chunkers : uniq(latest, 'sheet');
  const embeddings = opts.embeddings.length ? opts.embeddings : uniq(latest, 'embedding');
  const expected = chunkers.length * embeddings.length * stores.length;
  $('coverageHint').textContent = `${latest.length}/${expected || latest.length} chunker × embedding × vector DB combos run`;
  const latestMap = new Map(latest.map(r => [`${r.sheet}|${r.embedding}|${r.store}`, r]));
  const coverageRows = [];
  chunkers.forEach(sheet => embeddings.forEach(embedding => {
    const storeMap = {};
    stores.forEach(store => { storeMap[store] = latestMap.get(`${sheet}|${embedding}|${store}`); });
    coverageRows.push({ sheet, embedding, stores: storeMap });
  }));
  const cols = [
    {key:'sheet', label:'Chunker'},
    {key:'embedding', label:'Embedding'},
    ...stores.map(st => ({key:st, label:st, render:r => {
      const v = r.stores[st];
      if (v) return `<span class="coverage-ok">ok</span> <code>${fmt(v.total_store_seconds, 2)}s</code>`;
      return `<button class="mini-run-btn" data-sheet="${esc(r.sheet)}" data-embedding="${esc(r.embedding)}" data-store="${esc(st)}">Run</button>`;
    }})),
    {key:'total', label:'DBs ready', render:r => `${Object.values(r.stores).filter(Boolean).length}/${stores.length}`}
  ];
  table($('coverageTable'), coverageRows, cols);
}

function buildEvidenceRows(retrievalSmokes, rerankerSmokes) {
  const base = (retrievalSmokes || []).map(r => ({...r, reranker: 'none', evidence_type: 'initial topK retrieval'}));
  const reranked = (rerankerSmokes || []).map(r => ({...r, evidence_type: 'reranked final top3'}));
  return [...reranked, ...base].map(r => {
    const rawHits = r.hits || [];
    const hits = rawHits.map((hit, i) => ({
      rank: hit.rank || i + 1,
      pdf_name: hit.pdf_name || hit.pdf || hit.source || '—',
      score: hit.score ?? hit.relevance_score ?? hit.relevanceScore ?? hit.similarity ?? '',
      text: evidenceTextFromHit(hit),
    }));
    return {
      ...r,
      reranker: r.reranker || 'none',
      top_pdf: (hits[0] || {}).pdf_name || '—',
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

function renderEvidenceDetails(row) {
  const hits = (row.evidence_hits || []).filter(h => h.text).slice(0, 10);
  if (!hits.length) return '<span class="muted-text">No paragraph returned in artifact</span>';
  const stage = row.evidence_type === 'reranked final top3' ? 'Final reranked evidence' : 'Initial vector topK evidence';
  const body = hits.map((h, i) => `<article class="evidence-hit ${i < 3 ? 'final-hit' : ''}">
    <div><strong>${esc(stage)} · Rank ${esc(h.rank)}</strong><span>${esc(h.pdf_name || '—')}${h.score !== '' ? ` · score ${esc(fmt(h.score, 4))}` : ''}</span></div>
    <p>${esc(h.text)}</p>
  </article>`).join('');
  return `<div class="evidence-stage-note">${esc(stage)}. First three rows are the final answer candidates after reranking when a reranker artifact is selected.</div><div class="evidence-hit-list">${body}</div>`;
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
    {key:'top_pdf', label:'Top PDF/source', render:r=>esc((r.top_pdf || '').slice(0, 58))},
    {key:'initial_topk_count', label:'Initial K'},
    {key:'evidence_type', label:'Stage'},
    {key:'evidence_snippet', label:'TopK to rerank to top3 evidence', render:r=>`<details class="snippet"><summary>${esc(shortEvidenceLabel(r))}</summary>${renderEvidenceDetails(r)}</details>`},
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
  const benchmarkReport = benchmark.report || {};
  const benchmarkRows = benchmark.summary || [];
  const benchmarkDetails = benchmark.details || [];
  const bestBenchmark = benchmarkReport.ok ? (benchmarkRows[0] || benchmarkReport.best || null) : null;
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
  const nextStep = !health.ok ? 'Run health check first.' : !ingestion.ok ? 'Run Ingest current data before smoke/benchmark.' : !smoke.ok ? 'Run Test one query to verify retrieval evidence.' : !benchmarkReport.ok ? 'Run NVIDIA benchmark on groundtruth_500.csv.' : 'Benchmark artifact is ready.';
  cards.innerHTML = `
    <article class="nvidia-stage ${health.ok ? 'ok' : 'warn'}"><span>01</span><strong>Services</strong><p>${healthyCount}/${checkRows.length || 3} healthy · rag-server, ingestor, frontend</p></article>
    <article class="nvidia-stage ${ingestion.ok ? 'ok' : 'warn'}"><span>02</span><strong>Current data</strong><p>${ingestion.files ? `${ingestion.files.length} files selected · ${uploadedBatches}/${batches.length || 0} batches uploaded` : `Collection ${esc(collection)} has no ingestion artifact yet.`}</p></article>
    <article class="nvidia-stage ${benchmarkReport.ok ? 'ok' : 'warn'}"><span>03</span><strong>Benchmark</strong><p>${bestBenchmark ? `R@5 ${(num(bestBenchmark.recall_at_5)*100).toFixed(1)}% · MRR ${fmt(bestBenchmark.mrr,3)} · ${fmt(bestBenchmark.avg_latency_seconds,3)}s/query` : (benchmarkError || 'No NVIDIA benchmark artifact yet.')}</p></article>
    <article class="nvidia-stage action"><span>Next</span><strong>${esc(nextStep)}</strong><p>${smokeError && !smoke.ok ? esc(smokeError).slice(0, 160) : 'Keep the benchmark path evidence-led: ingest current PDFs, test one query, then run a limited benchmark.'}</p></article>`;

  const serviceRows = checkRows.map(r => ({kind:'health', name:r.name, status:r.ok ? 'ok' : 'check', detail:r.error || r.url, latency:r.latency_ms ? `${fmt(r.latency_ms,1)} ms` : '—'}));
  const artifactRows = files.map(f => ({kind:'artifact', name:f.split('/').pop(), status:'present', detail:f, latency:'—'}));
  const rows = [
    ...serviceRows,
    ...(smoke.created_at ? [{kind:'smoke', name:smoke.mode || 'search', status:smoke.ok ? 'ok' : 'check', detail:smoke.query || smokeError || '', latency:smoke.response?.latency_seconds ? `${fmt(smoke.response.latency_seconds,3)}s` : '—'}] : []),
    ...(ingestion.created_at ? [{kind:'ingestion', name:collection, status:ingestion.ok ? 'ok' : 'check', detail:`${(ingestion.files || []).length} files · ${batches.length} batches`, latency:'—'}] : []),
    ...(benchmarkReport.created_at ? [{kind:'benchmark', name:collection, status:benchmarkReport.ok ? 'ok' : 'check', detail:`${benchmarkReport.evaluated_rows || 0} rows · ${benchmarkReport.groundtruth || ''}`, latency:benchmarkReport.total_seconds ? `${fmt(benchmarkReport.total_seconds,2)}s total` : '—'}] : []),
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
      benchmarkCards.innerHTML = `<article class="run-result-card ok"><span>Recall@5</span><strong>${(num(bestBenchmark.recall_at_5)*100).toFixed(1)}%</strong><small>${esc(bestBenchmark.evaluated_queries || benchmarkReport.evaluated_rows || '—')} evaluated queries</small></article>
        <article class="run-result-card"><span>MRR</span><strong>${fmt(bestBenchmark.mrr,3)}</strong><small>First relevant rank strength</small></article>
        <article class="run-result-card"><span>Avg sec/query</span><strong>${fmt(bestBenchmark.avg_latency_seconds,3)}s</strong><small>${esc(bestBenchmark.successful_queries || benchmarkReport.successful_queries || '—')} successful service calls</small></article>`;
    } else {
      benchmarkCards.innerHTML = `<div class="empty-state">${esc(benchmarkError || 'Run NVIDIA benchmark after current PDFs are ingested into the collection.')}</div>`;
    }
  }
  $('nvidiaBenchmarkHint') && ($('nvidiaBenchmarkHint').textContent = benchmarkReport.created_at ? `${benchmarkReport.evaluated_rows || 0} rows · ${benchmarkReport.ok ? 'ok' : 'check'}` : 'No benchmark artifact yet');
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

function renderQwenAnalysis(analysis) {
  const hint = $('qwenAnalysisHint');
  const cards = $('qwenAnalysisCards');
  const tbl = $('qwenAnalysisTable');
  if (!hint || !cards || !tbl) return;
  const report = analysis?.report || {};
  const summary = analysis?.summary || [];
  const details = analysis?.details || [];
  const totalPaired = Number(report.paired_rows || 0);
  const totalWorse = details.filter(r => r.verdict === 'worse').length;
  const totalImproved = details.filter(r => r.verdict === 'improved').length;
  const worst = summary[0] || null;
  hint.textContent = totalPaired ? `${totalPaired} paired rows` : 'Waiting for paired run';
  if (!totalPaired) {
    const causes = report.likely_causes || ['Run the complete pipeline with Qwen selected to create paired baseline and reranked artifacts.'];
    cards.innerHTML = `<div class="empty-state">${causes.map(esc).join('<br>')}</div>`;
  } else {
    cards.innerHTML = `<article class="run-result-card ${totalWorse ? 'warn' : 'ok'}"><span>Qwen demotions</span><strong>${totalWorse}</strong><small>Relevant hit moved lower than baseline</small></article>
      <article class="run-result-card ok"><span>Qwen improvements</span><strong>${totalImproved}</strong><small>Relevant hit moved higher</small></article>
      <article class="run-result-card"><span>Worst pipeline</span><strong>${worst ? esc(`${worst.sheet} · ${worst.embedding} · ${worst.store}`) : '—'}</strong><small>Avg ΔMRR ${worst ? fmt(worst.avg_delta_mrr,3) : '—'}</small></article>`;
  }
  const rows = details.filter(r => r.verdict === 'worse').slice(0, 40);
  table(tbl, rows.length ? rows : details.slice(0, 40), [
    {key:'verdict', label:'Verdict', render:r=>`<span class="badge ${r.verdict === 'worse' ? 'warn' : 'ok'}">${esc(r.verdict || '')}</span>`},
    {key:'query', label:'Query', render:r=>esc((r.query || '').slice(0, 90))},
    {key:'sheet', label:'Chunker'},
    {key:'embedding', label:'Embedding'},
    {key:'store', label:'Vector DB'},
    {key:'base_rank', label:'Base rank'},
    {key:'reranked_rank', label:'Qwen rank'},
    {key:'delta_mrr', label:'ΔMRR'},
    {key:'expected_pdf', label:'Expected PDF', render:r=>esc((r.expected_pdf || '').slice(0, 60))},
    {key:'reranked_top_pdf', label:'Qwen top PDF', render:r=>esc((r.reranked_top_pdf || '').slice(0, 60))},
  ]);
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
  return num(r.winner_score) || ((0.45*num(r.recall_at_5)) + (0.30*num(r.mrr)) + (0.20*num(r.ndcg_at_5)) + (0.05*(1/(1+num(r.avg_latency_seconds)))));
}

function displayScore(r) {
  return fmt(metricScore(r), 3);
}

function renderEvaluation(evaluation) {
  const rows = evaluation?.summary || [];
  const section = $('qualitySection');
  if (!section) return;
  const report = evaluation?.report || {};
  const baseRows = rows.map(r => ({...r, reranker: r.reranker || 'none'}));
  const rerankedRows = (evaluation?.reranked?.summary || []).map(r => ({...r, reranker: r.reranker || 'none'}));
  const displayRows = [...baseRows, ...rerankedRows].sort((a,b)=>metricScore(b)-metricScore(a) || num(b.recall_at_5)-num(a.recall_at_5));
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
  $('qualityHint').textContent = filteredRows.length ? `${filteredRows.length}/${displayRows.length} configs · ${report.groundtruth_rows || evaluation?.reranked?.report?.groundtruth_rows || '—'} GT rows` : 'No matching configs';
  $('qualityInsight').innerHTML = bestRow ? `<strong>Current winner:</strong> ${esc(pipelineLabel(bestRow))} <span>Score ${displayScore(bestRow)} · R@5 ${(num(bestRow.recall_at_5)*100).toFixed(1)}% · MRR ${fmt(bestRow.mrr,3)} · ${fmt(bestRow.avg_latency_seconds,3)}s avg/query across ${esc(bestRow.evaluated_queries || '—')} queries</span>` : '<span>No evaluated combination matches these filters.</span>';
  table($('qualityTable'), filteredRows.slice(0, 60), [
    {key:'serial', label:'S/No.', render:(r,i)=>String(i + 1)},
    {key:'winner_score', label:'Score', render:displayScore},
    {key:'sheet', label:'Chunker'},
    {key:'embedding', label:'Embedding'},
    {key:'store', label:'Vector DB'},
    {key:'reranker', label:'Reranker', render:r=>esc(r.reranker || 'none')},
    {key:'evaluated_queries', label:'Queries'},
    {key:'recall_at_1', label:'R@1'},
    {key:'recall_at_3', label:'R@3'},
    {key:'recall_at_5', label:'R@5'},
    {key:'recall_at_10', label:'R@10'},
    {key:'mrr', label:'MRR'},
    {key:'precision_at_5', label:'P@5'},
    {key:'ndcg_at_5', label:'nDCG@5'},
    {key:'avg_first_relevant_rank', label:'Avg first relevant rank'},
    {key:'no_hit_queries', label:'No-hit queries'},
    {key:'avg_latency_seconds', label:'Avg latency/query'},
    {key:'cost', label:'Cost lane', render:costLabel},
  ]);
  renderBestMethods(filteredRows.length ? filteredRows : displayRows);
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
  const fastest = [...rows].sort((a,b)=>num(a.avg_latency_seconds)-num(b.avg_latency_seconds))[0];
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
  return [...(evaluation?.summary || []), ...(evaluation?.reranked?.summary || [])]
    .map(r => ({...r, reranker: r.reranker || 'none'}))
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
  if (!hint || !scoreEl || !scatterEl) return;
  const top = rows.slice(0, 10);
  if (!top.length) {
    hint.textContent = 'Waiting for evaluation';
    scoreEl.innerHTML = '<div class="empty-state">No top 10 pipeline data yet.</div>';
    scatterEl.innerHTML = '<div class="empty-state">No quality-latency plot yet.</div>';
    return;
  }
  hint.textContent = `${top.length} best pipelines`;
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

function renderPipelineComparison(evaluation) {
  const rows = evaluatedRows(evaluation);
  const grid = $('comparisonGrid');
  const hint = $('comparisonHint');
  if (!grid || !hint) return;
  if (!rows.length) {
    renderTop10Visualizations(rows);
    hint.textContent = 'Waiting for evaluation';
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
  hint.textContent = `${rows.length} evaluated configs`;
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

function showPage(page) {
  document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.toggle('active', btn.dataset.page === page));
  document.querySelectorAll('[data-page-panel]').forEach(panel => panel.classList.toggle('active', panel.dataset.pagePanel === page));
  history.replaceState(null, '', `#${page}`);
}

async function uploadDataset(ev) {
  ev.preventDefault();
  const file = $('datasetFile')?.files?.[0];
  if (!file) {
    $('uploadStatus').textContent = 'Choose file';
    $('uploadOutput').textContent = 'Select a PDF, ZIP, CSV, or XLSX first.';
    return;
  }
  const form = new FormData();
  form.append('file', file);
  form.append('label', $('datasetLabel')?.value || file.name.replace(/\.[^.]+$/, ''));
  $('uploadStatus').textContent = 'Uploading';
  $('uploadOutput').textContent = `Uploading ${file.name}...`;
  const res = await fetch('/api/upload-dataset', {method: 'POST', body: form});
  const payload = await res.json();
  $('uploadStatus').textContent = res.ok ? 'Uploaded' : 'Error';
  $('uploadOutput').textContent = JSON.stringify(payload, null, 2);
}


function renderDocumentRepository(repo) {
  const rows = repo?.rows || [];
  const hint = $('documentRepositoryHint');
  if (hint) hint.textContent = rows.length ? `${repo.ready_count || 0}/${repo.total || rows.length} chunked · ${repo.review_count || 0} in review` : 'No repository scan yet';
  const cards = $('documentRepositoryCards');
  if (cards) {
    cards.innerHTML = `<article class="run-result-card ok"><span>Repository PDFs</span><strong>${esc(repo.total || rows.length || 0)}</strong><small>from data/pdfs</small></article>
      <article class="run-result-card"><span>Chunked documents</span><strong>${esc(repo.ready_count || 0)}</strong><small>present in benchmark workbook</small></article>
      <article class="run-result-card warn"><span>Review queue</span><strong>${esc(repo.review_count || 0)}</strong><small>shown only in repository/audit views</small></article>`;
  }
  table($('documentRepositoryTable'), rows, [
    {key:'pdf_name', label:'PDF', render:r=>esc((r.pdf_name || '').slice(0, 88))},
    {key:'status', label:'Readiness'},
    {key:'chunked_rows', label:'Chunk rows'},
    {key:'pages', label:'Pages'},
    {key:'size_mb', label:'Size MB'},
    {key:'parser_method', label:'Parser', render:r=> (r.parser_method === 'PyPDF2_fallback' ? '<span class="badge warn">text-only fallback</span>' : esc(r.parser_method || '—'))},
    {key:'repository_path', label:'Path', render:r=>`<code>${esc(r.repository_path || '')}</code>`},
    {key:'note', label:'Note'},
  ]);
}

function renderOperational() {
  const op = state.operational || {};
  const ingestion = op.ingestion || {};
  const rows = ingestion.rows || [];
  const latest = latestRows(rows);
  const retrieval = op.retrieval_smokes || [];
  const rerank = op.reranker_smokes || [];
  const health = op.service_health || [];
  const snapshot = op.vm_snapshot || {};
  const optsForStatus = matrixOptions();
  const embeddings = optsForStatus.embeddings.length ? optsForStatus.embeddings : uniq(latest, 'embedding');
  const stores = optsForStatus.stores.length ? optsForStatus.stores : uniq(latest, 'store');
  const evaluation = op.evaluation || {};
  const evalRows = [...(evaluation.summary || []), ...(evaluation.reranked?.summary || [])].map(r => ({...r, reranker: r.reranker || 'none'})).sort((a,b)=>metricScore(b)-metricScore(a) || num(b.recall_at_5)-num(a.recall_at_5));
  const best = evalRows[0] || null;
  const fastest = [...evalRows].sort((a,b) => num(a.avg_latency_seconds) - num(b.avg_latency_seconds))[0] || null;

  $('modeLabel').textContent = 'Live artifacts';
  $('latestRun').textContent = ingestion.latest_run_id || snapshot?.ingestion?.latest_run_id || '—';
  $('snapshotAt').textContent = snapshot.created_at ? new Date(snapshot.created_at).toLocaleString() : '—';
  $('groundTruthStatus').textContent = evaluation?.report?.groundtruth_rows ? `${evaluation.report.groundtruth_rows} rows evaluated` : ((evaluation?.groundtruth_files || []).length ? 'Loaded, not evaluated' : 'Pending');
  $('pdfStatus').textContent = `${op.extracted_pdf_count || 0}/${op.known_pdf_count || 0}`;
  $('pdfNote').textContent = 'chunked or ready in repository';
  $('comboStatus').textContent = String(op.known_matrix_count || ingestion.combo_count || latest.length || 0);
  $('comboNote').textContent = op.options_formula || 'chunkers × embeddings × vector DBs × retrieval × rerankers';
  $('retrievalStatus').textContent = String(retrieval.length + rerank.length);
  $('retrievalNote').textContent = `${retrieval.length + rerank.length} loaded rows from ${op.retrieval_smoke_total || 0} retrieval + ${op.reranker_smoke_total || 0} reranker artifact files. Raw totals are generated evidence files, not query count.`;
  $('bestR5Status').textContent = best ? `${(num(best.recall_at_5)*100).toFixed(1)}%` : '—';
  $('bestConfigNote').textContent = best ? pipelineLabel(best) : 'best accuracy score';
  $('bestLatencyStatus').textContent = fastest ? `${fmt(fastest.avg_latency_seconds, 3)}s` : '—';
  $('bestLatencyNote').textContent = fastest ? `${pipelineLabel(fastest)} · avg/query` : 'average query latency';
  $('embeddingStatus').textContent = embeddings.length ? embeddings.join(' + ') : '—';
  $('dbStatus').textContent = stores.length ? stores.join(' + ') : '—';
  $('ingestionHint').textContent = `${latest.length} latest successful component rows`;
  $('rerankHint').textContent = `${rerank.length} artifacts`;
  $('artifactHint').textContent = `${state.files.length} files`;
  $('healthHint').textContent = `${health.filter(h => h.ok).length}/${health.length || 0} healthy`;
  const pdfAudit = op.pdf_audit || {};
  $('pdfAuditHint') && ($('pdfAuditHint').textContent = pdfAudit.total ? `${pdfAudit.ok_count || 0}/${pdfAudit.total} ready · review hidden from KPI` : 'No audit file yet');

  renderCoverage(rows);
  renderServices(health);
  renderRetrieval(retrieval, rerank);
  renderEvaluation(op.evaluation);
  renderPipelineComparison(op.evaluation);
  renderHallucination(op.hallucination || {});
  renderNvidiaRag(op.nvidia_rag || {});
  renderQwenAnalysis(op.reranker_analysis || {});
  renderDocumentRepository(op.document_repository || {});

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

async function refresh() {
  $('statusPill').textContent = 'Refreshing';
  const data = await api('/api/results');
  state = { operational: data.operational || {}, files: data.files || [], options: data.options || {} };
  renderOperational();
  $('statusPill').textContent = 'Live';
}

async function loadOptions() {
  benchmarkOptions = await api('/api/options');
  fillRunSelect('runSheet', benchmarkOptions.chunkers || [], 'All chunkers');
  fillRunSelect('runEmbedding', benchmarkOptions.embeddings || [], 'All embeddings');
  fillRunSelect('runStore', benchmarkOptions.vector_stores || [], 'All DBs');
  const mainReranker = $('runRerankerMain');
  if (mainReranker) {
    const cur = mainReranker.value || 'all';
    mainReranker.innerHTML = '<option value="all">All OSS rerankers</option><option value="qwen3_4b_rerank">qwen3_4b_rerank</option><option value="bge-reranker-base">bge-reranker-base</option><option value="none">No reranker baseline only</option>';
    mainReranker.value = ['all','qwen3_4b_rerank','bge-reranker-base','none'].includes(cur) ? cur : 'all';
  }
}

function selectedParams() {
  const p = new URLSearchParams();
  if ($('runSheet')?.value) p.append('sheet', $('runSheet').value);
  if ($('runEmbedding')?.value) p.append('embedding', $('runEmbedding').value);
  if ($('runStore')?.value) p.append('store', $('runStore').value);
  const reranker = $('runRerankerMain')?.value;
  if (reranker) p.append('reranker', reranker);
  const limit = $('runLimit')?.value || '0';
  p.set('limit', limit);
  p.set('chunk_limit', limit);
  p.set('query_limit', $('runQueryLimit')?.value || '0');
  p.set('top_k', $('runTopK')?.value || '10');
  p.set('groundtruth', $('runGroundtruth')?.value || 'data/groundtruth/groundtruth_500.csv');
  if ($('runFresh')?.checked) p.set('fresh_run', '1');
  p.set('max_runs', '1');
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
  const dataPath = ($('nvidiaDataPath')?.value || 'data/pdfs').trim();
  const groundtruth = ($('nvidiaGroundtruth')?.value || 'data/groundtruth/groundtruth_500.csv').trim();
  if (dataPath) p.set('path', dataPath);
  if (groundtruth) p.set('groundtruth', groundtruth);
  if ($('nvidiaCreateCollection')?.checked) p.set('create_collection', '1');
  if ($('nvidiaPoll')?.checked) p.set('poll', '1');
  if (kind === 'smoke') {
    p.set('mode', $('nvidiaMode')?.value || 'search');
    const query = ($('nvidiaQuery')?.value || '').trim();
    if (query) p.append('query', query);
    if ($('nvidiaAgentic')?.checked) p.set('agentic', '1');
  }
  if ($('nvidiaDisableReranker')?.checked) p.set('disable_reranker', '1');
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
  $('runStatus').textContent = ok ? 'Ready' : 'Needs inputs';
  $('runResultCards').innerHTML = `<article class="run-result-card ${ok ? 'ok' : 'warn'}"><span>Requirements</span><strong>${ok ? 'Ready' : 'Blocked'}</strong><small>${missing.length ? `${missing.length} item(s) needed` : 'All required inputs found'}</small></article>
    <article class="run-result-card"><span>Pipeline size</span><strong>${esc(payload.combo_count || 0)}</strong><small>chunker · embedding · DB combinations</small></article>
    <article class="run-result-card"><span>Ground truth</span><strong>${esc(payload.groundtruth || '—')}</strong><small>${payload.query_limit ? `${payload.query_limit} query limit` : 'all queries'}</small></article>`;
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

async function runPreflight() {
  const p = selectedParams();
  $('runStatus').textContent = 'Checking';
  const payload = await api(`/api/run/preflight-complete-pipeline?${p.toString()}`, {method:'POST'});
  renderPreflight(payload);
  return payload;
}

function parseLiveOutputRows(output) {
  const lines = (output || '').split('\n').filter(Boolean);
  const stageRows = lines.filter(l => l.includes('START ') || l.includes('DONE ') || l.includes('FAILED') || l.includes('groundtruth_rows=') || l.includes('OK ')).slice(-80);
  return stageRows.map((line, i) => ({i:i+1, line}));
}

async function pollRunJob(jobId) {
  const payload = await api(`/api/run/status?job_id=${encodeURIComponent(jobId)}`);
  const output = payload.output || '';
  const running = !!payload.running;
  $('runStatus').textContent = running ? 'Running' : (payload.exit_code === 0 ? 'Done' : 'Check output');
  $('runOutput').textContent = output;
  $('runResultCards').innerHTML = `<article class="run-result-card ${payload.exit_code === 0 ? 'ok' : running ? '' : 'warn'}"><span>Status</span><strong>${running ? 'Running' : payload.exit_code === 0 ? 'Passed' : 'Needs attention'}</strong><small>job ${esc(jobId)}</small></article>
    <article class="run-result-card"><span>Live log</span><strong>${output.split('\n').filter(Boolean).length}</strong><small>${esc(payload.log_path || '')}</small></article>
    <article class="run-result-card"><span>Result view</span><strong>Quality + evidence</strong><small>Tables refresh when stages finish</small></article>`;
  table($('runResultTable'), parseLiveOutputRows(output), [
    {key:'i', label:'#'},
    {key:'line', label:'Live stage/output', render:r=>esc(r.line).slice(0, 260)},
  ]);
  if (running) {
    setTimeout(() => pollRunJob(jobId).catch(e => { $('runStatus').textContent='Error'; $('runOutput').textContent=String(e); }), 1500);
  } else {
    await refresh();
  }
}

async function runCompletePipeline() {
  const preflight = await runPreflight();
  if (!preflight.ok) return;
  if ((preflight.combo_count || 0) > 12) {
    const ok = window.confirm(`This will run ${preflight.combo_count} pipeline combinations through ingestion, retrieval, reranking and evaluation. Continue?`);
    if (!ok) {
      $('runStatus').textContent = 'Cancelled';
      return;
    }
  }
  const p = selectedParams();
  $('runStatus').textContent = 'Starting';
  const payload = await api(`/api/run/complete-pipeline?${p.toString()}`, {method:'POST'});
  $('runOutput').textContent = `Started job ${payload.job_id}\n${payload.output || ''}`;
  await pollRunJob(payload.job_id);
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
    p.set('top_k', $('runTopK')?.value || '10');
    p.set('max_runs', String(benchmarkOptions.matrix_count || 135));
  }
  if (kind === 'rerank') {
    p.set('top_k', $('runTopK')?.value || '10');
    p.set('limit', $('rerankerLimit')?.value || '100');
    const mainChoice = $('runRerankerMain')?.value || 'qwen3_4b_rerank';
    let selected = [];
    if (mainChoice === 'all') {
      selected = ['bge-reranker-base', 'qwen3_4b_rerank'];
    } else if (mainChoice && mainChoice !== 'none') {
      selected = [mainChoice];
    } else {
      selected = [...($('runReranker')?.selectedOptions || [])].map(o=>o.value);
    }
    selected.forEach(r => p.append('reranker', r));
  }
  if (kind === 'hallucination' || kind === 'hallucinationLlm') {
    p.set('limit', $('rerankerLimit')?.value || '120');
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

['retrievalChunkerFilter','retrievalDbFilter','retrievalEmbeddingFilter','retrievalRerankerFilter'].forEach(id => $(id)?.addEventListener('input', () => renderRetrieval(state.operational?.retrieval_smokes || [], state.operational?.reranker_smokes || [])));
['qualityComboFilter','qualityChunkerFilter','qualityEmbeddingFilter','qualityDbFilter','qualityRerankerFilter'].forEach(id => $(id)?.addEventListener('input', () => renderEvaluation(state.operational?.evaluation || {})));
$('refreshBtn').addEventListener('click', () => refresh().catch(e => { $('statusPill').textContent = 'Error'; console.error(e); }));
$('runPreflightBtn')?.addEventListener('click', () => runPreflight().catch(e => { $('runStatus').textContent='Error'; $('runOutput').textContent=String(e); }));
$('runCompletePipelineBtn')?.addEventListener('click', () => runCompletePipeline().catch(e => { $('runStatus').textContent='Error'; $('runOutput').textContent=String(e); }));
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
  const btn = e.target.closest('.mini-run-btn');
  if (btn) setRunSelection(btn.dataset.sheet, btn.dataset.embedding, btn.dataset.store);
});
document.querySelectorAll('.tab-btn').forEach(btn => btn.addEventListener('click', () => showPage(btn.dataset.page || 'overview')));
showPage((location.hash || '#overview').slice(1));
$('uploadForm')?.addEventListener('submit', e => uploadDataset(e).catch(err => { $('uploadStatus').textContent='Error'; $('uploadOutput').textContent=String(err); }));
$('openTestOptionsBtn')?.addEventListener('click', () => $('testOptionsDialog')?.showModal());
loadOptions().then(refresh).catch(e => { $('statusPill').textContent = 'Error'; document.body.insertAdjacentHTML('beforeend', `<pre class="fatal">${esc(e.message)}</pre>`); });
