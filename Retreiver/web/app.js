const $ = (id) => document.getElementById(id);
let state = { operational: {}, files: [], options: {} };
let benchmarkOptions = {};

const esc = (v) => String(v ?? '').replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
const num = (v) => Number.parseFloat(v || 0) || 0;
const uniq = (rows, key) => [...new Set(rows.map(r => r[key]).filter(Boolean))].sort();
const fmt = (v, d = 3) => Number.isFinite(num(v)) ? num(v).toFixed(d) : '—';

async function api(path) {
  const res = await fetch(path, {cache: 'no-store'});
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

function fillRunSelect(id, values) {
  const el = $(id); if (!el) return;
  const cur = el.value || (values[0] || 'all');
  el.innerHTML = values.map(v => `<option value="${esc(v)}">${esc(v)}</option>`).join('');
  el.value = values.includes(cur) ? cur : (values[0] || 'all');
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

function renderCoverage(rows) {
  const latest = latestRows(rows);
  const stores = uniq(latest, 'store');
  $('coverageHint').textContent = `${latest.length} successful combos`;
  const grouped = new Map();
  latest.forEach(r => {
    const key = `${r.sheet}|${r.embedding}`;
    if (!grouped.has(key)) grouped.set(key, { sheet: r.sheet, embedding: r.embedding, stores: {} });
    grouped.get(key).stores[r.store] = r;
  });
  const coverageRows = [...grouped.values()].sort((a,b) => `${a.sheet}${a.embedding}`.localeCompare(`${b.sheet}${b.embedding}`));
  const cols = [
    {key:'sheet', label:'Chunker'},
    {key:'embedding', label:'Embedding'},
    ...stores.map(st => ({key:st, label:st, render:r => {
      const v = r.stores[st];
      return v ? `<span class="coverage-ok">ok</span> <code>${fmt(v.total_store_seconds, 2)}s</code>` : '<span class="coverage-miss">not run</span>';
    }})),
    {key:'total', label:'DBs ready', render:r => `${Object.keys(r.stores).length}/${stores.length}`}
  ];
  table($('coverageTable'), coverageRows, cols);
}

function filteredRetrieval(smokes) {
  const db = $('retrievalDbFilter')?.value || 'all';
  const emb = $('retrievalEmbeddingFilter')?.value || 'all';
  return smokes.filter(r => (db === 'all' || r.store === db) && (emb === 'all' || r.embedding === emb));
}

function renderRetrieval(smokes) {
  fillSelect('retrievalDbFilter', uniq(smokes, 'store'), 'All DBs');
  fillSelect('retrievalEmbeddingFilter', uniq(smokes, 'embedding'), 'All embeddings');
  const rows = filteredRetrieval(smokes).slice(0, 100).map(r => ({
    ...r,
    top_pdf: ((r.hits || [])[0] || {}).pdf_name || '—',
  }));
  table($('retrievalTable'), rows, [
    {key:'query', label:'Query', render:r=>esc((r.query || '').slice(0, 64))},
    {key:'store', label:'DB'},
    {key:'embedding', label:'Embedding'},
    {key:'sheet', label:'Chunker'},
    {key:'top_pdf', label:'Top PDF', render:r=>esc((r.top_pdf || '').slice(0, 70))},
    {key:'retrieved_count', label:'Hits'},
    {key:'retrieval_seconds', label:'Latency', render:r=>`${fmt(r.retrieval_seconds, 3)}s`},
  ]);
}

function renderEvaluation(evaluation) {
  const rows = evaluation?.summary || [];
  const section = $('qualitySection');
  if (!section) return;
  section.classList.toggle('hidden', rows.length === 0);
  $('qualityHint').textContent = rows.length ? `${rows.length} evaluated configs` : 'No ground truth yet';
  table($('qualityTable'), rows.slice(0, 40), [
    {key:'sheet', label:'Chunker'},
    {key:'embedding', label:'Embedding'},
    {key:'store', label:'DB'},
    {key:'evaluated_queries', label:'Queries'},
    {key:'recall_at_1', label:'R@1'},
    {key:'recall_at_3', label:'R@3'},
    {key:'recall_at_5', label:'R@5'},
    {key:'mrr', label:'MRR'},
    {key:'ndcg_at_5', label:'nDCG@5'},
    {key:'avg_latency_seconds', label:'Avg sec'},
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
  const embeddings = uniq(latest, 'embedding');
  const stores = uniq(latest, 'store');

  $('modeLabel').textContent = 'Live artifacts';
  $('latestRun').textContent = ingestion.latest_run_id || snapshot?.ingestion?.latest_run_id || '—';
  $('snapshotAt').textContent = snapshot.created_at ? new Date(snapshot.created_at).toLocaleString() : '—';
  $('pdfStatus').textContent = `${op.extracted_pdf_count || 0}/${op.known_pdf_count || 0}`;
  $('pdfNote').textContent = `${op.failed_pdf_count || 0} PDFs need OCR decision`;
  $('comboStatus').textContent = String(ingestion.combo_count ?? latest.length ?? 0);
  $('retrievalStatus').textContent = String(retrieval.length || 0);
  $('rerankerStatus').textContent = String(rerank.length || 0);
  $('embeddingStatus').textContent = embeddings.length ? embeddings.join(' + ') : '—';
  $('dbStatus').textContent = stores.length ? stores.join(' + ') : '—';
  $('ingestionHint').textContent = `${latest.length} latest successful rows`;
  $('rerankHint').textContent = `${rerank.length} artifacts`;
  $('artifactHint').textContent = `${state.files.length} files`;
  $('healthHint').textContent = `${health.filter(h => h.ok).length}/${health.length || 0} healthy`;

  renderCoverage(rows);
  renderServices(health);
  renderRetrieval(retrieval);
  renderEvaluation(op.evaluation);

  table($('ingestionTable'), latest.slice(0, 18), [
    {key:'run_id', label:'Run'},
    {key:'sheet', label:'Chunker'},
    {key:'embedding', label:'Embedding'},
    {key:'store', label:'DB'},
    {key:'chunk_count', label:'Chunks'},
    {key:'collection_or_table', label:'Collection/table', render:r=>`<code>${esc((r.collection_or_table || '').slice(0, 46))}</code>`},
    {key:'total_store_seconds', label:'DB sec', render:r=>fmt(r.total_store_seconds, 2)},
    {key:'search_hits', label:'Hits'},
    {key:'status', label:'Status', render:r=>`<span class="badge ok">ok</span>`}
  ]);
  table($('rerankerTable'), rerank.slice(0, 10), [
    {key:'reranker', label:'Reranker'},
    {key:'query', label:'Query', render:r=>esc((r.query || '').slice(0, 58))},
    {key:'sheet', label:'Chunker'},
    {key:'embedding', label:'Embedding'},
    {key:'rerank_seconds', label:'Sec', render:r=>fmt(r.rerank_seconds, 3)},
    {key:'retrieved_count', label:'Retrieved'},
    {key:'artifact', label:'Artifact', render:r=>`<code>${esc((r.artifact || '').slice(0, 56))}</code>`}
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
  fillRunSelect('runSheet', benchmarkOptions.chunkers || []);
  fillRunSelect('runEmbedding', benchmarkOptions.embeddings || []);
  fillRunSelect('runStore', benchmarkOptions.vector_stores || []);
}

function selectedParams() {
  const p = new URLSearchParams();
  if ($('runSheet')?.value) p.append('sheet', $('runSheet').value);
  if ($('runEmbedding')?.value) p.append('embedding', $('runEmbedding').value);
  if ($('runStore')?.value) p.append('store', $('runStore').value);
  const limit = $('runLimit')?.value || '0';
  p.set('limit', limit);
  p.set('max_runs', '1');
  return p;
}

async function runAction(kind) {
  const endpoints = {
    ingest: '/api/run/ingest-selected',
    smoke: '/api/run/retrieval-smoke',
    eval: '/api/run/evaluate-groundtruth',
  };
  const p = selectedParams();
  if (kind === 'smoke') {
    p.delete('limit');
    p.set('max_runs', '1');
    ($('runQueries')?.value || '').split('\n').map(s=>s.trim()).filter(Boolean).forEach(q => p.append('query', q));
  }
  $('runStatus').textContent = 'Running';
  $('runOutput').textContent = `POST ${endpoints[kind]}?${p.toString()}\n`;
  const res = await fetch(`${endpoints[kind]}?${p.toString()}`, {method:'POST'});
  const payload = await res.json();
  $('runOutput').textContent += JSON.stringify(payload, null, 2).slice(0, 8000);
  $('runStatus').textContent = payload.exit_code === 0 ? 'Done' : 'Check output';
  await refresh();
}

['retrievalDbFilter','retrievalEmbeddingFilter'].forEach(id => $(id)?.addEventListener('input', () => renderRetrieval(state.operational?.retrieval_smokes || [])));
$('refreshBtn').addEventListener('click', () => refresh().catch(e => { $('statusPill').textContent = 'Error'; console.error(e); }));
$('runIngestBtn')?.addEventListener('click', () => runAction('ingest').catch(e => { $('runStatus').textContent='Error'; $('runOutput').textContent=String(e); }));
$('runSmokeBtn')?.addEventListener('click', () => runAction('smoke').catch(e => { $('runStatus').textContent='Error'; $('runOutput').textContent=String(e); }));
$('runEvalBtn')?.addEventListener('click', () => runAction('eval').catch(e => { $('runStatus').textContent='Error'; $('runOutput').textContent=String(e); }));
loadOptions().then(refresh).catch(e => { $('statusPill').textContent = 'Error'; document.body.insertAdjacentHTML('beforeend', `<pre class="fatal">${esc(e.message)}</pre>`); });
