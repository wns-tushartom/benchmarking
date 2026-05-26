const $ = (id) => document.getElementById(id);
let state = {summary: [], chunking: [], files: [], modular: {summary: [], analysis: {}, manifest: {}}, options: {}};

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  const text = await res.text();
  let data;
  try { data = text ? JSON.parse(text) : {}; } catch { data = {error: text}; }
  if (!res.ok) throw new Error(data.error || text || `HTTP ${res.status}`);
  return data;
}
const num = (v) => Number.parseFloat(v || 0) || 0;
const uniq = (rows, key) => [...new Set(rows.map(r => r[key]).filter(Boolean))].sort();
const color = (a = 1) => `rgba(125,211,252,${a})`;
const mint = (a = 1) => `rgba(167,243,208,${a})`;
const amber = (a = 1) => `rgba(251,191,36,${a})`;
function escapeHtml(s) { return String(s ?? '').replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c])); }
function fillSelect(id, values, keep = true) {
  const el = $(id); if (!el) return;
  const cur = keep ? (el.value || 'all') : 'all';
  el.innerHTML = '<option value="all">All</option>' + values.map(v => `<option value="${escapeHtml(v)}">${escapeHtml(v)}</option>`).join('');
  el.value = values.includes(cur) ? cur : 'all';
}
function filteredRows() {
  return state.summary.filter(r =>
    ($('dbFilter').value === 'all' || r.vector_database === $('dbFilter').value) &&
    ($('embeddingFilter').value === 'all' || r.embedding_model === $('embeddingFilter').value) &&
    ($('chunkFilter').value === 'all' || r.chunking_method === $('chunkFilter').value) &&
    ($('rerankFilter').value === 'all' || r.reranking_model === $('rerankFilter').value));
}
function rankRows(rows) {
  const metric = $('rankMetric').value;
  const asc = ['avg_query_latency_ms', 'chunk_count'].includes(metric);
  return [...rows].sort((a, b) => asc ? num(a[metric]) - num(b[metric]) : num(b[metric]) - num(a[metric]));
}
function table(el, rows, cols) {
  if (!rows || rows.length === 0) { el.innerHTML = '<tr><td>No data yet</td></tr>'; return; }
  el.innerHTML = '<thead><tr>' + cols.map(c => `<th>${c.label}</th>`).join('') + '</tr></thead><tbody>' +
    rows.map((r, i) => '<tr>' + cols.map(c => `<td>${c.render ? c.render(r, i) : escapeHtml(r[c.key] ?? '')}</td>`).join('') + '</tr>').join('') + '</tbody>';
}
function setupCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1, rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.floor(rect.width * dpr)); canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  const ctx = canvas.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); return {ctx, w: rect.width, h: rect.height};
}
function clearChart(ctx, w, h) {
  ctx.clearRect(0, 0, w, h); ctx.fillStyle = 'rgba(5,9,19,.15)'; ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = 'rgba(159,177,200,.16)'; for (let i = 1; i < 5; i++) { ctx.beginPath(); ctx.moveTo(44, i*h/5); ctx.lineTo(w-14, i*h/5); ctx.stroke(); }
}
function drawScatterLike(canvasId, rows, xKey, yKey, labelKey = 'benchmark_run_id') {
  const canvas = $(canvasId); if (!canvas) return;
  const {ctx, w, h} = setupCanvas(canvas); clearChart(ctx, w, h); if (!rows.length) return;
  const pad = {l: 52, r: 18, t: 18, b: 38}; const xs = rows.map(r => num(r[xKey])), ys = rows.map(r => num(r[yKey]));
  const maxX = Math.max(...xs, 1), minX = Math.min(...xs, 0), maxY = Math.max(...ys, 1), minY = Math.min(...ys, 0);
  ctx.fillStyle = '#9fb1c8'; ctx.font = '12px system-ui'; ctx.fillText('Latency ms →', w-110, h-12); ctx.save(); ctx.translate(14, 120); ctx.rotate(-Math.PI/2); ctx.fillText('Recall@5 →', 0, 0); ctx.restore();
  rows.forEach((r, i) => {
    const x = pad.l + ((num(r[xKey])-minX)/(maxX-minX || 1))*(w-pad.l-pad.r);
    const y = h-pad.b - ((num(r[yKey])-minY)/(maxY-minY || 1))*(h-pad.t-pad.b);
    ctx.beginPath(); ctx.fillStyle = i === 0 ? mint(.95) : (r.vector_database === 'PGVector' || r.vector_store === 'PGVector') ? mint(.72) : (r.vector_database === 'Weaviate' || r.vector_store === 'Weaviate') ? amber(.72) : color(.78);
    ctx.arc(x, y, 5 + Math.min(8, Math.sqrt(num(r.chunk_count || 100))/8), 0, Math.PI*2); ctx.fill();
    if (i < 5) { ctx.fillStyle = '#edf5ff'; ctx.fillText(r[labelKey] || String(i+1), x+8, y-6); }
  });
}
function drawBar(rows) {
  const canvas = $('barChart'); const {ctx, w, h} = setupCanvas(canvas); clearChart(ctx, w, h);
  const top = rankRows(rows).slice(0, Math.max(3, Math.min(20, Number($('topN').value) || 15))); if (!top.length) return;
  const metric = $('rankMetric').value, maxVal = Math.max(...top.map(r => num(r[metric])), 1); const padL = 120, padR = 70, barH = Math.max(10, Math.min(22, (h-50)/top.length-5)); ctx.font = '11px system-ui';
  top.forEach((r, i) => { const y = 22 + i*(barH+5), v = num(r[metric]), bw = (w-padL-padR)*(v/(maxVal || 1)); ctx.fillStyle = 'rgba(125,211,252,.82)'; ctx.fillRect(padL, y, bw, barH); ctx.fillStyle = '#9fb1c8'; ctx.fillText(`${i+1}. ${r.benchmark_run_id}`, 10, y+barH-3); ctx.fillStyle = '#edf5ff'; ctx.fillText(v.toFixed(metric.includes('recall') ? 3 : 2), Math.min(w-58, padL+bw+8), y+barH-3); });
}
function aggregate(rows, key) {
  const m = new Map(); rows.forEach(r => { const k = r[key] || 'unknown'; if (!m.has(k)) m.set(k, []); m.get(k).push(r); });
  return [...m.entries()].map(([k, rs]) => ({key: k, recall_at_5: rs.reduce((a, r) => a+num(r.recall_at_5), 0)/rs.length, recall_at_10: rs.reduce((a, r) => a+num(r.recall_at_10), 0)/rs.length, avg_query_latency_ms: rs.reduce((a, r) => a+num(r.avg_query_latency_ms), 0)/rs.length, chunk_count: rs.reduce((a, r) => a+num(r.chunk_count), 0)/rs.length})).sort((a, b) => b.recall_at_5-a.recall_at_5);
}
function drawChunkChart(rows) {
  const canvas = $('chunkChart'); const {ctx, w, h} = setupCanvas(canvas); clearChart(ctx, w, h); const grouped = aggregate(rows, 'chunking_method'); if (!grouped.length) return;
  const maxChunks = Math.max(...grouped.map(g => g.chunk_count), 1), padL = 150, padR = 18, rowH = Math.max(24, (h-45)/grouped.length); ctx.font = '12px system-ui';
  grouped.forEach((g, i) => { const y = 22 + i*rowH; ctx.fillStyle = '#9fb1c8'; ctx.fillText(g.key, 8, y+13); ctx.fillStyle = 'rgba(167,243,208,.82)'; ctx.fillRect(padL, y, (w-padL-padR)*g.recall_at_5, 10); ctx.fillStyle = 'rgba(125,211,252,.36)'; ctx.fillRect(padL, y+13, (w-padL-padR)*(g.chunk_count/maxChunks), 10); ctx.fillStyle = '#edf5ff'; ctx.fillText(`R@5 ${g.recall_at_5.toFixed(3)} · ${Math.round(g.chunk_count)} chunks`, padL+6, y+37); });
}
function renderHeatmap(rows) {
  const chunks = uniq(rows, 'chunking_method'), dbs = uniq(rows, 'vector_database'), el = $('heatmap'); el.style.setProperty('--cols', Math.max(1, dbs.length));
  if (!rows.length) { el.innerHTML = '<p>No data yet</p>'; return; }
  let html = `<div class="heat-row"><div class="heat-label">Chunking</div>${dbs.map(d => `<div class="heat-label">${escapeHtml(d)}</div>`).join('')}</div>`;
  chunks.forEach(ch => { html += `<div class="heat-row"><div class="heat-label">${escapeHtml(ch)}</div>`; dbs.forEach(db => { const cr = rows.filter(r => r.chunking_method === ch && r.vector_database === db); const avg = cr.length ? cr.reduce((a, r) => a+num(r.recall_at_5), 0)/cr.length : 0; html += `<div class="heat-cell" style="background:linear-gradient(135deg, rgba(167,243,208,${0.18+avg*.75}), rgba(125,211,252,${0.12+avg*.55}))">${avg.toFixed(3)}</div>`; }); html += '</div>'; }); el.innerHTML = html;
}
function render() {
  const rows = filteredRows(), ranked = rankRows(rows), topN = Math.max(3, Math.min(50, Number($('topN').value) || 15)); const modularRows = state.modular.summary || [], bestMod = state.modular.analysis?.best_config || modularRows[0] || {};
  const selectableCount = state.options.matrix_count || 0;
  $('totalRuns').textContent = selectableCount || state.summary.length || '0'; $('queryCount').textContent = state.summary[0]?.query_count || state.modular.manifest?.query_count || '—'; $('bestRecall').textContent = state.summary.length ? Math.max(...state.summary.map(r => num(r.recall_at_5))).toFixed(3) : '—';
  const lat = state.summary.map(r => num(r.avg_query_latency_ms)).filter(Boolean); $('bestLatency').textContent = lat.length ? `${Math.min(...lat).toFixed(2)} ms` : '—'; $('modularRuns').textContent = modularRows.length || '0'; $('modularBest').textContent = bestMod.recall_at_5 ? num(bestMod.recall_at_5).toFixed(3) : '—';
  $('experimentName').textContent = state.modular.manifest?.experiment?.name || '—'; $('configHash').textContent = state.modular.manifest?.config_hash || '—'; $('datasetHash').textContent = state.modular.manifest?.dataset_hash || '—'; $('bestConfigLabel').textContent = bestMod.chunker ? `${bestMod.chunker} + ${bestMod.embedding} + ${bestMod.vector_store}` : '—';
  $('whyBest').innerHTML = (state.modular.analysis?.why_best || []).map(x => `<li>${escapeHtml(x)}</li>`).join('') || '<li>No modular analysis yet</li>'; $('rowCount').textContent = `${ranked.length} matching configs`; $('topHint').textContent = `ranked by ${$('rankMetric').value}`;
  table($('summaryTable'), ranked.slice(0, topN), [{key:'benchmark_run_id',label:'Run',render:r=>`<span class="badge">${escapeHtml(r.benchmark_run_id)}</span>`},{key:'chunking_method',label:'Chunking'},{key:'embedding_model',label:'Embedding'},{key:'vector_database',label:'DB'},{key:'reranking_model',label:'Rerank'},{key:'recall_at_5',label:'R@5'},{key:'recall_at_10',label:'R@10'},{key:'avg_query_latency_ms',label:'Avg ms'},{key:'chunk_count',label:'Chunks'}]);
  table($('chunkTable'), rankRows(state.chunking).slice(0, 12), [{key:'chunking_method',label:'Chunking'},{key:'recall_at_3',label:'R@3'},{key:'recall_at_5',label:'R@5'},{key:'recall_at_10',label:'R@10'},{key:'chunk_count',label:'Chunks'}]);
  table($('modularTable'), modularRows.slice(0, topN), [{key:'benchmark_run_id',label:'Run'},{key:'chunker',label:'Chunking'},{key:'embedding',label:'Embedding'},{key:'vector_store',label:'DB'},{key:'index_type',label:'Index'},{key:'retrieval_method',label:'Retrieval'},{key:'reranker',label:'Rerank'},{key:'recall_at_5',label:'R@5'},{key:'mrr',label:'MRR'},{key:'ndcg_at_10',label:'nDCG'},{key:'recall_at_5_ci_low',label:'CI low'},{key:'recall_at_5_ci_high',label:'CI high'}]);
  $('files').innerHTML = state.files.map(f => `<li><code>${escapeHtml(f)}</code></li>`).join(''); drawScatterLike('scatterChart', ranked, 'avg_query_latency_ms', 'recall_at_5'); drawBar(ranked); drawChunkChart(rows); renderHeatmap(rows); drawScatterLike('paretoChart', state.modular.analysis?.pareto || modularRows, 'avg_latency_ms', 'recall_at_5');
}
function selectionParams() {
  const p = new URLSearchParams(); p.set('limit', $('limit').value || '50'); p.set('max_runs', $('maxRuns').value || '0');
  const map = {selectedChunker:'chunker', selectedEmbedding:'embedding', selectedVectorStore:'vector_store', selectedIndexType:'index_type', selectedRetrievalMethod:'retrieval_method', selectedReranker:'reranker'};
  Object.entries(map).forEach(([id, key]) => { if ($(id).value !== 'all') p.set(key, $(id).value); });
  if ($('includeCandidates').checked) p.set('include_candidates', '1'); if ($('includeMilvus').checked) p.set('include_milvus', '1'); return p;
}
async function refresh() {
  $('statusPill').textContent = 'Loading'; const data = await api('/api/results'); state.summary = data.summary || []; state.chunking = data.chunking || []; state.modular = data.modular || {summary: [], analysis: {}, manifest: {}}; state.files = data.files || []; state.options = data.options || {};
  fillSelect('dbFilter', uniq(state.summary, 'vector_database')); fillSelect('embeddingFilter', uniq(state.summary, 'embedding_model')); fillSelect('chunkFilter', uniq(state.summary, 'chunking_method')); fillSelect('rerankFilter', uniq(state.summary, 'reranking_model'));
  fillSelect('selectedChunker', state.options.chunkers || []); fillSelect('selectedEmbedding', state.options.embeddings || []); fillSelect('selectedVectorStore', state.options.vector_stores || []); fillSelect('selectedIndexType', state.options.index_types || []); fillSelect('selectedRetrievalMethod', state.options.retrieval_methods || []); fillSelect('selectedReranker', state.options.rerankers || []);
  $('statusPill').textContent = `Ready · ${state.options.matrix_count || '—'} selectable combos · ${state.options.mode || 'local'}`;
  const realMode = state.options.mode === 'vm_real_adapters';
  $('runBtn').disabled = realMode;
  $('chunkBtn').disabled = realMode;
  if (realMode) {
    $('runBtn').title = 'Legacy dummy fallback matrix is disabled in VM real mode.';
    $('chunkBtn').title = 'Chunk recall fallback is disabled in VM real mode.';
  }
  if (!state.summary.length && !(state.modular.summary || []).length) {
    $('runLog').textContent = `Config loaded: ${state.options.matrix_count || 0} selectable combinations. Choose chunking, embedding, vector database, index, retrieval, reranker, then click Run selected combo.`;
  }
  render();
}
async function run(kind) {
  $('runLog').textContent = 'Starting...'; $('statusPill').textContent = 'Running'; ['runBtn','modularBtn','selectedBtn','chunkBtn','refreshBtn'].forEach(id => $(id).disabled = true);
  try { const data = await api(`/api/run/${kind}?${selectionParams()}`, {method: 'POST'}); $('runLog').textContent = data.output || JSON.stringify(data, null, 2); await refresh(); }
  finally {
    ['runBtn','modularBtn','selectedBtn','chunkBtn','refreshBtn'].forEach(id => $(id).disabled = false);
    if (state.options.mode === 'vm_real_adapters') { $('runBtn').disabled = true; $('chunkBtn').disabled = true; }
  }
}
['rankMetric','topN','dbFilter','embeddingFilter','chunkFilter','rerankFilter'].forEach(id => $(id).addEventListener('input', render));
$('runBtn').addEventListener('click', () => run('matrix').catch(e => {$('statusPill').textContent='Error'; $('runLog').textContent=e.message;}));
$('modularBtn').addEventListener('click', () => run('modular').catch(e => {$('statusPill').textContent='Error'; $('runLog').textContent=e.message;}));
$('selectedBtn').addEventListener('click', () => run('selected').catch(e => {$('statusPill').textContent='Error'; $('runLog').textContent=e.message;}));
$('chunkBtn').addEventListener('click', () => run('chunking').catch(e => {$('statusPill').textContent='Error'; $('runLog').textContent=e.message;}));
$('refreshBtn').addEventListener('click', () => refresh().catch(e => $('runLog').textContent = e.message));
window.addEventListener('resize', () => render());
refresh().catch(e => {$('statusPill').textContent='Error'; $('runLog').textContent=e.message;});
