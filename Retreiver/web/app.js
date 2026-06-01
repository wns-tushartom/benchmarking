const $ = (id) => document.getElementById(id);
let state = { operational: {}, files: [], options: {} };

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
  const chunkers = uniq(latest, 'sheet');
  const embeddings = uniq(latest, 'embedding');
  const stores = uniq(latest, 'store');
  const el = $('coverageMatrix');
  $('coverageHint').textContent = `${latest.length} live combos`;
  if (!latest.length) {
    el.innerHTML = '<div class="empty-state">No successful DB ingestion rows found.</div>';
    return;
  }
  const byKey = new Map(latest.map(r => [`${r.sheet}|${r.embedding}|${r.store}`, r]));
  el.style.setProperty('--cols', String(Math.max(1, stores.length + 1)));
  let html = `<div class="matrix-head">Chunker / embedding</div>${stores.map(s => `<div class="matrix-head">${esc(s)}</div>`).join('')}`;
  chunkers.forEach(ch => {
    embeddings.forEach(em => {
      const hasAny = stores.some(st => byKey.has(`${ch}|${em}|${st}`));
      if (!hasAny) return;
      html += `<div class="matrix-label"><strong>${esc(ch)}</strong><span>${esc(em)}</span></div>`;
      stores.forEach(st => {
        const r = byKey.get(`${ch}|${em}|${st}`);
        html += r ? `<div class="matrix-cell ok"><strong>${esc(r.search_hits || '5')} hits</strong><span>${fmt(r.total_store_seconds, 2)}s</span></div>` : '<div class="matrix-cell missing">—</div>';
      });
    });
  });
  el.innerHTML = html;
}

function filteredRetrieval(smokes) {
  const db = $('retrievalDbFilter')?.value || 'all';
  const emb = $('retrievalEmbeddingFilter')?.value || 'all';
  return smokes.filter(r => (db === 'all' || r.store === db) && (emb === 'all' || r.embedding === emb));
}

function renderRetrieval(smokes) {
  fillSelect('retrievalDbFilter', uniq(smokes, 'store'), 'All DBs');
  fillSelect('retrievalEmbeddingFilter', uniq(smokes, 'embedding'), 'All embeddings');
  const rows = filteredRetrieval(smokes).slice(0, 18);
  const el = $('retrievalCards');
  if (!rows.length) {
    el.innerHTML = '<div class="empty-state">No retrieval smoke JSON yet. Run run_retrieval_smoke_from_vm_dbs.py on the VM.</div>';
    return;
  }
  el.innerHTML = rows.map(r => {
    const top = (r.hits || [])[0] || {};
    return `<article class="retrieval-card">
      <div class="card-top"><span>${esc(r.store)}</span><code>${fmt(r.retrieval_seconds, 3)}s</code></div>
      <h3>${esc(r.query || 'query')}</h3>
      <p>${esc(top.pdf_name || 'No top PDF captured')}</p>
      <div class="meta-row"><span>${esc(r.sheet)}</span><span>${esc(r.embedding)}</span><span>${esc(r.retrieved_count || 0)} hits</span></div>
    </article>`;
  }).join('');
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

  $('modeLabel').textContent = state.options?.mode || 'vm_remote_required';
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

['retrievalDbFilter','retrievalEmbeddingFilter'].forEach(id => $(id)?.addEventListener('input', () => renderRetrieval(state.operational?.retrieval_smokes || [])));
$('refreshBtn').addEventListener('click', () => refresh().catch(e => { $('statusPill').textContent = 'Error'; console.error(e); }));
refresh().catch(e => { $('statusPill').textContent = 'Error'; document.body.insertAdjacentHTML('beforeend', `<pre class="fatal">${esc(e.message)}</pre>`); });
