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

function buildEvidenceRows(retrievalSmokes, rerankerSmokes) {
  const base = (retrievalSmokes || []).map(r => ({...r, reranker: 'none', evidence_type: 'retrieval'}));
  const reranked = (rerankerSmokes || []).map(r => ({...r, evidence_type: 'reranked'}));
  return [...reranked, ...base].map(r => ({
    ...r,
    reranker: r.reranker || 'none',
    top_pdf: ((r.hits || [])[0] || {}).pdf_name || '—',
    evidence_snippet: bestEvidenceSnippet(r),
  }));
}

function bestEvidenceSnippet(row) {
  const hit = (row.hits || [])[0] || {};
  const text = hit.paragraph || hit.text || hit.chunk || '';
  return String(text).replace(/\s+/g, ' ').trim();
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
  fillSelect('retrievalChunkerFilter', uniq(evidence, 'sheet'), 'All chunkers');
  fillSelect('retrievalEmbeddingFilter', uniq(evidence, 'embedding'), 'All embeddings');
  fillSelect('retrievalDbFilter', uniq(evidence, 'store'), 'All DBs');
  fillSelect('retrievalRerankerFilter', uniq(evidence, 'reranker'), 'All rerankers');
  const rows = filteredRetrieval(evidence).slice(0, 80);
  table($('retrievalTable'), rows, [
    {key:'query', label:'Query', render:r=>esc((r.query || '').slice(0, 58))},
    {key:'sheet', label:'Chunker'},
    {key:'embedding', label:'Embedding'},
    {key:'store', label:'DB'},
    {key:'reranker', label:'Reranker'},
    {key:'top_pdf', label:'Top PDF', render:r=>esc((r.top_pdf || '').slice(0, 58))},
    {key:'evidence_snippet', label:'Exact retrieved evidence', render:r=>`<details class="snippet"><summary>${esc((r.evidence_snippet || 'Open evidence').slice(0, 90))}</summary><p>${esc(r.evidence_snippet || 'No paragraph returned in artifact')}</p></details>`},
    {key:'retrieved_count', label:'Hits'},
    {key:'evidence_type', label:'Type'},
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
    {key:'store', label:'DB'},
    {key:'reranker', label:'Reranker'},
    {key:'verdict', label:'Verdict', render:r=>`<span class="risk ${esc(r.risk || '')}">${esc(r.verdict || '')}</span>`},
    {key:'support_score', label:'Support'},
    {key:'risk', label:'Risk'},
    {key:'reason', label:'Reason', render:r=>esc((r.reason || '').slice(0, 180))},
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
    {key:'store', label:'DB'},
    {key:'reranker', label:'Reranker', render:r=>esc(r.reranker || 'none')},
    {key:'evaluated_queries', label:'Queries'},
    {key:'recall_at_1', label:'R@1'},
    {key:'recall_at_3', label:'R@3'},
    {key:'recall_at_5', label:'R@5'},
    {key:'recall_at_10', label:'R@10'},
    {key:'mrr', label:'MRR'},
    {key:'precision_at_5', label:'P@5'},
    {key:'ndcg_at_5', label:'nDCG@5'},
    {key:'avg_first_relevant_rank', label:'Avg rank'},
    {key:'no_hit_queries', label:'No hit'},
    {key:'avg_latency_seconds', label:'Avg sec/query'},
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
    {step:'Best vector database', value:groupWinner(rows,'store').name, note:`Avg latency ${fmt(groupWinner(rows,'store').latency,3)}s · score ${fmt(groupWinner(rows,'store').score,3)}`, accent:'rose'},
    {step:'Best reranker', value:groupWinner(rows,'reranker').name || 'none', note:`Average score ${fmt(groupWinner(rows,'reranker').score,3)} across ${groupWinner(rows,'reranker').configs} configs`, accent:'rank'},
    {step:'Fastest evaluated config', value:`${fastest.store}`, note:`${fastest.sheet} · ${fastest.embedding}${fastest.reranker && fastest.reranker !== 'none' ? ' · ' + fastest.reranker : ''} · ${fmt(fastest.avg_latency_seconds,3)}s`, accent:'speed'},
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
      ${metricBar('Latency', maxLatency - num(r.avg_latency_seconds) + 0.0001, maxLatency, `${fmt(r.avg_latency_seconds,3)}s avg`)}
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
    ['Vector DB', groupWinner(rows, 'store')],
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
  const evaluation = op.evaluation || {};
  const evalRows = [...(evaluation.summary || []), ...(evaluation.reranked?.summary || [])].map(r => ({...r, reranker: r.reranker || 'none'})).sort((a,b)=>metricScore(b)-metricScore(a) || num(b.recall_at_5)-num(a.recall_at_5));
  const best = evalRows[0] || null;
  const fastest = [...evalRows].sort((a,b) => num(a.avg_latency_seconds) - num(b.avg_latency_seconds))[0] || null;

  $('modeLabel').textContent = 'Live artifacts';
  $('latestRun').textContent = ingestion.latest_run_id || snapshot?.ingestion?.latest_run_id || '—';
  $('snapshotAt').textContent = snapshot.created_at ? new Date(snapshot.created_at).toLocaleString() : '—';
  $('groundTruthStatus').textContent = evaluation?.report?.groundtruth_rows ? `${evaluation.report.groundtruth_rows} rows evaluated` : ((evaluation?.groundtruth_files || []).length ? 'Loaded, not evaluated' : 'Pending');
  $('pdfStatus').textContent = `${op.extracted_pdf_count || 0}/${op.known_pdf_count || 0}`;
  $('pdfNote').textContent = `${op.failed_pdf_count || 0} PDFs need OCR decision`;
  $('comboStatus').textContent = String(ingestion.combo_count ?? latest.length ?? 0);
  $('retrievalStatus').textContent = String(op.retrieval_smoke_total || retrieval.length || 0);
  $('retrievalNote').textContent = `${retrieval.length} retrieval · ${rerank.length} reranked artifacts`;
  $('bestR5Status').textContent = best ? `${(num(best.recall_at_5)*100).toFixed(1)}%` : '—';
  $('bestConfigNote').textContent = best ? pipelineLabel(best) : 'ground-truth winner';
  $('bestLatencyStatus').textContent = fastest ? `${fmt(fastest.avg_latency_seconds, 3)}s` : '—';
  $('bestLatencyNote').textContent = fastest ? `${pipelineLabel(fastest)} · avg/query` : 'fastest evaluated config';
  $('embeddingStatus').textContent = embeddings.length ? embeddings.join(' + ') : '—';
  $('dbStatus').textContent = stores.length ? stores.join(' + ') : '—';
  $('ingestionHint').textContent = `${latest.length} latest successful rows`;
  $('rerankHint').textContent = `${rerank.length} artifacts`;
  $('artifactHint').textContent = `${state.files.length} files`;
  $('healthHint').textContent = `${health.filter(h => h.ok).length}/${health.length || 0} healthy`;
  const pdfAudit = op.pdf_audit || {};
  $('pdfAuditHint') && ($('pdfAuditHint').textContent = pdfAudit.total ? `${pdfAudit.needs_ocr_count || 0}/${pdfAudit.total} need OCR/review` : 'No audit file yet');

  renderCoverage(rows);
  renderServices(health);
  renderRetrieval(retrieval, rerank);
  renderEvaluation(op.evaluation);
  renderPipelineComparison(op.evaluation);
  renderHallucination(op.hallucination || {});

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
  table($('pdfAuditTable'), (pdfAudit.rows || []).slice(0, 80), [
    {key:'pdf_name', label:'PDF', render:r=>esc((r.pdf_name || '').slice(0, 72))},
    {key:'status', label:'Status'},
    {key:'parser_method', label:'Parser'},
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
  fillRunSelect('runSheet', benchmarkOptions.chunkers || []);
  fillRunSelect('runEmbedding', benchmarkOptions.embeddings || []);
  fillRunSelect('runStore', benchmarkOptions.vector_stores || []);
  const mainReranker = $('runRerankerMain');
  if (mainReranker) {
    mainReranker.innerHTML = '<option value="qwen3_4b_rerank">qwen3_4b_rerank</option><option value="bge-reranker-base">bge-reranker-base</option><option value="all">Both OSS rerankers</option>';
  }
}

function selectedParams() {
  const p = new URLSearchParams();
  if ($('runSheet')?.value) p.append('sheet', $('runSheet').value);
  if ($('runEmbedding')?.value) p.append('embedding', $('runEmbedding').value);
  if ($('runStore')?.value) p.append('store', $('runStore').value);
  const limit = $('runLimit')?.value || '0';
  p.set('limit', limit);
  p.set('top_k', $('runTopK')?.value || '10');
  p.set('max_runs', '1');
  return p;
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
    p.set('max_runs', '36');
  }
  if (kind === 'rerank') {
    p.set('top_k', $('runTopK')?.value || '10');
    p.set('limit', $('rerankerLimit')?.value || '100');
    const mainChoice = $('runRerankerMain')?.value || 'qwen3_4b_rerank';
    let selected = [];
    if (mainChoice === 'all') {
      selected = ['bge-reranker-base', 'qwen3_4b_rerank'];
    } else if (mainChoice) {
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
$('runParseMineruBtn')?.addEventListener('click', () => runAction('parseMineru').catch(e => { $('runStatus').textContent='Error'; $('runOutput').textContent=String(e); }));
$('runIngestBtn')?.addEventListener('click', () => runAction('ingest').catch(e => { $('runStatus').textContent='Error'; $('runOutput').textContent=String(e); }));
$('runSmokeBtn')?.addEventListener('click', () => runAction('smoke').catch(e => { $('runStatus').textContent='Error'; $('runOutput').textContent=String(e); }));
$('runFullGtBtn')?.addEventListener('click', () => runAction('full').catch(e => { $('runStatus').textContent='Error'; $('runOutput').textContent=String(e); }));
$('runRerankerBtn')?.addEventListener('click', () => runAction('rerank').catch(e => { $('runStatus').textContent='Error'; $('runOutput').textContent=String(e); }));
$('runRerankedEvalBtn')?.addEventListener('click', () => runAction('rerankEval').catch(e => { $('runStatus').textContent='Error'; $('runOutput').textContent=String(e); }));
$('runEvalBtn')?.addEventListener('click', () => runAction('eval').catch(e => { $('runStatus').textContent='Error'; $('runOutput').textContent=String(e); }));
$('runHallucinationBtn')?.addEventListener('click', () => runAction('hallucination').catch(e => { $('runStatus').textContent='Error'; $('runOutput').textContent=String(e); }));
$('runHallucinationLlmBtn')?.addEventListener('click', () => runAction('hallucinationLlm').catch(e => { $('runStatus').textContent='Error'; $('runOutput').textContent=String(e); }));
document.querySelectorAll('.tab-btn').forEach(btn => btn.addEventListener('click', () => showPage(btn.dataset.page || 'overview')));
showPage((location.hash || '#overview').slice(1));
$('uploadForm')?.addEventListener('submit', e => uploadDataset(e).catch(err => { $('uploadStatus').textContent='Error'; $('uploadOutput').textContent=String(err); }));
$('openTestOptionsBtn')?.addEventListener('click', () => $('testOptionsDialog')?.showModal());
loadOptions().then(refresh).catch(e => { $('statusPill').textContent = 'Error'; document.body.insertAdjacentHTML('beforeend', `<pre class="fatal">${esc(e.message)}</pre>`); });
