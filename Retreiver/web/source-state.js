(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  root.DashboardSourceState = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';

  const SOURCE_KEYS = Object.freeze([
    'dataset_id',
    'groundtruth_id',
    'result_set_id',
    'source_type',
    'scoring_mode',
  ]);

  function canonicalRerankerName(value) {
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

  function finite(value) {
    if (value === '' || value === null || value === undefined || typeof value === 'boolean') return null;
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }

  function metricScore(row) {
    const explicit = finite(row?.winner_score);
    if (explicit !== null && explicit !== 0) return explicit;
    const recall = finite(row?.recall_at_5) || 0;
    const mrr = finite(row?.mrr) || 0;
    const ndcg = finite(row?.ndcg_at_5) || 0;
    const latency = finite(row?.avg_latency_seconds) || 0;
    return (0.45 * recall) + (0.30 * mrr) + (0.20 * ndcg) + (0.05 * (1 / (1 + latency)));
  }

  function canonicalMetricRow(row, extra) {
    const normalized = Object.assign({}, row || {}, extra || {});
    normalized.reranker = canonicalRerankerName(normalized.reranker || normalized.reranking_model || 'none');
    return normalized;
  }

  function metricRowKey(row) {
    return `${row?.sheet || ''}|${row?.embedding || ''}|${row?.store || ''}|${canonicalRerankerName(row?.reranker || row?.reranking_model || 'none')}`;
  }

  function isCompleted(row) {
    const status = String(row?.status || 'completed').toLowerCase();
    return !['failed', 'error', 'blocked', 'pending', 'running', 'cancelled'].includes(status);
  }

  function isNvidiaLane(row) {
    return ['source_type', 'result_set', 'result_set_id', 'dataset_id', 'lane'].some(key =>
      /(^|[^a-z])nvidia([^a-z]|$)/i.test(String(row?.[key] || ''))
    );
  }

  function normalizeOfficialRows(rows, reranked) {
    const byId = new Map();
    (Array.isArray(rows) ? rows : []).forEach(raw => {
      const row = canonicalMetricRow(raw, {status: raw?.status || 'completed'});
      if (!isCompleted(row) || isNvidiaLane(row)) return;
      if (reranked ? row.reranker === 'none' : row.reranker !== 'none') return;
      const comboId = metricRowKey(row);
      if (!byId.has(comboId)) {
        byId.set(comboId, Object.assign({}, row, {
          combo_id: comboId,
          winner_score: metricScore(row),
        }));
      }
    });
    return [...byId.values()].sort((left, right) =>
      metricScore(right) - metricScore(left) || left.combo_id.localeCompare(right.combo_id)
    );
  }

  function officialResultPayload(evaluation, descriptor, evidenceRows) {
    const referenceRows = evaluation?.benchmark_reference?.summary || [];
    const rows = normalizeOfficialRows(referenceRows, true);
    const evidenceCounts = new Map();
    (Array.isArray(evidenceRows) ? evidenceRows : []).forEach(raw => {
      const key = metricRowKey(canonicalMetricRow(raw));
      evidenceCounts.set(key, (evidenceCounts.get(key) || 0) + 1);
    });
    const decorated = rows.map(row => Object.assign({}, row, {
      evidence_count: evidenceCounts.has(row.combo_id) ? evidenceCounts.get(row.combo_id) : null,
    }));
    return Object.freeze({
      source_type: 'official',
      result_set: 'official',
      scoring_mode: 'retrieval_labels',
      metric_k: 5,
      configured: descriptor?.configured ?? 180,
      evaluated: descriptor?.evaluated ?? decorated.length,
      rows: Object.freeze(decorated),
    });
  }

  function baselineResultPayload(evaluation, evidenceRows) {
    const rows = normalizeOfficialRows([
      ...(evaluation?.summary || []),
      ...(evaluation?.reranked?.summary || []),
    ], false);
    const evidenceCounts = new Map();
    (Array.isArray(evidenceRows) ? evidenceRows : []).forEach(raw => {
      const key = metricRowKey(canonicalMetricRow(raw));
      evidenceCounts.set(key, (evidenceCounts.get(key) || 0) + 1);
    });
    const decorated = rows.map(row => Object.assign({}, row, {
      evidence_count: evidenceCounts.has(row.combo_id) ? evidenceCounts.get(row.combo_id) : null,
    }));
    return Object.freeze({
      source_type: 'official',
      result_set: 'baseline',
      scoring_mode: 'retrieval_labels',
      metric_k: 5,
      configured: decorated.length,
      evaluated: decorated.length,
      rows: Object.freeze(decorated),
    });
  }

  function normalizeResultPayload(payload) {
    const normalized = Object.assign({}, payload || {});
    normalized.source_type = normalized.source_type || 'uploaded_project';
    normalized.scoring_mode = normalized.scoring_mode || 'evidence_only';
    normalized.rows = Object.freeze((Array.isArray(normalized.rows) ? normalized.rows : []).map(row =>
      Object.freeze(Object.assign({}, row, {status: row?.status || 'completed'}))
    ));
    return Object.freeze(normalized);
  }

  function normalizeActiveSource(source) {
    const input = source || {};
    return Object.freeze({
      dataset_id: String(input.dataset_id || 'official'),
      groundtruth_id: String(input.groundtruth_id || 'groundtruth:official'),
      result_set_id: String(input.result_set_id || 'official'),
      source_type: String(input.source_type || 'official'),
      scoring_mode: String(input.scoring_mode || 'retrieval_labels'),
    });
  }

  function sourceIdentity(source) {
    const normalized = normalizeActiveSource(source);
    return SOURCE_KEYS.map(key => normalized[key]).join('|');
  }

  function createSelection(source, generation) {
    return Object.freeze(Object.assign({}, normalizeActiveSource(source), {
      generation: Number.isInteger(generation) && generation >= 0 ? generation : 0,
    }));
  }

  function advanceSelection(current, patch) {
    const previous = createSelection(current, current?.generation);
    const nextSource = normalizeActiveSource(Object.assign({}, previous, patch || {}));
    const changed = sourceIdentity(previous) !== sourceIdentity(nextSource);
    return createSelection(nextSource, previous.generation + (changed ? 1 : 0));
  }

  function matchesSelection(expected, candidate) {
    return Number(expected?.generation) === Number(candidate?.generation) && sourceIdentity(expected) === sourceIdentity(candidate);
  }

  return Object.freeze({
    SOURCE_KEYS,
    canonicalRerankerName,
    canonicalMetricRow,
    metricRowKey,
    metricScore,
    officialResultPayload,
    baselineResultPayload,
    normalizeResultPayload,
    normalizeActiveSource,
    sourceIdentity,
    createSelection,
    advanceSelection,
    matchesSelection,
  });
});
