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
    if (explicit !== null) return explicit;
    const recall = finite(row?.recall_at_5);
    const mrr = finite(row?.mrr);
    const ndcg = finite(row?.ndcg_at_5);
    const latency = finite(row?.avg_latency_seconds);
    if ([recall, mrr, ndcg, latency].some(value => value === null)) return null;
    return (0.45 * recall) + (0.30 * mrr) + (0.20 * ndcg) + (0.05 * (1 / (1 + latency)));
  }

  const OFFICIAL_METRICS = Object.freeze([
    'recall_at_1', 'recall_at_3', 'recall_at_5', 'recall_at_10',
    'mrr', 'precision_at_5', 'ndcg_at_5', 'avg_first_relevant_rank',
    'no_hit_queries', 'avg_latency_seconds',
  ]);

  function officialMetricCompleteness(row) {
    const missing = OFFICIAL_METRICS.filter(field => finite(row?.[field]) === null);
    const queryCount = finite(row?.evaluated_queries);
    if (queryCount === null || queryCount <= 0) missing.unshift('evaluated_queries');
    return Object.freeze({complete: missing.length === 0, missing_metrics: Object.freeze(missing)});
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
        const score = metricScore(row);
        byId.set(comboId, Object.assign({}, row, {
          combo_id: comboId,
          winner_score: score,
        }));
      }
    });
    return [...byId.values()].sort((left, right) =>
      (metricScore(right) ?? -Infinity) - (metricScore(left) ?? -Infinity) || left.combo_id.localeCompare(right.combo_id)
    );
  }

  function officialResultPayload(evaluation, descriptor, evidenceRows) {
    const referenceRows = evaluation?.benchmark_reference?.summary || [];
    const discoveredRows = normalizeOfficialRows(referenceRows, true);
    const evidenceCounts = new Map();
    (Array.isArray(evidenceRows) ? evidenceRows : []).forEach(raw => {
      const key = metricRowKey(canonicalMetricRow(raw));
      evidenceCounts.set(key, (evidenceCounts.get(key) || 0) + 1);
    });
    const rows = [];
    const incompleteRows = [];
    discoveredRows.forEach(row => {
      const completeness = officialMetricCompleteness(row);
      const next = Object.freeze(Object.assign({}, row, completeness, {
        evidence_count: evidenceCounts.has(row.combo_id) ? evidenceCounts.get(row.combo_id) : null,
      }));
      (completeness.complete ? rows : incompleteRows).push(next);
    });
    return Object.freeze({
      source_type: 'official',
      result_set: 'official',
      scoring_mode: 'retrieval_labels',
      metric_k: 5,
      configured: descriptor?.configured ?? 180,
      evaluated: rows.length,
      reported_evaluated: descriptor?.evaluated ?? discoveredRows.length,
      discovered: discoveredRows.length,
      matrix_complete: rows.length >= (descriptor?.configured ?? 180),
      stability_status: rows.length >= (descriptor?.configured ?? 180) ? 'complete' : 'partial',
      rows: Object.freeze(rows),
      incomplete_rows: Object.freeze(incompleteRows),
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
    officialMetricCompleteness,
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
