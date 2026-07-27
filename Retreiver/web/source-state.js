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
    const invalid = OFFICIAL_METRICS.filter(field => finite(row?.[field]) === null);
    const reject = field => { if (!invalid.includes(field)) invalid.push(field); };
    const queryCount = finite(row?.evaluated_queries);
    if (queryCount !== 500 || !Number.isInteger(queryCount)) reject('evaluated_queries');

    [
      'recall_at_1', 'recall_at_3', 'recall_at_5', 'recall_at_10',
      'mrr', 'precision_at_5', 'ndcg_at_5',
    ].forEach(field => {
      const value = finite(row?.[field]);
      if (value !== null && (value < 0 || value > 1)) reject(field);
    });
    ['avg_first_relevant_rank', 'avg_latency_seconds'].forEach(field => {
      const value = finite(row?.[field]);
      if (value !== null && value < 0) reject(field);
    });
    const noHitQueries = finite(row?.no_hit_queries);
    if (
      noHitQueries !== null
      && (noHitQueries < 0 || !Number.isInteger(noHitQueries)
        || (queryCount !== null && queryCount > 0 && noHitQueries > queryCount))
    ) reject('no_hit_queries');
    return Object.freeze({complete: invalid.length === 0, missing_metrics: Object.freeze(invalid)});
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
    return String(row?.status || '').trim().toLowerCase() === 'completed';
  }

  function isNvidiaLane(row) {
    return ['source_type', 'result_set', 'result_set_id', 'dataset_id', 'lane'].some(key =>
      /(^|[^a-z])nvidia([^a-z]|$)/i.test(String(row?.[key] || ''))
    );
  }

  function isProjectLane(row) {
    return ['source_type', 'result_set', 'result_set_id', 'dataset_id', 'lane'].some(key =>
      /(^|[^a-z])(project|uploaded_project)([^a-z]|$)/i.test(String(row?.[key] || ''))
    );
  }

  function expectedKeySet(values) {
    if (values instanceof Set) return values;
    return Array.isArray(values) ? new Set(values.map(String)) : null;
  }

  function officialRowAdmission(raw, options) {
    const row = canonicalMetricRow(raw);
    const comboId = metricRowKey(row);
    const expectedKeys = expectedKeySet(options?.expectedKeys);
    const completeness = officialMetricCompleteness(row);
    const reranked = options?.reranked !== false;
    const reasons = [];
    if (isNvidiaLane(row)) reasons.push('nvidia_lane');
    if (isProjectLane(row)) reasons.push('project_lane');
    if (reranked ? row.reranker === 'none' : row.reranker !== 'none') reasons.push('wrong_reranker_lane');
    if (!isCompleted(row)) reasons.push('status_not_completed');
    if (row.official_provenance !== 'trusted') reasons.push('provenance_untrusted');
    if (!completeness.complete) reasons.push('metrics_incomplete');
    if (!expectedKeys || !expectedKeys.has(comboId)) reasons.push('not_in_expected_matrix');
    return Object.freeze({
      row: Object.freeze(row),
      combo_id: comboId,
      complete: completeness.complete,
      missing_metrics: completeness.missing_metrics,
      eligible: reasons.length === 0,
      reasons: Object.freeze(reasons),
    });
  }

  function admittedOfficialRows(rows, options, evidenceCounts, incompleteRows) {
    const admitted = new Map();
    (Array.isArray(rows) ? rows : []).forEach(raw => {
      const admission = officialRowAdmission(raw, options);
      const next = Object.freeze(Object.assign({}, admission.row, {
        complete: admission.complete,
        missing_metrics: admission.missing_metrics,
        combo_id: admission.combo_id,
        winner_score: admission.eligible ? metricScore(admission.row) : null,
        evidence_count: evidenceCounts?.has(admission.combo_id) ? evidenceCounts.get(admission.combo_id) : null,
        admission_reasons: admission.reasons,
      }));
      if (!admission.eligible) {
        incompleteRows.push(next);
        return;
      }
      const current = admitted.get(admission.combo_id);
      if (!current || (metricScore(next) ?? -Infinity) > (metricScore(current) ?? -Infinity)) {
        admitted.set(admission.combo_id, next);
      }
    });
    return [...admitted.values()].sort((left, right) =>
      (metricScore(right) ?? -Infinity) - (metricScore(left) ?? -Infinity) || left.combo_id.localeCompare(right.combo_id)
    );
  }

  function officialResultPayload(evaluation, descriptor, evidenceRows) {
    const referenceRows = Array.isArray(evaluation?.benchmark_reference?.summary)
      ? evaluation.benchmark_reference.summary : [];
    const diagnosticRows = Array.isArray(evaluation?.benchmark_reference?.diagnostics)
      ? evaluation.benchmark_reference.diagnostics : [];
    const expectedValues = descriptor?.expected_keys || evaluation?.benchmark_reference?.report?.expected_keys;
    const expectedKeys = expectedKeySet(expectedValues);
    const evidenceCounts = new Map();
    (Array.isArray(evidenceRows) ? evidenceRows : []).forEach(raw => {
      const key = metricRowKey(canonicalMetricRow(raw));
      evidenceCounts.set(key, (evidenceCounts.get(key) || 0) + 1);
    });
    const incompleteRows = [];
    const rows = admittedOfficialRows(
      referenceRows,
      {expectedKeys, reranked: true},
      evidenceCounts,
      incompleteRows,
    );
    diagnosticRows.forEach(raw => {
      const row = canonicalMetricRow(raw);
      const comboId = metricRowKey(row);
      const completeness = officialMetricCompleteness(row);
      incompleteRows.push(Object.freeze(Object.assign({}, row, completeness, {
        combo_id: comboId,
        winner_score: null,
        evidence_count: evidenceCounts.has(comboId) ? evidenceCounts.get(comboId) : null,
        diagnostic_only: true,
      })));
    });
    const actualKeys = new Set(rows.map(row => row.combo_id));
    const configured = descriptor?.configured ?? expectedKeys?.size ?? 180;
    const matrixComplete = expectedKeys
      ? actualKeys.size === expectedKeys.size && [...expectedKeys].every(key => actualKeys.has(key))
      : false;
    return Object.freeze({
      source_type: 'official',
      result_set: 'official',
      scoring_mode: 'retrieval_labels',
      metric_k: 5,
      configured,
      evaluated: rows.length,
      reported_evaluated: descriptor?.evaluated ?? rows.length,
      discovered: referenceRows.length,
      matrix_complete: matrixComplete,
      stability_status: matrixComplete ? 'complete' : 'partial',
      rows: Object.freeze(rows),
      incomplete_rows: Object.freeze(incompleteRows),
    });
  }

  function baselineResultPayload(evaluation, evidenceRows) {
    const expectedKeys = expectedKeySet(evaluation?.benchmark_reference?.report?.baseline_expected_keys);
    const evidenceCounts = new Map();
    (Array.isArray(evidenceRows) ? evidenceRows : []).forEach(raw => {
      const key = metricRowKey(canonicalMetricRow(raw));
      evidenceCounts.set(key, (evidenceCounts.get(key) || 0) + 1);
    });
    const incompleteRows = [];
    const rows = admittedOfficialRows(
      [
        ...(evaluation?.summary || []),
        ...(evaluation?.reranked?.summary || []),
      ],
      {expectedKeys, reranked: false},
      evidenceCounts,
      incompleteRows,
    );
    return Object.freeze({
      source_type: 'official',
      result_set: 'baseline',
      scoring_mode: 'retrieval_labels',
      metric_k: 5,
      configured: expectedKeys?.size ?? 0,
      evaluated: rows.length,
      rows: Object.freeze(rows),
      incomplete_rows: Object.freeze(incompleteRows),
    });
  }

  function normalizeResultPayload(payload) {
    const normalized = Object.assign({}, payload || {});
    normalized.source_type = normalized.source_type || 'uploaded_project';
    normalized.scoring_mode = normalized.scoring_mode || 'evidence_only';
    const metricK = finite(normalized.metric_k);
    normalized.rows = Object.freeze((Array.isArray(normalized.rows) ? normalized.rows : []).map(row => {
      const next = Object.assign({}, row || {});
      // Alias project-matrix fields onto the dashboard's official display keys so
      // Metrics/Overview/Evidence filters and pipeline labels update immediately.
      next.sheet = next.sheet || next.chunker_id || '';
      next.embedding = next.embedding || next.embedding_id || '';
      next.store = next.store || next.vector_store_id || '';
      next.reranker = canonicalRerankerName(next.reranker || next.reranker_id || 'none');
      if (finite(next.recall_at_5) === null && finite(next.recall_at_k) !== null) next.recall_at_5 = next.recall_at_k;
      if (finite(next.mrr) === null && finite(next.mrr_at_k) !== null) next.mrr = next.mrr_at_k;
      if (finite(next.ndcg_at_5) === null && finite(next.ndcg_at_k) !== null) next.ndcg_at_5 = next.ndcg_at_k;
      if (finite(next.avg_latency_seconds) === null && finite(next.avg_query_latency_s) !== null) {
        next.avg_latency_seconds = next.avg_query_latency_s;
      }
      if (finite(next.evaluated_queries) === null) {
        const labelled = finite(next.labelled_queries);
        const queries = finite(next.query_count);
        next.evaluated_queries = labelled !== null ? labelled : queries;
      }
      if (!next.combo_id) {
        next.combo_id = metricRowKey(next);
      }
      if (metricK !== null && next.metric_k == null) next.metric_k = metricK;
      if (finite(next.winner_score) === null) {
        const score = metricScore(next);
        if (score !== null) next.winner_score = score;
      }
      return Object.freeze(next);
    }));
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
    officialRowAdmission,
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
