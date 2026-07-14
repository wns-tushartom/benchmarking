(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  root.PipelineRecommendations = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';

  const PRICING = Object.freeze({
    'openai_text-embedding-3-large': Object.freeze({
      unit: 'million_input_tokens',
      usd_rate: 0.13,
    }),
    'Amazon Rerank v1': Object.freeze({
      unit: 'search_unit',
      usd_rate: 0.001,
    }),
  });

  const finite = value => {
    if (value === null || value === undefined || value === '' || typeof value === 'boolean') return null;
    const number = Number(value);
    return Number.isFinite(number) ? number : null;
  };
  const positive = value => {
    const number = finite(value);
    return number !== null && number > 0 ? number : null;
  };
  const stableKey = row => [
    row.chunker_id || row.sheet,
    row.embedding_id || row.embedding,
    row.vector_store_id || row.store,
    row.reranker_id || row.reranker || 'none',
  ].join('|');
  const completed = (rows, sourceType) => (rows || []).filter(row => {
    if (!row) return false;
    if (sourceType === 'official') return !row.status || row.status === 'completed';
    return row.status === 'completed';
  });

  function projectQualityCompare(a, b) {
    return (finite(b.ndcg_at_k) ?? -1) - (finite(a.ndcg_at_k) ?? -1)
      || (finite(b.mrr_at_k) ?? -1) - (finite(a.mrr_at_k) ?? -1)
      || (finite(b.recall_at_k) ?? -1) - (finite(a.recall_at_k) ?? -1)
      || (positive(a.avg_query_latency_s) ?? Infinity) - (positive(b.avg_query_latency_s) ?? Infinity)
      || stableKey(a).localeCompare(stableKey(b));
  }

  function officialQualityCompare(a, b) {
    return (finite(b.winner_score) ?? -1) - (finite(a.winner_score) ?? -1)
      || (finite(b.recall_at_5) ?? -1) - (finite(a.recall_at_5) ?? -1)
      || (positive(a.avg_latency_seconds) ?? Infinity) - (positive(b.avg_latency_seconds) ?? Infinity)
      || stableKey(a).localeCompare(stableKey(b));
  }

  function measuredNumber(value) {
    if (value === null || value === undefined || value === '' || typeof value === 'boolean') {
      return null;
    }
    const number = finite(value);
    return number !== null && number >= 0 ? number : null;
  }

  function modelIdsForRow(row) {
    const ids = [];
    const declared = Array.isArray(row && row.commercial_model_ids)
      ? row.commercial_model_ids
      : [];
    declared.forEach(id => {
      if (typeof id === 'string' && id && !ids.includes(id)) ids.push(id);
    });

    const exactComponents = [
      row && (row.embedding_id || row.embedding),
      row && (row.reranker_id || row.reranker),
    ];
    exactComponents.forEach(id => {
      if (Object.prototype.hasOwnProperty.call(PRICING, id) && !ids.includes(id)) {
        ids.push(id);
      }
    });
    return ids;
  }

  function chargeForModel(row, modelId) {
    const usage = row && row.measured_usage && typeof row.measured_usage === 'object'
      ? row.measured_usage
      : {};
    const pricing = PRICING[modelId];

    if (modelId === 'openai_text-embedding-3-large') {
      const amount = measuredNumber(usage.embedding_input_tokens);
      const scope = usage.embedding_usage_scope;
      const usageKey = usage.embedding_usage_key;
      const validScope = scope === 'shared_embedding' || scope === 'combination';
      const validKey = scope !== 'shared_embedding'
        || (typeof usageKey === 'string' && usageKey.length > 0);
      if (amount === null || !validScope || !validKey) {
        return {
          model_id: modelId,
          unit: pricing.unit,
          usd_rate: pricing.usd_rate,
          usage: null,
          usage_scope: scope || null,
          cost_usd: null,
        };
      }
      return {
        model_id: modelId,
        unit: pricing.unit,
        usd_rate: pricing.usd_rate,
        usage: amount,
        usage_scope: scope,
        cost_usd: scope === 'combination' ? amount / 1000000 * pricing.usd_rate : null,
      };
    }

    const amount = measuredNumber(usage.rerank_search_units);
    const scope = usage.rerank_usage_scope;
    if (amount === null || scope !== 'combination') {
      return {
        model_id: modelId,
        unit: pricing.unit,
        usd_rate: pricing.usd_rate,
        usage: null,
        usage_scope: scope || null,
        cost_usd: null,
      };
    }
    return {
      model_id: modelId,
      unit: pricing.unit,
      usd_rate: pricing.usd_rate,
      usage: amount,
      usage_scope: scope,
      cost_usd: amount * pricing.usd_rate,
    };
  }

  function pricingForRow(row) {
    const modelIds = modelIdsForRow(row || {});
    if (!modelIds.length) {
      return {state: 'no_api_fee', charges: [], total_cost_usd: null};
    }
    if (modelIds.some(id => !Object.prototype.hasOwnProperty.call(PRICING, id))) {
      return {state: 'pricing_unavailable', charges: [], total_cost_usd: null};
    }

    const charges = modelIds.map(modelId => chargeForModel(row || {}, modelId));
    if (charges.some(charge => charge.usage === null)) {
      return {state: 'usage_missing', charges, total_cost_usd: null};
    }

    const allCombinationScoped = charges.every(charge => charge.usage_scope === 'combination');
    const total = allCombinationScoped
      ? charges.reduce((sum, charge) => sum + charge.cost_usd, 0)
      : null;
    return {state: 'measured_usage', charges, total_cost_usd: total};
  }

  function ledgerEntry(row, modelId) {
    const usage = row && row.measured_usage && typeof row.measured_usage === 'object'
      ? row.measured_usage
      : {};
    const pricing = PRICING[modelId];
    const comboId = typeof row.combo_id === 'string' && row.combo_id ? row.combo_id : null;

    if (modelId === 'openai_text-embedding-3-large') {
      const scope = usage.embedding_usage_scope || null;
      const usageKey = typeof usage.embedding_usage_key === 'string' && usage.embedding_usage_key
        ? usage.embedding_usage_key
        : null;
      const amount = measuredNumber(usage.embedding_input_tokens);
      const shared = scope === 'shared_embedding' && usageKey !== null;
      const valid = amount !== null && (shared || scope === 'combination');
      return {
        key: !valid
          ? `usage_missing:${modelId}`
          : shared
            ? `shared_embedding:${usageKey}`
            : `combination:${comboId || stableKey(row)}:${modelId}`,
        model_id: modelId,
        state: valid ? 'measured_usage' : 'usage_missing',
        unit: pricing.unit,
        usd_rate: pricing.usd_rate,
        usage: valid ? amount : null,
        usage_scope: scope,
        cost_usd: valid ? amount / 1000000 * pricing.usd_rate : null,
        combo_id: !valid || shared ? null : comboId,
      };
    }

    const amount = measuredNumber(usage.rerank_search_units);
    const scope = usage.rerank_usage_scope || null;
    const valid = amount !== null && scope === 'combination';
    return {
      key: valid
        ? `combination:${comboId || stableKey(row)}:${modelId}`
        : `usage_missing:${modelId}`,
      model_id: modelId,
      state: valid ? 'measured_usage' : 'usage_missing',
      unit: pricing.unit,
      usd_rate: pricing.usd_rate,
      usage: valid ? amount : null,
      usage_scope: scope,
      cost_usd: valid ? amount * pricing.usd_rate : null,
      combo_id: valid ? comboId : null,
    };
  }

  function pricingLedger(source) {
    const rows = Array.isArray(source && source.rows)
      ? source.rows.filter(Boolean).slice()
      : [];
    rows.sort((a, b) => stableKey(a).localeCompare(stableKey(b))
      || String(a.combo_id || '').localeCompare(String(b.combo_id || '')));

    const entries = [];
    const entriesByKey = new Map();
    rows.forEach(row => {
      modelIdsForRow(row).forEach(modelId => {
        let entry;
        if (!Object.prototype.hasOwnProperty.call(PRICING, modelId)) {
          entry = {
            key: `pricing_unavailable:${modelId}`,
            model_id: modelId,
            state: 'pricing_unavailable',
            unit: null,
            usd_rate: null,
            usage: null,
            usage_scope: null,
            cost_usd: null,
            combo_id: null,
          };
        } else {
          entry = ledgerEntry(row, modelId);
        }
        const existing = entriesByKey.get(entry.key);
        if (!existing) {
          entriesByKey.set(entry.key, entry);
          entries.push(entry);
        } else if (
          existing.state !== entry.state
          || existing.usage !== entry.usage
          || existing.usage_scope !== entry.usage_scope
          || existing.cost_usd !== entry.cost_usd
        ) {
          existing.state = 'usage_missing';
          existing.usage = null;
          existing.cost_usd = null;
        }
      });
    });
    return entries;
  }

  function projectQualityApplicable(row) {
    const labelled = measuredNumber(row.labelled_queries ?? row.labelled_query_count);
    return row.quality_applicable !== false && labelled !== null && labelled > 0;
  }

  function qualityRecorded(row, sourceType) {
    if (sourceType === 'official') {
      return measuredNumber(row.winner_score) !== null
        || measuredNumber(row.recall_at_5) !== null;
    }
    if (!projectQualityApplicable(row)) return false;
    return measuredNumber(row.ndcg_at_k) !== null
      || measuredNumber(row.mrr_at_k) !== null
      || measuredNumber(row.recall_at_k) !== null;
  }

  function qualityValue(row, sourceType) {
    if (sourceType === 'official') return measuredNumber(row.winner_score);
    if (!projectQualityApplicable(row)) return null;
    return measuredNumber(row.ndcg_at_k)
      ?? measuredNumber(row.mrr_at_k)
      ?? measuredNumber(row.recall_at_k);
  }

  function rowLatency(row, sourceType) {
    return sourceType === 'official'
      ? positive(row.avg_latency_seconds)
      : positive(row.avg_query_latency_s);
  }

  function sameRow(a, b) {
    if (!a || !b) return false;
    if (a.combo_id && b.combo_id) return a.combo_id === b.combo_id;
    return a === b || stableKey(a) === stableKey(b);
  }

  function valueRole(rows, sourceType, qualityCompare) {
    const measured = rows.map(row => ({
      row,
      pricing: pricingForRow(row),
      quality: qualityValue(row, sourceType),
    })).filter(candidate => candidate.pricing.state === 'measured_usage'
      && positive(candidate.pricing.total_cost_usd) !== null
      && candidate.quality !== null);

    if (measured.length >= 2) {
      measured.sort((a, b) => {
        const aRatio = a.quality / a.pricing.total_cost_usd;
        const bRatio = b.quality / b.pricing.total_cost_usd;
        return bRatio - aRatio || qualityCompare(a.row, b.row);
      });
      return Object.assign({}, measured[0].row, {recommendation_kind: 'value'});
    }

    const noApiFee = rows.filter(row => pricingForRow(row).state === 'no_api_fee'
      && qualityRecorded(row, sourceType));
    noApiFee.sort(qualityCompare);
    return noApiFee.length
      ? Object.assign({}, noApiFee[0], {recommendation_kind: 'no_api_fee'})
      : null;
  }

  function countOrFallback(value, fallback) {
    const number = measuredNumber(value);
    return number === null ? fallback : number;
  }

  function evidenceCoverage(source, rows) {
    const queryCounts = rows.map(row => measuredNumber(row.query_count));
    const derivedQueryCount = queryCounts.length && !queryCounts.some(value => value === null)
      ? Math.max(...queryCounts)
      : null;
    const queryCount = measuredNumber(source && source.query_count) ?? derivedQueryCount;
    const byCombo = source && source.evidence_counts_by_combo;
    const rawEvidenceCounts = byCombo && typeof byCombo === 'object' && !Array.isArray(byCombo)
      ? rows.map(row => Object.prototype.hasOwnProperty.call(byCombo, row.combo_id)
        ? byCombo[row.combo_id]
        : null)
      : rows.map(row => row.evidence_count);
    const measuredEvidenceCounts = rawEvidenceCounts.map(measuredNumber);
    const evidenceCount = measuredEvidenceCounts.some(value => value === null)
      ? null
      : measuredEvidenceCounts.reduce((sum, value) => sum + value, 0);
    return {state: 'evidence_coverage', query_count: queryCount, evidence_count: evidenceCount};
  }

  function recommendationsForSource(source) {
    const safeSource = source && typeof source === 'object' ? source : {};
    const sourceType = safeSource.source_type === 'official' ? 'official' : 'uploaded_project';
    const allRows = Array.isArray(safeSource.rows) ? safeSource.rows.filter(Boolean) : [];
    const completedRows = completed(allRows, sourceType);
    const failedRows = allRows.filter(row => row.status === 'failed').slice()
      .sort((a, b) => stableKey(a).localeCompare(stableKey(b)));
    const qualityCompare = sourceType === 'official'
      ? officialQualityCompare
      : projectQualityCompare;
    const mode = safeSource.scoring_mode === 'retrieval_labels' ? 'labelled' : 'evidence_only';

    if (mode === 'evidence_only') {
      const speedRows = completedRows.slice().sort((a, b) =>
        (positive(a.avg_query_latency_s) ?? Infinity)
          - (positive(b.avg_query_latency_s) ?? Infinity)
        || stableKey(a).localeCompare(stableKey(b)));
      const fastest = speedRows.find(row => positive(row.avg_query_latency_s) !== null)
        || {state: 'latency_not_recorded'};
      const sortedRows = speedRows.some(row => positive(row.avg_query_latency_s) !== null)
        ? speedRows
        : completedRows.slice().sort((a, b) => stableKey(a).localeCompare(stableKey(b)));
      const succeeded = countOrFallback(safeSource.succeeded, completedRows.length);
      const failed = countOrFallback(safeSource.failed, failedRows.length);
      return {
        source_type: sourceType,
        mode,
        roles: {
          fastest,
          run_health: {state: 'run_health', succeeded, failed, total: succeeded + failed},
          evidence_coverage: evidenceCoverage(safeSource, completedRows),
        },
        rows: sortedRows.concat(failedRows),
        completed: sortedRows,
        failed: failedRows,
        completed_rows: sortedRows,
        failed_rows: failedRows,
        pricing_ledger: pricingLedger(safeSource),
      };
    }

    const qualityRows = completedRows.slice().sort(qualityCompare);
    const quality = qualityRows.find(row => qualityRecorded(row, sourceType)) || null;
    const speedRows = completedRows.filter(row => rowLatency(row, sourceType) !== null);
    speedRows.sort((a, b) => rowLatency(a, sourceType) - rowLatency(b, sourceType)
      || qualityCompare(a, b));
    const roles = {
      quality,
      speed: speedRows[0] || null,
      value: valueRole(completedRows, sourceType, qualityCompare),
    };
    return {
      source_type: sourceType,
      mode,
      roles,
      rows: qualityRows.concat(failedRows),
      completed: qualityRows,
      failed: failedRows,
      completed_rows: qualityRows,
      failed_rows: failedRows,
      pricing_ledger: pricingLedger(safeSource),
    };
  }

  function metricLabels(source) {
    if (source && source.source_type === 'official') {
      return {
        recall: 'Recall@5',
        mrr: 'MRR',
        ndcg: 'nDCG@5',
        latency: 'Avg sec/query',
      };
    }
    const metricK = measuredNumber(source && source.metric_k);
    const suffix = Number.isInteger(metricK) && metricK > 0 ? String(metricK) : 'K';
    return {
      recall: `Recall@${suffix}`,
      mrr: `MRR@${suffix}`,
      ndcg: `nDCG@${suffix}`,
      latency: 'Avg sec/query',
    };
  }

  function rowBadges(row, roles) {
    if (!row || row.status === 'failed' || !roles) return [];
    const badges = [];
    if (sameRow(row, roles.quality)) badges.push('quality');
    if (sameRow(row, roles.speed)) badges.push('speed');
    if (sameRow(row, roles.value)) {
      badges.push(roles.value.recommendation_kind === 'no_api_fee' ? 'no_api_fee' : 'value');
    }
    return badges;
  }

  return Object.freeze({
    PRICING,
    finite,
    positive,
    stableKey,
    completed,
    projectQualityCompare,
    officialQualityCompare,
    pricingForRow,
    pricingLedger,
    recommendationsForSource,
    metricLabels,
    rowBadges,
  });
});
