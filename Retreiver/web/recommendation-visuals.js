(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  root.RecommendationVisuals = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';

  const palette = Object.freeze([
    'var(--chart-cyan)',
    'var(--chart-mint)',
    'var(--chart-violet)',
    'var(--chart-amber)',
    'var(--chart-rose)',
    'var(--chart-blue)',
  ]);

  function categoryColorMap(values, existing = {}) {
    const categories = [...new Set((values || []).map(value => String(value || 'none')))]
      .sort((a, b) => a.localeCompare(b));
    const output = {};
    const occupied = new Set(Object.values(existing || {}));
    categories.forEach(value => {
      if (existing && existing[value]) {
        output[value] = existing[value];
        return;
      }
      const available = palette.find(color => !occupied.has(color));
      const color = available || palette[Object.keys(output).length % palette.length];
      output[value] = color;
      occupied.add(color);
    });
    return output;
  }

  function rerankerShape(value) {
    const key = String(value || 'none').trim().toLowerCase();
    if (!key || key === 'none' || key.includes('no-reranker') || key.includes('no rerank')) {
      return 'circle';
    }
    if (key.includes('bge')) return 'diamond';
    if (key.includes('amazon') || key.includes('aws')) return 'triangle';
    return 'square';
  }

  function familyKey(row = {}) {
    return [
      row.chunker_id || row.sheet || row.chunker,
      row.embedding_id || row.embedding,
      row.vector_store_id || row.store || row.vector_store,
    ].map(value => String(value || 'none')).join('|');
  }

  function applyQuickView(rows, view) {
    const safeRows = Array.isArray(rows) ? rows : [];
    if (view === 'top10') return safeRows.slice(0, 10);
    if (view === 'bottom10') return safeRows.slice(Math.max(0, safeRows.length - 10));
    return safeRows.slice();
  }

  function paretoRows(
    rows,
    xValue,
    yValue,
    xDirection = 'lower',
    yDirection = 'higher',
  ) {
    const safeRows = Array.isArray(rows) ? rows : [];
    const betterOrEqual = (a, b, direction) => direction === 'lower' ? a <= b : a >= b;
    const strictlyBetter = (a, b, direction) => direction === 'lower' ? a < b : a > b;
    return safeRows.filter(candidate => !safeRows.some(other => (
      other !== candidate
      && betterOrEqual(xValue(other), xValue(candidate), xDirection)
      && betterOrEqual(yValue(other), yValue(candidate), yDirection)
      && (
        strictlyBetter(xValue(other), xValue(candidate), xDirection)
        || strictlyBetter(yValue(other), yValue(candidate), yDirection)
      )
    )));
  }

  function collisionOffsets(rows, xValue, yValue) {
    const groups = new Map();
    (rows || []).forEach(row => {
      const x = Number(xValue(row));
      const y = Number(yValue(row));
      const key = `${x.toPrecision(8)}|${y.toPrecision(8)}`;
      groups.set(key, [...(groups.get(key) || []), row]);
    });
    const offsets = new Map();
    groups.forEach(group => {
      group.forEach((row, index) => {
        const angle = group.length === 1 ? 0 : (Math.PI * 2 * index) / group.length;
        const radius = group.length === 1 ? 0 : Math.min(4, 1.5 + group.length * 0.35);
        offsets.set(row, {
          dx: Math.cos(angle) * radius,
          dy: Math.sin(angle) * radius,
        });
      });
    });
    return offsets;
  }

  return Object.freeze({
    palette,
    categoryColorMap,
    rerankerShape,
    familyKey,
    applyQuickView,
    paretoRows,
    collisionOffsets,
  });
});
