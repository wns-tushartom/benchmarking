const { test, expect } = require('@playwright/test');
const fs = require('node:fs');
const path = require('node:path');

const baseURL = process.env.DASHBOARD_QA_URL || 'http://127.0.0.1:5098';
const artifactRoot = process.env.DASHBOARD_QA_ARTIFACTS || '/tmp/canonical-dashboard-browser-qa';
const pages = [
  'overview',
  'quality',
  'compare',
  'run',
  'evidence',
  'hallucination',
  'nvidia',
  'repository',
  'upload',
  'ops',
];
const viewports = {
  desktop: { width: 1440, height: 1100 },
  mobile: { width: 390, height: 844 },
};

function recommendationFixtureRows() {
  const rows = [];
  const chunkers = ['Heading sections', 'Fixed 1200', 'Entity w4', 'Entity w5', 'Entity w6'];
  const embeddings = ['OpenAI large', 'BGE large', 'Qwen embedding'];
  const stores = ['FAISS', 'PGVector', 'Qdrant', 'Weaviate'];
  const rerankers = ['none', 'bge-reranker-base', 'Amazon Rerank v1'];
  let index = 0;
  for (const chunker of chunkers) for (const embedding of embeddings) for (const store of stores) for (const reranker of rerankers) {
    const recall = 0.68 + ((179 - index) / 179) * 0.28;
    const latency = 0.08 + (index % 30) * 0.027 + Math.floor(index / 30) * 0.014;
    rows.push({
      combo_id: `visual-fixture-${index}`,
      chunker_id: chunker,
      embedding_id: embedding,
      vector_store_id: store,
      reranker_id: reranker,
      sheet: chunker,
      embedding,
      store,
      reranker,
      status: 'completed',
      state: 'complete',
      labelled_queries: 500,
      evaluated_queries: 500,
      recall_at_k: recall,
      recall_at_5: recall,
      mrr_at_k: recall - 0.045,
      mrr: recall - 0.045,
      ndcg_at_k: recall - 0.02,
      ndcg_at_5: recall - 0.02,
      avg_query_latency_s: latency,
      avg_latency_seconds: latency,
      commercial_model_ids: [],
      measured_usage: {},
      evidence_count: 1,
    });
    index += 1;
  }
  return rows;
}

for (const [deviceName, viewport] of Object.entries(viewports)) {
  test(`${deviceName}: canonical dashboard pages and source contracts`, async ({ browser }) => {
    test.setTimeout(90_000);
    const context = await browser.newContext({ viewport });
    const page = await context.newPage();
    const consoleErrors = [];
    const pageErrors = [];
    const failedRequests = [];
    page.on('console', message => {
      if (message.type() === 'error') consoleErrors.push(message.text());
    });
    page.on('pageerror', error => pageErrors.push(String(error)));
    page.on('requestfailed', request => {
      failedRequests.push(`${request.method()} ${request.url()} :: ${request.failure()?.errorText || 'failed'}`);
    });

    await page.goto(baseURL, { waitUntil: 'networkidle' });
    await expect(page.locator('#globalDataset')).toHaveValue('dataset:wns-default');
    await expect(page.locator('#globalGroundtruth')).toHaveValue(
      'groundtruth:repository:groundtruth_500.csv'
    );
    await expect(page.locator('#globalGroundtruth')).toBeDisabled();
    await expect(page.locator('#coverageHint')).toContainText(/\/180 .* combos covered/);
    await expect(page.locator('#coverageTable tbody tr')).toHaveCount(45);
    await expect(
      page.locator('#coverageTable tbody tr td:nth-child(n+4):nth-child(-n+7)')
    ).toHaveCount(180);

    const outputDir = path.join(artifactRoot, deviceName);
    fs.mkdirSync(outputDir, { recursive: true });
    for (const pageName of pages) {
      await page.evaluate(name => {
        if (typeof window.switchPage === 'function') {
          window.switchPage(name);
          return;
        }
        document.querySelectorAll('[data-page-panel]').forEach(panel => {
          panel.classList.toggle('active', panel.dataset.pagePanel === name);
        });
      }, pageName);
      await expect(page.locator(`[data-page-panel="${pageName}"]`)).toHaveClass(/active/);
      if (pageName === 'compare') {
        await page.evaluate(rows => {
          window.renderRecommendationSource({
            source_type: 'uploaded_project',
            project_id: 'visual-contract-fixture',
            project_label: 'Deterministic 180-row UI contract fixture',
            run_id: 'visual-contract-run',
            run_state: 'completed',
            scoring_mode: 'retrieval_labels',
            metric_k: 5,
            rows,
          });
        }, recommendationFixtureRows());
        await expect(page.locator('#recommendationContext')).toContainText('Deterministic 180-row UI contract fixture');
        await expect(page.locator('#tradeoffChart')).toBeVisible();
        await expect(page.locator('#metricHeatmap')).toBeVisible();
        await expect(page.locator('#tradeoffChart .metric-point')).toHaveCount(180);
        await expect(page.locator('#tradeoffChart .metric-point.metric-rank-top')).toHaveCount(5);
        await expect(page.locator('#tradeoffChart .metric-point.metric-rank-bottom')).toHaveCount(5);
        await expect(page.locator('#tradeoffChart .metric-shape-circle')).toHaveCount(60);
        await expect(page.locator('#tradeoffChart .metric-shape-diamond')).toHaveCount(60);
        await expect(page.locator('#tradeoffChart .metric-shape-triangle')).toHaveCount(60);
        await expect(page.locator('#tradeoffChart .metric-legend-filter')).toHaveCount(4);
        const distinctFills = await page.locator('#tradeoffChart .metric-tradeoff-dot').evaluateAll(nodes =>
          new Set(nodes.map(node => getComputedStyle(node).fill)).size
        );
        expect(distinctFills).toBeGreaterThanOrEqual(4);
        await page.locator('#metricQuickView').selectOption('top10');
        await expect(page.locator('#tradeoffChart .metric-point')).toHaveCount(10);
        await page.locator('#metricQuickView').selectOption('bottom10');
        await expect(page.locator('#tradeoffChart .metric-point')).toHaveCount(10);
        await page.locator('#metricQuickView').selectOption('all');
        await expect(page.locator('#tradeoffChart .metric-point')).toHaveCount(180);
        await page.screenshot({
          path: path.join(outputDir, 'recommendations-option1.png'),
          fullPage: true,
        });
        const firstPoint = page.locator('#tradeoffChart .metric-point').first();
        await firstPoint.focus();
        await expect(page.locator('#metricPointDetails')).not.toContainText('Select a point');
        await expect(page.locator('#tradeoffChart .metric-point.is-related')).toHaveCount(3);
        await expect(page.locator('#tradeoffChart .metric-family-link')).toHaveCount(2);
        await expect(page.locator('#metricHeatmap .metric-heatmap-cell.is-best')).not.toHaveCount(0);
        await expect(page.locator('#metricHeatmap .metric-heatmap-cell.is-worst')).not.toHaveCount(0);
        await page.locator('#tradeoffChart').scrollIntoViewIfNeeded();
        await firstPoint.press('Enter');
        await expect(page.locator('#tradeoffChart .metric-point.is-selected')).toHaveCount(1);
        await page.screenshot({
          path: path.join(outputDir, 'recommendations-family-selected.png'),
          fullPage: true,
        });
      }
      const globalOverflow = await page.evaluate(() => ({
        clientWidth: document.documentElement.clientWidth,
        scrollWidth: document.documentElement.scrollWidth,
      }));
      expect(
        globalOverflow.scrollWidth,
        `${deviceName}/${pageName} has global horizontal overflow: ${JSON.stringify(globalOverflow)}`
      ).toBeLessThanOrEqual(globalOverflow.clientWidth + 1);
      await page.screenshot({
        path: path.join(outputDir, `${pageName}.png`),
        fullPage: true,
      });
    }

    expect(consoleErrors, `console errors: ${consoleErrors.join('\n')}`).toEqual([]);
    expect(pageErrors, `page errors: ${pageErrors.join('\n')}`).toEqual([]);
    expect(failedRequests, `failed requests: ${failedRequests.join('\n')}`).toEqual([]);
    await context.close();
  });
}
