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

for (const [deviceName, viewport] of Object.entries(viewports)) {
  test(`${deviceName}: canonical dashboard pages and source contracts`, async ({ browser }) => {
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
        await expect(page.locator('#tradeoffChart')).toBeVisible();
        await expect(page.locator('#metricHeatmap')).toBeVisible();
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
