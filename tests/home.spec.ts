import { test, expect } from '@playwright/test';

test.describe('Home page', () => {
  test('loads and can perform a search', async ({ page }) => {
    await page.goto('http://127.0.0.1:7860/', { waitUntil: 'networkidle', timeout: 60000 });
    // Verify h1 is visible
    await expect(page.locator('h1')).toBeVisible({ timeout: 60000 });
    await expect(page.locator('h1')).toHaveText('RAG vs AI Agent Comparison', { timeout: 60000 });
    const queryInput = page.locator('#query');
    await queryInput.fill('What is AI?');
    await page.click('#searchBtn');
    const answer = page.locator('#ragAnswer');
    await expect(answer).not.toHaveClass(/loading/);
    await expect(answer).not.toHaveText('Loading…');
    await expect(answer).not.toBeEmpty();
  });
});
