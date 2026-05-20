import { expect, test } from '@playwright/test';

test('index page renders MOTIS tabs', async ({ page }) => {
	await page.goto('/');
	await expect(page.getByRole('tab', { name: 'Connections' })).toBeVisible();
});
