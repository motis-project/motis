import { expect, test, type Page } from '@playwright/test';

const setupRoutesOverlayMocks = async (page: Page) => {
	let routesRequests = 0;

	await page.route('**/api/v1/map/initial', async (route) => {
		await route.fulfill({
			status: 200,
			contentType: 'application/json',
			body: JSON.stringify({
				lat: 48.137154,
				lon: 11.576124,
				zoom: 12,
				serverConfig: {
					shapesDebugEnabled: false,
					maxOneToAllTravelTimeLimit: 120,
					maxPrePostTransitTimeLimit: 7200,
					maxDirectTimeLimit: 7200
				}
			})
		});
	});

	await page.route('**/api/experimental/map/routes**', async (route) => {
		routesRequests += 1;
		await route.fulfill({
			status: 200,
			contentType: 'application/json',
			body: JSON.stringify({
				routes: [],
				polylines: [],
				stops: [],
				zoomFiltered: false
			})
		});
	});

	await page.route('**/api/v1/rentals**', async (route) => {
		await route.fulfill({
			status: 200,
			contentType: 'application/json',
			body: JSON.stringify({
				providerGroups: [],
				providers: [],
				stations: [],
				vehicles: [],
				zones: []
			})
		});
	});

	return {
		getRoutesRequests: () => routesRequests
	};
};

// The routes overlay toggle is only rendered when the UI is opened in
// debug mode (`?debug`), so all tests below navigate with that flag.
test.describe('routes overlay toggle', () => {
	test('requests routes data when enabled', async ({ page }) => {
		const { getRoutesRequests } = await setupRoutesOverlayMocks(page);

		await page.goto('/?debug');

		const routesButton = page.getByRole('button', { name: 'Toggle routes overlay' });
		await expect(routesButton).toBeVisible();
		await routesButton.click();

		await expect.poll(() => getRoutesRequests()).toBeGreaterThan(0);
	});

	test('re-enabling routes overlay triggers a fresh fetch', async ({ page }) => {
		const { getRoutesRequests } = await setupRoutesOverlayMocks(page);

		await page.goto('/?debug');

		const routesButton = page.getByRole('button', { name: 'Toggle routes overlay' });
		await expect(routesButton).toBeVisible();

		// First enable: triggers the initial fetch.
		await routesButton.click();
		await expect.poll(() => getRoutesRequests()).toBeGreaterThanOrEqual(1);

		// Disable, then re-enable: the overlay should re-mount and fetch again.
		await routesButton.click();
		await routesButton.click();

		await expect.poll(() => getRoutesRequests()).toBeGreaterThanOrEqual(2);
	});
});
