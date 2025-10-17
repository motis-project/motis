import { rentals, type RentalProviderGroup } from '$lib/api/openapi';
import { browser } from '$app/environment';

let cachedGroups: RentalProviderGroup[] | null = null;
let pendingPromise: Promise<RentalProviderGroup[]> | null = null;

const sortProviderGroups = (groups: RentalProviderGroup[]): RentalProviderGroup[] => {
	return [...groups].sort((a, b) =>
		a.name.localeCompare(b.name, undefined, { sensitivity: 'base' })
	);
};

export const getRentalProviderGroups = async (): Promise<RentalProviderGroup[]> => {
	if (cachedGroups) {
		return cachedGroups;
	}

	if (!browser) {
		return [];
	}

	if (!pendingPromise) {
		pendingPromise = rentals({ query: { withProviders: false } })
			.then(({ data, error }) => {
				if (error) {
					throw error;
				}
				const groups = sortProviderGroups(data?.providerGroups ?? []);
				cachedGroups = groups;
				return groups;
			})
			.finally(() => {
				pendingPromise = null;
			});
	}

	try {
		return await pendingPromise;
	} catch (err) {
		// Reset cache so future retries are possible after an error.
		cachedGroups = null;
		throw err;
	}
};

export const getCachedRentalProviderGroups = (): RentalProviderGroup[] | null => cachedGroups;
