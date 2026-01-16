import { browser } from '$app/environment';
import type { SavedOptions } from './types';
import { defaultQuery } from '$lib/defaults';
import type { PlanData } from '@motis-project/motis-client';
import { getPrePostDirectModes } from '$lib/Modes';

const STORAGE_KEY = 'motis-ui-options';

/**
 * Load saved options from localStorage
 * Returns null if localStorage is unavailable or data is corrupted
 */
export function loadOptions(): Partial<SavedOptions> | null {
	if (!browser) {
		return null;
	}

	try {
		const stored = localStorage.getItem(STORAGE_KEY);
		if (stored === null) {
			return null;
		}

		const parsed = JSON.parse(stored) as Partial<SavedOptions>;
		
		// Validate that parsed data has the expected structure
		// We don't validate all fields, just check it's an object
		if (typeof parsed !== 'object' || parsed === null) {
			console.warn('Invalid options data in localStorage, ignoring');
			return null;
		}

		return parsed;
	} catch (error) {
		console.warn('Error loading options from localStorage:', error);
		// If data is corrupted, remove it
		try {
			localStorage.removeItem(STORAGE_KEY);
		} catch {
			// Ignore errors when removing
		}
		return null;
	}
}

/**
 * Save options to localStorage
 * Returns true if successful, false otherwise
 */
export function saveOptions(options: Partial<SavedOptions>): boolean {
	if (!browser) {
		return false;
	}

	try {
		const serialized = JSON.stringify(options);
		localStorage.setItem(STORAGE_KEY, serialized);
		return true;
	} catch (error) {
		// Handle quota exceeded or other storage errors
		if (error instanceof DOMException) {
			if (error.name === 'QuotaExceededError') {
				console.warn('localStorage quota exceeded, cannot save options');
			} else {
				console.warn('Error saving options to localStorage:', error);
			}
		} else {
			console.warn('Error saving options to localStorage:', error);
		}
		return false;
	}
}

/**
 * Get default options based on defaultQuery
 */
export function getDefaultOptions(): SavedOptions {
	return {
		// Search options
		arriveBy: defaultQuery.arriveBy,
		timetableView: defaultQuery.timetableView,
		searchWindow: defaultQuery.searchWindow,
		numItineraries: defaultQuery.numItineraries,
		algorithm: defaultQuery.algorithm as PlanData['query']['algorithm'],

		// Advanced options
		useRoutedTransfers: defaultQuery.useRoutedTransfers,
		pedestrianProfile: defaultQuery.pedestrianProfile,
		requireBikeTransport: defaultQuery.requireBikeTransport,
		requireCarTransport: defaultQuery.requireCarTransport,
		transitModes: defaultQuery.transitModes,
		preTransitModes: getPrePostDirectModes(defaultQuery.preTransitModes, []),
		postTransitModes: getPrePostDirectModes(defaultQuery.postTransitModes, []),
		directModes: getPrePostDirectModes(defaultQuery.directModes, []),
		elevationCosts: defaultQuery.elevationCosts,
		maxTransfers: defaultQuery.maxTransfers,
		maxPreTransitTime: defaultQuery.maxPreTransitTime,
		maxPostTransitTime: defaultQuery.maxPostTransitTime,
		maxDirectTime: defaultQuery.maxDirectTime,
		ignorePreTransitRentalReturnConstraints: defaultQuery.ignorePreTransitRentalReturnConstraints,
		ignorePostTransitRentalReturnConstraints: defaultQuery.ignorePostTransitRentalReturnConstraints,
		ignoreDirectRentalReturnConstraints: defaultQuery.ignoreDirectRentalReturnConstraints,
		preTransitProviderGroups: defaultQuery.preTransitRentalProviderGroups,
		postTransitProviderGroups: defaultQuery.postTransitRentalProviderGroups,
		directProviderGroups: defaultQuery.directRentalProviderGroups,

		// Isochrones options
		isochronesDisplayLevel: defaultQuery.isochronesDisplayLevel,
		isochronesColor: defaultQuery.isochronesColor,
		isochronesOpacity: defaultQuery.isochronesOpacity,
		isochronesCircleResolution: defaultQuery.circleResolution,

		// Display options (defaults based on system preferences)
		theme: browser && window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches
			? 'dark'
			: 'light',
		colorMode: 'none',
		showMap: true,

		// Map options (default center is Paris)
		center: [2.258882912876089, 48.72559118651327],
		zoom: 15,
		level: 0
	};
}
