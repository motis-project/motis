import { browser } from '$app/environment';
import type { Location } from './Location';

const STORAGE_KEY = 'motis:history-locations';
const COORDINATE_ROUNDING_ERROR_THRESHOLD = 0.00001;
const DAY = 1000 * 60 * 60 * 24;
const MAX_AGE_MS = 4 * DAY;
const TOP_COUNT_KEEP_LIMIT = 10;
const MAX_RECENT_HISTORY = 8;
const MAX_OFTEN_USED = 5;

export type HistoryLocation = Location & {
	count: number;
	lastUsed: number;
};

function load(): HistoryLocation[] {
	if (!browser) return [];
	const raw = localStorage.getItem(STORAGE_KEY);
	if (!raw) return [];
	try {
		const parsed = JSON.parse(raw) as HistoryLocation[];
		return Array.isArray(parsed) ? parsed : [];
	} catch {
		return [];
	}
}

function save(data: HistoryLocation[]): void {
	if (!browser) return;
	localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
}

function isSamePlace(l1: Location, l2: Location): boolean {
	return (
		Math.abs(l1.match!.lat - l2.match!.lat) < COORDINATE_ROUNDING_ERROR_THRESHOLD &&
		Math.abs(l1.match!.lon - l2.match!.lon) < COORDINATE_ROUNDING_ERROR_THRESHOLD
	);
}

function sortByUsageAndRecency(items: HistoryLocation[]): HistoryLocation[] {
	const recent = [...items].sort((a, b) => b.lastUsed - a.lastUsed).slice(0, MAX_RECENT_HISTORY);
	const oftenUsed = [...items]
		.filter((i) => !recent.some((i2) => isSamePlace(i, i2)))
		.sort((a, b) => b.count - a.count)
		.slice(0, MAX_OFTEN_USED);
	return recent.concat(oftenUsed);
}

function filterExpiredHistory(data: HistoryLocation[]): HistoryLocation[] {
	const now = Date.now();
	const topByCount = [...data].sort((a, b) => b.count - a.count).slice(0, TOP_COUNT_KEEP_LIMIT);

	return data.filter((item) => {
		const isOlderThan4Days = now - item.lastUsed > MAX_AGE_MS;
		if (!isOlderThan4Days) {
			return true;
		}
		return item.count > 1 && topByCount.includes(item);
	});
}

function filterAndSort(data: HistoryLocation[]) {
	return sortByUsageAndRecency(filterExpiredHistory(data));
}

function isValidLocation(input: Location): boolean {
	return input.label !== undefined && input.match !== undefined;
}

export function recordHistoryLocation(input: Location): void {
	if (!browser) {
		return;
	}
	if (!isValidLocation(input)) {
		return;
	}
	const data = load();
	const now = Date.now();
	const existing = data.find((item) => isSamePlace(item, input));
	if (existing) {
		existing.count += 1;
		existing.lastUsed = now;
	} else {
		data.push({
			...input,
			count: 1,
			lastUsed: now
		});
	}
	save(filterAndSort(data));
}

export function getHistoryLocations(): HistoryLocation[] {
	return filterAndSort(load());
}

export function clearHistoryLocations(): void {
	if (!browser) {
		return;
	}
	localStorage.removeItem(STORAGE_KEY);
}
