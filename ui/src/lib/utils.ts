import { type ClassValue, clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';
import { browser } from '$app/environment';
import { pushState, replaceState } from '$app/navigation';
import { page } from '$app/state';
import { trip } from '@motis-project/motis-client';
import { joinInterlinedLegs } from './preprocessItinerary';
import { language } from './i18n/translation';
import { tick } from 'svelte';

export function cn(...inputs: ClassValue[]) {
	return twMerge(clsx(inputs));
}

const urlParams = browser ? new URLSearchParams(window.location.search) : undefined;

export const getUrlArray = (key: string, defaultValue?: string[]): string[] => {
	if (urlParams) {
		const value = urlParams.get(key);
		if (value != null) {
			return value.split(',').filter((m) => m.length);
		}
	}
	if (defaultValue) {
		return defaultValue;
	}
	return [];
};

export const preserveFromUrl = (
	// eslint-disable-next-line
	queryParams: Record<string, any>,
	field: string
) => {
	if (urlParams?.has(field)) {
		queryParams[field] = urlParams.get(field);
	}
};

export const pushStateWithQueryString = async (
	// eslint-disable-next-line
	queryParams: Record<string, any>,

	newState: App.PageState,
	replace: boolean = false
) => {
	preserveFromUrl(queryParams, 'debug');
	preserveFromUrl(queryParams, 'dark');
	preserveFromUrl(queryParams, 'light');
	preserveFromUrl(queryParams, 'motis');
	preserveFromUrl(queryParams, 'language');
	const params = new URLSearchParams(queryParams);
	const updateState = replace ? replaceState : pushState;
	try {
		updateState('?' + params.toString(), newState);
	} catch (e) {
		console.log(e);
		await tick();
		updateState('?' + params.toString(), newState);
	}
};

export const closeItinerary = () => {
	if (page.state.selectedStop) {
		onClickStop(
			page.state.selectedStop.name,
			page.state.selectedStop.stopId,
			page.state.selectedStop.time,
			page.state.stopArriveBy ?? false,
			true
		);
		return;
	}

	pushStateWithQueryString({}, {});
};

export const onClickStop = (
	name: string,
	stopId: string,
	time: Date,
	arriveBy: boolean = false,
	replace: boolean = false
) => {
	pushStateWithQueryString(
		{ stopArriveBy: arriveBy, stopId, time: time.toISOString() },
		{
			stopArriveBy: arriveBy,
			selectedStop: { name, stopId, time },
			selectedItinerary: replace ? undefined : page.state.selectedItinerary,
			tripId: replace ? undefined : page.state.tripId,
			activeTab: 'departures'
		},
		replace
	);
};

export const onClickTrip = async (tripId: string, replace: boolean = false) => {
	const { data: itinerary, error } = await trip({
		query: { tripId, joinInterlinedLegs: false, language: [language] }
	});
	if (error) {
		console.log(error);
		alert(String((error as Record<string, unknown>).error?.toString() ?? error));
		return;
	}
	joinInterlinedLegs(itinerary!);
	pushStateWithQueryString(
		{ tripId },
		{
			selectedItinerary: itinerary,
			tripId: tripId,
			selectedStop: replace ? undefined : page.state.selectedStop,
			activeTab: 'connections'
		},
		replace
	);
};
