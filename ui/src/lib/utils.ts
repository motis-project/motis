import { type ClassValue, clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';
import { browser } from '$app/environment';
import { pushState, replaceState } from '$app/navigation';
import { page } from '$app/state';
import { trip } from '$lib/openapi';

export function cn(...inputs: ClassValue[]) {
	return twMerge(clsx(inputs));
}

const urlParams = browser ? new URLSearchParams(window.location.search) : undefined;

export const preserveFromUrl = (
	// eslint-disable-next-line
	queryParams: Record<string, any>,
	field: string
) => {
	if (urlParams?.has(field)) {
		queryParams[field] = urlParams.get(field);
	}
};

export const pushStateWithQueryString = (
	// eslint-disable-next-line
	queryParams: Record<string, any>,
	// eslint-disable-next-line
	newState: App.PageState,
	replace: boolean = false
) => {
	preserveFromUrl(queryParams, 'debug');
	preserveFromUrl(queryParams, 'dark');
	preserveFromUrl(queryParams, 'motis');
	const params = new URLSearchParams(queryParams);
	const updateState = replace ? replaceState : pushState;
	updateState('?' + params.toString(), newState);
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
			showDepartures: true
		},
		replace
	);
};

export const onClickTrip = async (tripId: string, replace: boolean = false) => {
	const { data: itinerary, error } = await trip({ query: { tripId } });
	if (error) {
		console.log(error);
		alert(String((error as Record<string, unknown>).error?.toString() ?? error));
		return;
	}
	pushStateWithQueryString(
		{ tripId },
		{
			selectedItinerary: itinerary,
			tripId: tripId,
			selectedStop: replace ? undefined : page.state.selectedStop,
			showDepartures: false
		},
		replace
	);
};
