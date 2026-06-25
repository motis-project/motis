import type { Location } from '$lib/Location';
import type { Translations } from '$lib/i18n/translation';
import type { Itinerary } from '@motis-project/motis-client';
import { formatDateTimeWithTimeZone } from '$lib/toDateTime';

export type PageTitleState = {
	activeTab: 'connections' | 'departures' | 'isochrones';
	from: Location;
	to: Location;
	one: Location;
	selectedStop?: { name: string };
	stopArriveBy?: boolean;
	stopName?: string;
	selectedItinerary?: Itinerary;
};

const locationLabel = (location: Location): string => location.match?.name || location.label || '';

export const getPageTitle = (state: PageTitleState, translations: Translations): string => {
	const { pageTitle } = translations;

	if (state.activeTab === 'departures' && state.selectedStop) {
		const stopName = state.stopName || state.selectedStop.name;
		if (stopName) {
			return state.stopArriveBy ? pageTitle.arrivalsAt(stopName) : pageTitle.departuresAt(stopName);
		}
	}

	if (state.activeTab === 'isochrones') {
		const label = locationLabel(state.one);
		if (label) {
			return pageTitle.isochronesFrom(label);
		}
	}

	if (state.activeTab === 'connections') {
		const fromLabel = locationLabel(state.from);
		const toLabel = locationLabel(state.to);
		if (state.selectedItinerary) {
			const it = state.selectedItinerary;
			return pageTitle.fromTo(
				`${fromLabel} (${formatDateTimeWithTimeZone(new Date(it.legs[0].startTime), it.legs[0].from.tz)})`,
				`${toLabel} (${formatDateTimeWithTimeZone(new Date(it.legs[it.legs.length - 1].endTime), it.legs[it.legs.length - 1].to.tz)})`
			);
		}
		if (fromLabel && toLabel) {
			return pageTitle.fromTo(fromLabel, toLabel);
		}
	}

	return pageTitle.default;
};
