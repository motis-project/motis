import type { Location } from '$lib/Location';
import type { Translations } from '$lib/i18n/translation';

export type PageTitleState = {
	activeTab: 'connections' | 'departures' | 'isochrones';
	from: Location;
	to: Location;
	one: Location;
	selectedStop?: { name: string };
	stopArriveBy?: boolean;
	stopName?: string;
};

const locationLabel = (location: Location): string => location.label || location.match?.name || '';

export const getPageTitle = (state: PageTitleState, translations: Translations): string => {
	const { pageTitle } = translations;

	if (state.activeTab === 'departures' && state.selectedStop) {
		const stopName = state.stopName || state.selectedStop.name;
		if (stopName) {
			return state.stopArriveBy
				? pageTitle.arrivalsAt(stopName)
				: pageTitle.departuresAt(stopName);
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
		if (fromLabel && toLabel) {
			return pageTitle.fromTo(fromLabel, toLabel);
		}
	}

	return pageTitle.default;
};
