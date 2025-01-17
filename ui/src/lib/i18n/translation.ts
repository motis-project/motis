import { browser } from '$app/environment';
import en from './en';
import de from './de';
import fr from './fr';
import pl from './pl';

export type Translations = {
	journeyDetails: string;
	transfers: string;
	walk: string;
	bike: string;
	cargoBike: string;
	scooterStanding: string;
	scooterSeated: string;
	car: string;
	moped: string;
	from: string;
	to: string;
	arrival: string;
	departure: string;
	duration: string;
	later: string;
	earlier: string;
	arrivals: string;
	departures: string;
	switchToArrivals: string;
	switchToDepartures: string;
	arrivalOnTrack: string;
	track: string;
	tripIntermediateStops: (n: number) => string;
	sharingProvider: string;
	roundtripStationReturnConstraint: string;
	noItinerariesFound: string;
};

const translations: Map<string, Translations> = new Map(
	Object.entries({
		pl,
		en,
		de,
		fr
	})
);

export const language = (browser ? navigator.languages.find((l) => l.length == 2) : 'en') ?? 'en';
export const t = translations.get(language) ?? en;
