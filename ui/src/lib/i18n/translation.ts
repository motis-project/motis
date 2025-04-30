import { browser } from '$app/environment';
import en from './en';
import de from './de';
import fr from './fr';
import pl from './pl';

export type Translations = {
	ticket: string;
	ticketOptions: string;
	includedInTicket: string;
	journeyDetails: string;
	transfers: string;
	walk: string;
	bike: string;
	cargoBike: string;
	scooterStanding: string;
	scooterSeated: string;
	car: string;
	taxi: string;
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
	connections: string;
	switchToArrivals: string;
	switchToDepartures: string;
	arrivalOnTrack: string;
	track: string;
	tripIntermediateStops: (n: number) => string;
	sharingProvider: string;
	roundtripStationReturnConstraint: string;
	noItinerariesFound: string;
	advancedSearchOptions: string;
	selectModes: string;
	defaultSelectedModes: string;
	wheelchair: string;
	bikeRental: string;
	bikeCarriage: string;
	unreliableOptions: string;
	timetableSources: string;
	tripCancelled: string;
	stopCancelled: string;
	inOutDisallowed: string;
	inDisallowed: string;
	outDisallowed: string;
	unscheduledTrip: string;
	WALK: string;
	BIKE: string;
	RENTAL: string;
	CAR: string;
	CAR_PARKING: string;
	TRANSIT: string;
	TRAM: string;
	SUBWAY: string;
	FERRY: string;
	AIRPLANE: string;
	METRO: string;
	BUS: string;
	COACH: string;
	RAIL: string;
	HIGHSPEED_RAIL: string;
	LONG_DISTANCE: string;
	NIGHT_RAIL: string;
	REGIONAL_FAST_RAIL: string;
	REGIONAL_RAIL: string;
	OTHER: string;
};

const translations: Map<string, Translations> = new Map(
	Object.entries({
		pl,
		en,
		de,
		fr
	})
);

const translationsKey = (
	browser ? (navigator.languages.find((l) => translations.has(l.slice(0, 2))) ?? 'en') : 'en'
)?.slice(0, 2);

export const language = translationsKey ?? (browser ? navigator.language : 'en');
export const t = translationsKey ? translations.get(translationsKey)! : en;
