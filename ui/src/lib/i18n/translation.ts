import { browser } from '$app/environment';
import en from './en';
import de from './de';
import fr from './fr';
import pl from './pl';
import cz from './cz';

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
	position: string;
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
	selectTransitModes: string;
	defaultSelectedModes: string;
	selectElevationCosts: string;
	wheelchair: string;
	useRoutedTransfers: string;
	bikeRental: string;
	requireBikeTransport: string;
	requireCarTransport: string;
	considerRentalReturnConstraints: string;
	default: string;
	timetableSources: string;
	tripCancelled: string;
	stopCancelled: string;
	inOutDisallowed: string;
	inDisallowed: string;
	outDisallowed: string;
	unscheduledTrip: string;
	alertsAvailable: string;
	dataExpiredSince: string;
	FLEX: string;
	WALK: string;
	BIKE: string;
	RENTAL: string;
	CAR: string;
	CAR_DROPOFF: string;
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
	routingSegments: {
		maxTransfers: string;
		maxTravelTime: string;
		firstMile: string;
		lastMile: string;
		direct: string;
		maxPreTransitTime: string;
		maxPostTransitTime: string;
		maxDirectTime: string;
	};
	elevationCosts: {
		NONE: string;
		LOW: string;
		HIGH: string;
	};
	isochrones: {
		title: string;
		displayLevel: string;
		maxComputeLevel: string;
		canvasRects: string;
		canvasCircles: string;
		geojsonCircles: string;
		styling: string;
		noData: string;
		requestFailed: string;
	};
	RENTAL_BICYCLE: string;
	RENTAL_CARGO_BICYCLE: string;
	RENTAL_CAR: string;
	RENTAL_MOPED: string;
	RENTAL_SCOOTER_STANDING: string;
	RENTAL_SCOOTER_SEATED: string;
	RENTAL_OTHER: string;
	incline: string;
	CABLE_CAR: string;
	FUNICULAR: string;
	AREAL_LIFT: string;
	toll: string;
	accessRestriction: string;
	continuesAs: string;
};

const translations: Map<string, Translations> = new Map(
	Object.entries({
		pl,
		en,
		de,
		fr,
		cz
	})
);

const translationsKey = (
	browser ? (navigator.languages.find((l) => translations.has(l.slice(0, 2))) ?? 'en') : 'en'
)?.slice(0, 2);

export const language = translationsKey ?? (browser ? navigator.language : 'en');
export const t = translationsKey ? translations.get(translationsKey)! : en;
