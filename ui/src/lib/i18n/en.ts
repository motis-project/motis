import type { Translations } from './translation';

const translations: Translations = {
	ticket: 'Ticket',
	ticketOptions: 'Ticket Options',
	includedInTicket: 'Included in ticket',
	journeyDetails: 'Journey Details',
	transfers: 'transfers',
	walk: 'Walk',
	bike: 'Bike',
	cargoBike: 'Cargo bike',
	scooterStanding: 'Standing kick scooter',
	scooterSeated: 'Seated kick scooter',
	car: 'Car',
	taxi: 'Taxi',
	moped: 'Moped',
	from: 'From',
	to: 'To',
	arrival: 'Arrival',
	departure: 'Departure',
	connections: 'Connections',
	duration: 'Duration',
	arrivals: 'Arrivals',
	later: 'later',
	earlier: 'earlier',
	departures: 'Departures',
	switchToArrivals: 'Switch to arrivals',
	switchToDepartures: 'Switch to departures',
	track: 'Track',
	arrivalOnTrack: 'Arrival on track',
	tripIntermediateStops: (n: number) => {
		switch (n) {
			case 0:
				return 'No intermediate stops';
			case 1:
				return '1 intermediate stop';
			default:
				return `${n} intermediate stops`;
		}
	},
	sharingProvider: 'Provider',
	roundtripStationReturnConstraint: 'The vehicle must be returned to the departure station.',
	noItinerariesFound: 'No itineraries found.',
	advancedSearchOptions: 'Options',
	selectTransitModes: 'Select transit modes',
	defaultSelectedModes: 'All transit modes',
	selectElevationCosts: 'Avoid steep incline.',
	wheelchair: 'Accessible transfers',
	bikeRental: 'Allow usage of sharing vehicles',
	requireBikeTransport: 'Bike carriage',
	requireCarTransport: 'Car carriage',
	default: 'Default',
	timetableSources: 'Timetable sources',
	tripCancelled: 'Trip cancelled',
	stopCancelled: 'Stop cancelled',
	inOutDisallowed: 'Entry/exit not possible',
	inDisallowed: 'Entry not possible',
	outDisallowed: 'Exit not possible',
	unscheduledTrip: 'Additional service',
	alertsAvailable: 'Service alerts present',
	FLEX: 'On-Demand',
	WALK: 'Walking',
	BIKE: 'Bike',
	RENTAL: 'Sharing',
	CAR: 'Car',
	CAR_PARKING: 'Car Parking',
	TRANSIT: 'Transit',
	TRAM: 'Tram',
	SUBWAY: 'Subway',
	FERRY: 'Ferry',
	AIRPLANE: 'Airplane',
	METRO: 'Metropolitan Rail',
	BUS: 'Bus',
	COACH: 'Coach',
	RAIL: 'Train',
	HIGHSPEED_RAIL: 'High Speed Rail',
	LONG_DISTANCE: 'Intercity Rail',
	NIGHT_RAIL: 'Night Rail',
	REGIONAL_FAST_RAIL: 'Regional Fast Rail',
	REGIONAL_RAIL: 'Regional Rail',
	OTHER: 'Other',
	RENTAL_BICYCLE: 'Shared bike',
	RENTAL_CARGO_BICYCLE: 'Shared cargo bike',
	RENTAL_CAR: 'Shared car',
	RENTAL_MOPED: 'Shared moped',
	RENTAL_SCOOTER_STANDING: 'Shared standing scooter',
	RENTAL_SCOOTER_SEATED: 'Shared seated scooter',
	RENTAL_OTHER: 'Other shared vehicle',
	routingSegments: {
		firstMile: 'First mile',
		lastMile: 'Last mile',
		direct: 'Direct connection',
		maxPreTransitTime: 'Max. pre-transit time',
		maxPostTransitTime: 'Max. post-transit time',
		maxDirectTime: 'Max. direct time'
	},
	elevationCosts: {
		NONE: 'No detours',
		LOW: 'Small detours',
		HIGH: 'Large detours'
	}
};

export default translations;
