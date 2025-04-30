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
	selectModes: 'Select transit modes',
	defaultSelectedModes: 'All transit modes',
	wheelchair: 'Accessible transfers',
	bikeRental: 'Allow usage of sharing vehicles',
	bikeCarriage: 'Bike carriage',
	unreliableOptions: 'Depending on data availability, these options may be unreliable.',
	timetableSources: 'Timetable sources',
	tripCancelled: 'Trip cancelled',
	stopCancelled: 'Stop cancelled',
	inOutDisallowed: 'Entry/exit not possible',
	inDisallowed: 'Entry not possible',
	outDisallowed: 'Exit not possible',
	unscheduledTrip: 'Additional service',
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
	OTHER: 'Other'
};

export default translations;
