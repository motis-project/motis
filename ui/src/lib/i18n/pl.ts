import type { Translations } from './translation';

const translations: Translations = {
	ticket: 'Bilet',
	ticketOptions: 'Opcje biletu',
	includedInTicket: 'Zawarte w bilecie',
	journeyDetails: 'Szczegóły podróży',
	transfers: 'przesiadki',
	walk: 'Pieszo',
	bike: 'Rower',
	cargoBike: 'Rower cargo',
	scooterStanding: 'Hulajnoga stojąca',
	scooterSeated: 'Hulajnoga z siedziskiem',
	car: 'Samochód',
	taxi: 'Taksówka',
	moped: 'Skuter',
	from: 'Z',
	to: 'Do',
	arrival: 'Przyjazd',
	departure: 'Odjazd',
	duration: 'Czas trwania',
	arrivals: 'Przyjazdy',
	later: 'później',
	earlier: 'wcześniej',
	departures: 'Odjazdy',
	connections: 'Połączenia',
	switchToArrivals: 'Przełącz na przyjazdy',
	switchToDepartures: 'Przełącz na odjazdy',
	track: 'Tor',
	arrivalOnTrack: 'Przyjazd na tor',
	tripIntermediateStops: (n: number) => {
		switch (n) {
			case 0:
				return 'Brak przystanków pośrednich';
			case 1:
				return '1 przystanek pośredni';
			default:
				return `${n} przystanków pośrednich`;
		}
	},
	sharingProvider: 'Dostawca',
	roundtripStationReturnConstraint: 'Pojazd musi zostać zwrócony do stacji początkowej.',
	noItinerariesFound: 'No itineraries found.',
	advancedSearchOptions: 'Options',
	selectModes: 'Select transit modes',
	defaultSelectedModes: 'All transit modes',
	wheelchair: 'Barrier-free transfers',
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
