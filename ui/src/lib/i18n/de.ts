import type { Translations } from './translation';

const translations: Translations = {
	ticket: 'Fahrschein',
	ticketOptions: 'Fahrscheinoptionen',
	includedInTicket: 'Im Fahrschein enthalten',
	journeyDetails: 'Verbindungsdetails',
	transfers: 'Umstiege',
	walk: 'Fußweg',
	bike: 'Fahrrad',
	cargoBike: 'Lastenfahrrad',
	scooterStanding: 'Stehroller',
	scooterSeated: 'Sitzroller',
	car: 'Auto',
	taxi: 'Taxi',
	moped: 'Moped',
	from: 'Von',
	to: 'Nach',
	arrival: 'Ankunft',
	departure: 'Abfahrt',
	duration: 'Dauer',
	arrivals: 'Ankünfte',
	connections: 'Verbindungen',
	departures: 'Abfahrten',
	later: 'später',
	earlier: 'früher',
	track: 'Gleis',
	arrivalOnTrack: 'Ankunft auf Gleis',
	switchToArrivals: 'Wechsel zu Ankünften',
	switchToDepartures: 'Wechsel zu Abfahrten',
	tripIntermediateStops: (n: number) => {
		switch (n) {
			case 0:
				return 'Fahrt ohne Zwischenhalt';
			case 1:
				return 'Fahrt eine Station';
			default:
				return `Fahrt ${n} Stationen`;
		}
	},
	sharingProvider: 'Anbieter',
	roundtripStationReturnConstraint:
		'Das Fahrzeug muss wieder an der Abfahrtsstation abgestellt werden.',
	noItinerariesFound: 'Keine Verbindungen gefunden.',
	advancedSearchOptions: 'Optionen',
	selectModes: 'Öffentliche Verkehrsmittel auswählen',
	defaultSelectedModes: 'Alle Verkehrsmittel',
	wheelchair: 'Barrierefreie Umstiege',
	bikeRental: 'Sharing-Fahrzeuge berücksichtigen',
	bikeCarriage: 'Fahrradmitnahme',
	unreliableOptions: 'Je nach Datenverfügbarkeit können diese Optionen unzuverlässig sein.',
	timetableSources: 'Fahrplandatenquellen',
	tripCancelled: 'Fahrt entfällt',
	stopCancelled: 'Halt entfällt',
	inOutDisallowed: 'Ein-/Ausstieg nicht möglich',
	inDisallowed: 'Einstieg nicht möglich',
	outDisallowed: 'Ausstieg nicht möglich',
	unscheduledTrip: 'Zusätzliche Fahrt',
	WALK: 'Zu Fuß',
	BIKE: 'Fahrrad',
	RENTAL: 'Sharing',
	CAR: 'Auto',
	CAR_PARKING: 'Car parking',
	TRANSIT: 'ÖPV',
	TRAM: 'Tram',
	SUBWAY: 'U-Bahn',
	FERRY: 'Fähre',
	AIRPLANE: 'Flugzeug',
	METRO: 'S-Bahn',
	BUS: 'Bus',
	COACH: 'Reisebus',
	RAIL: 'Zug',
	HIGHSPEED_RAIL: 'Hochgeschwindigkeitszug',
	LONG_DISTANCE: 'Intercityzug',
	NIGHT_RAIL: 'Nachtzug',
	REGIONAL_FAST_RAIL: 'Regionalexpresszug',
	REGIONAL_RAIL: 'Regionalzug',
	OTHER: 'Andere'
};

export default translations;
