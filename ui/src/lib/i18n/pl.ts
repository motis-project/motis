import type { Translations } from './translation';

const translations: Translations = {
	ticket: 'Bilet',
	ticketOptions: 'Opcje biletu',
	includedInTicket: 'Zawarte w ramach biletu',
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
	unknownVehicleType: 'Nieznany typ pojazdu',
	electricAssist: 'Wspomaganie elektryczne',
	electric: 'Elektryczny',
	combustion: 'Spalinowy',
	combustionDiesel: 'Diesel',
	hybrid: 'Hybrydowy',
	plugInHybrid: 'Hybryda plug-in',
	hydrogenFuelCell: 'Ogniwo paliwowe na wodór',
	from: 'Z',
	to: 'Do',
	viaStop: 'Przystanek pośredni',
	viaStops: 'Przystanki pośrednie',
	addViaStop: 'Dodaj przystanek pośredni',
	removeViaStop: 'Usuń przystanek pośredni',
	viaStayDuration: 'Minimalny postój',
	position: 'Pozycja',
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
	platform: 'Peron',
	trackAbr: 'T.',
	platformAbr: 'Pr.',
	arrivalOnTrack: 'Przyjazd na tor',
	tripIntermediateStops: (n: number) => {
		if (n == 0) {
			return 'Brak przystanków pośrednich';
		}
		if (n == 1) {
			return '1 przystanek pośredni';
		}
		if (n % 10 > 1 && n % 10 < 5 && n != 12 && n != 13 && n != 14) {
			return `${n} przystanki pośrednie`;
		}
		return `${n} przystanków pośrednich`;
	},
	sharingProvider: 'Dostawca danych',
	sharingProviders: 'Dostawcy danych',
	returnOnlyAtStations: 'Pojazd musi zostać zwrócony na stacji.',
	roundtripStationReturnConstraint: 'Pojazd musi zostać zwrócony do stacji początkowej.',
	rentalStation: 'Stacja',
	rentalGeofencingZone: 'Strefa',
	noItinerariesFound: 'Nie znaleziono połączeń.',
	advancedSearchOptions: 'Opcje',
	selectTransitModes: 'Wybierz środki transportu',
	defaultSelectedModes: 'Wszystkie środki transportu',
	defaultSelectedProviders: 'Wszyscy dostawcy',
	selectElevationCosts: 'Unikaj stromych nachyleń.',
	wheelchair: 'Bezbarierowe przesiadki',
	useRoutedTransfers: 'Wyznacz trasy dla przesiadek',
	bikeRental: 'Użyj pojazdów współdzielonych',
	requireBikeTransport: 'Przewóz roweru',
	requireCarTransport: 'Przewóz samochodu',
	considerRentalReturnConstraints: 'Zwróć pojazd współdzielony podczas podróży',
	default: 'Domyślne',
	timetableSources: 'Źródła danych rozkładowych',
	tripCancelled: 'Kurs odwołany',
	stopCancelled: 'Przystanek nieobsługiwany',
	inOutDisallowed: 'Zabronione wejście i wyjście',
	inDisallowed: 'Zabronione wejście',
	outDisallowed: 'Zabronione wyjście',
	unscheduledTrip: 'Kurs dodatkowy',
	alertsAvailable: 'Istnieją ogłoszenia',
	dataExpiredSince: 'Uwaga: Dane nieaktualne, ostatnio ważne',
	FLEX: 'Transport na żądanie',
	WALK: 'Pieszo',
	BIKE: 'Rower',
	RENTAL: 'Współdzielenie pojazdów',
	RIDE_SHARING: 'Wspólne przejazdy',
	CAR: 'Samochód',
	CAR_PARKING: 'Samochód (użyj parkingów)',
	CAR_DROPOFF: 'Samochód (tylko zatrzymanie)',
	TRANSIT: 'Transport publiczny',
	TRAM: 'Tramwaj',
	SUBWAY: 'Metro',
	FERRY: 'Prom',
	AIRPLANE: 'Samolot',
	SUBURBAN: 'Kolej miejska',
	BUS: 'Autobus',
	COACH: 'Autokar dalekobieżny',
	RAIL: 'Kolej',
	HIGHSPEED_RAIL: 'Kolej dużych prędkości',
	LONG_DISTANCE: 'Kolej dalekobieżna',
	NIGHT_RAIL: 'Nocne pociągi',
	REGIONAL_FAST_RAIL: 'Pociąg pospieszny',
	REGIONAL_RAIL: 'Kolej regionalna',
	OTHER: 'Inne',
	routingSegments: {
		maxTransfers: 'Maks. ilość przesiadek',
		maxTravelTime: 'Maks. czas podróży',
		firstMile: 'Początek podróży',
		lastMile: 'Osiągnięcie celu',
		direct: 'Połączenie bezpośrednie',
		maxPreTransitTime: 'Maks. czas dotarcia',
		maxPostTransitTime: 'Maks. czas dotarcia',
		maxDirectTime: 'Maks. czas dotarcia'
	},
	elevationCosts: {
		NONE: 'Bez odchyleń od trasy',
		LOW: 'Małe odchylenia od trasy',
		HIGH: 'Duże odchylenia od trasy'
	},
	isochrones: {
		title: 'Izochrony',
		displayLevel: 'Poziom wyświetlania',
		maxComputeLevel: 'Maks. poziom wyliczenia',
		canvasRects: 'Kwadraty (warstwa)',
		canvasCircles: 'Okręgi (warstwa)',
		geojsonCircles: 'Okręgi (geometria)',
		styling: 'Styl izochron',
		noData: 'Brak danych',
		requestFailed: 'Błąd zapytania'
	},
	alerts: {
		validFrom: 'Ważne od',
		until: 'do',
		information: 'Informacje',
		more: 'więcej'
	},
	RENTAL_BICYCLE: 'Rower współdzielony',
	RENTAL_CARGO_BICYCLE: 'Rower cargo współdzielony',
	RENTAL_CAR: 'Samochód współdzielony',
	RENTAL_MOPED: 'Skuter współdzielony',
	RENTAL_SCOOTER_STANDING: 'Hulajnoga stojąca współdzielona',
	RENTAL_SCOOTER_SEATED: 'Hulajnoga z siedziskiem współdzielona',
	RENTAL_OTHER: 'Inny pojazd współdzielony',
	incline: 'Nachylenie',
	CABLE_CAR: 'Kolej linowa',
	FUNICULAR: 'Kolej linowo-terenowa',
	AERIAL_LIFT: 'Wyciąg krzesełkowy',
	toll: 'Uwaga! Za przejazd tą trasą pobierana jest opłata.',
	accessRestriction: 'Ograniczony dostęp',
	continuesAs: 'Kontynuuje jako',
	rent: 'Wypożycz',
	copyToClipboard: 'Kopiuj do schowka',
	rideThroughAllowed: 'Przejazd dozwolony',
	rideThroughNotAllowed: 'Przejazd niedozwolony',
	rideEndAllowed: 'Parkowanie dozwolone',
	rideEndNotAllowed: 'Parkowanie tylko na stacjach',
	DEBUG_BUS_ROUTE: 'Trasa autobusu (Debug)',
	DEBUG_RAILWAY_ROUTE: 'Trasa kolejowa (Debug)',
	DEBUG_FERRY_ROUTE: 'Trasa promu (Debug)',
	routes: (n: number) => {
		switch (n) {
			case 0:
				return 'Brak trasy';
			case 1:
				return '1 trasa';
			default:
				return `${n} trasy`;
		}
	}
};

export default translations;
