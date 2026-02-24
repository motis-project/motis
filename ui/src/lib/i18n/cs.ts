import type { Translations } from './translation';

const translations: Translations = {
	ticket: 'Jízdenka',
	ticketOptions: 'Možnosti jízdenky',
	includedInTicket: 'Zahrnuté v jízdence',
	journeyDetails: 'Detail cesty',
	transfers: 'přestupy',
	walk: 'Pěšky',
	bike: 'Kolo',
	cargoBike: 'Nákladní kolo',
	scooterStanding: 'Koloběžka',
	scooterSeated: 'Koloběžka se sedačkou',
	car: 'Auto',
	taxi: 'Taxi',
	moped: 'Skútr',
	unknownVehicleType: 'Neznámý typ vozidla',
	electricAssist: 'Elektrická podpora',
	electric: 'Elektrické',
	combustion: 'Spalovací',
	combustionDiesel: 'Diesel',
	hybrid: 'Hybridní',
	plugInHybrid: 'Plug-in hybrid',
	hydrogenFuelCell: 'Vodíkový palivový článek',
	from: 'Z',
	to: 'Do',
	viaStop: 'Mezizastávka',
	viaStops: 'Mezizastávky',
	addViaStop: 'Přidat mezizastávku',
	removeViaStop: 'Odebrat mezizastávku',
	viaStayDuration: 'Minimální pobyt',
	position: 'Pozice',
	arrival: 'Příjezd',
	departure: 'Odjezd',
	duration: 'Čas cesty',
	arrivals: 'Příjezdy',
	later: 'později',
	earlier: 'dřive',
	departures: 'Odjezdy',
	connections: 'Spoje',
	switchToArrivals: 'Přepni na příjezdy',
	switchToDepartures: 'Přepni na odjezdy',
	track: 'Kolej',
	platform: 'Nástupiště',
	platformAbr: 'Nást.',
	trackAbr: 'K.',
	arrivalOnTrack: 'Příjezd na kolej',
	tripIntermediateStops: (n: number) => {
		if (n == 0) {
			return 'Bez mezizastávek';
		}
		if (n == 1) {
			return '1 mezizastávka';
		}
		if (n % 10 > 1 && n % 10 < 5 && n != 12 && n != 13 && n != 14) {
			return `${n} mezizastávky`;
		}
		return `${n} mezizastávek`;
	},
	sharingProvider: 'Poskytovatel dat',
	sharingProviders: 'Poskytovatelé dat',
	returnOnlyAtStations: 'Vozidlo musí být vráceno na stanici.',
	roundtripStationReturnConstraint: 'Pojezd musí být vrácen k počáteční stanice',
	rentalStation: 'Stanice',
	rentalGeofencingZone: 'Zóna',
	noItinerariesFound: 'Spojení nebylo nalezeno.',
	advancedSearchOptions: 'Možnosti',
	selectTransitModes: 'Vyber dopravní prostředky',
	defaultSelectedModes: 'Všechny dopravní prostředky',
	defaultSelectedProviders: 'Všichni poskytovatelé',
	selectElevationCosts: 'Bez prudkého stoupání.',
	wheelchair: 'Bezbariérové přestupy',
	useRoutedTransfers: 'Počítej trasu pro přestupy',
	bikeRental: 'Povol použití sdílených vozidel',
	requireBikeTransport: 'Přeprava kola',
	requireCarTransport: 'Přeprava auta',
	considerRentalReturnConstraints: 'Vrať sdílené vozidla během cesty',
	default: 'default',
	timetableSources: 'Zdroje dát JŘ',
	tripCancelled: 'Spoj odřeknut',
	stopCancelled: 'Zastávka bez obsluhy',
	inOutDisallowed: 'Vstup/výstup není povolen',
	inDisallowed: 'Vstup není povolen',
	outDisallowed: 'Výstup není povolen',
	unscheduledTrip: 'Doplňkový spoj',
	alertsAvailable: 'Oznámení o provozu',
	dataExpiredSince: 'Pozor: Zastaralá data, platná naposledy',
	FLEX: 'Poptávková doprava',
	WALK: 'Chůze',
	BIKE: 'Kolo',
	RENTAL: 'Sdílené prostředky',
	RIDE_SHARING: 'Spolujízda',
	CAR: 'Auto',
	CAR_PARKING: 'Auto (využití parkovíšť)',
	CAR_DROPOFF: 'Auto (pouze zastavení)',
	TRANSIT: 'Hromadná doprava',
	TRAM: 'Tramvaj',
	SUBWAY: 'Metro',
	FERRY: 'Přívoz',
	AIRPLANE: 'Letadlo',
	SUBURBAN: 'Městská železnice',
	BUS: 'Autobus',
	COACH: 'Dálkový autokar',
	RAIL: 'Železnice',
	HIGHSPEED_RAIL: 'Vysokorychlostní železnice',
	LONG_DISTANCE: 'Dálková železnice',
	NIGHT_RAIL: 'Noční vlaky',
	REGIONAL_FAST_RAIL: 'Zrychlená železnice',
	REGIONAL_RAIL: 'Regionální železnice',
	OTHER: 'Jiné',
	routingSegments: {
		maxTransfers: 'Max. počet přestupů',
		maxTravelTime: 'Max. čas cesty',
		firstMile: 'Přesun k první zastávce',
		lastMile: 'Přesun od poslední zastávky',
		direct: 'Přímé spojení',
		maxPreTransitTime: 'Max. čas přesunu',
		maxPostTransitTime: 'Max. čas přesunu',
		maxDirectTime: 'Max. čas přesunu'
	},
	elevationCosts: {
		NONE: 'Bez odklonů',
		LOW: 'Malé odklony',
		HIGH: 'Velké odklony'
	},
	isochrones: {
		title: 'Izochrony',
		displayLevel: 'Úroveň ukazování',
		maxComputeLevel: 'Max. úroveň vypočítání',
		canvasRects: 'Čtverce',
		canvasCircles: 'Okruhy (zjednodušená projekce)',
		geojsonCircles: 'Okruhy (pokročilá projekce)',
		styling: 'Styl izochron',
		noData: 'Žádné data',
		requestFailed: 'Chyba žádosti'
	},
	alerts: {
		validFrom: 'Platí od',
		until: 'do',
		information: 'Informace',
		more: 'více'
	},
	RENTAL_BICYCLE: 'Sdílené kolo',
	RENTAL_CARGO_BICYCLE: 'Sdílené nákladní kolo',
	RENTAL_CAR: 'Sdílené auto',
	RENTAL_MOPED: 'Sdílený skútr',
	RENTAL_SCOOTER_STANDING: 'Sdílená koloběžka',
	RENTAL_SCOOTER_SEATED: 'Sdílená koloběžka se sedačkou',
	RENTAL_OTHER: 'Jiné sdílené vozidla',
	incline: 'Sklon',
	CABLE_CAR: 'Lanová dráha',
	FUNICULAR: 'Lanová dráha',
	AERIAL_LIFT: 'Lanová dráha',
	toll: 'Pozor! Průjezd tuto trasou je placený.',
	accessRestriction: 'Omezený dostup',
	continuesAs: 'Pokračuje jako',
	rent: 'Půjčit si',
	copyToClipboard: 'Kopírovat do schránky',
	rideThroughAllowed: 'Průjezd povolen',
	rideThroughNotAllowed: 'Průjezd zakázán',
	rideEndAllowed: 'Parkování povoleno',
	rideEndNotAllowed: 'Parkování pouze na stanicích',
	DEBUG_BUS_ROUTE: 'Trasa autobusu (Debug)',
	DEBUG_RAILWAY_ROUTE: 'Trasa vlaku (Debug)',
	DEBUG_FERRY_ROUTE: 'Trasa trajektu (Debug)',
	routes: (n: number) => {
		switch (n) {
			case 0:
				return 'Žádná trasa';
			case 1:
				return '1 trasa';
			case 2:
			case 3:
			case 4:
				return `${n} trasy`;
			default:
				return `${n} tras`;
		}
	}
};

export default translations;
