import type { Translations } from './translation';

const translations: Translations = {
	journeyDetails: 'Szczegóły podróży',
	transfers: 'przesiadki',
	walk: 'Pieszo',
	bike: 'Rower',
	cargoBike: 'Rower cargo',
	scooterStanding: 'Hulajnoga stojąca',
	scooterSeated: 'Hulajnoga z siedziskiem',
	car: 'Samochód',
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
	noItinerariesFound: 'No itineraries found.'
};

export default translations;
