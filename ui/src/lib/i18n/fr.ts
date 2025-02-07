import type { Translations } from './translation';

const translations: Translations = {
	journeyDetails: 'Détails du voyage',
	walk: 'à pied',
	bike: 'Vélo',
	cargoBike: 'Vélo Cargo',
	scooterStanding: 'Trottinette',
	scooterSeated: 'Trottinette avec siège',
	car: 'Voiture',
	taxi: 'Taxi',
	moped: 'Mobylette',
	from: 'De',
	to: 'À',
	arrival: 'Arrivée',
	departure: 'Départ',
	duration: 'Durée',
	arrivals: 'Arrivées',
	later: 'plus tard',
	earlier: 'plus tôt',
	departures: 'Départs',
	switchToArrivals: 'Afficher les arrivées',
	switchToDepartures: 'Afficher les départs',
	track: 'Voie',
	arrivalOnTrack: 'Arrivée sur la voie',
	tripIntermediateStops: (n: number) => {
		switch (n) {
			case 0:
				return 'Aucun arrêt intermédiaire';
			case 1:
				return '1 arrêt intermédiaire';
			default:
				return `${n} arrêts intermédiaires`;
		}
	},
	sharingProvider: 'Fournisseur',
	transfers: 'correspondances',
	roundtripStationReturnConstraint: 'Le véhicule doit être retourné à la station de départ.',
	noItinerariesFound: 'Aucun itinéraire trouvé.'
};

export default translations;
