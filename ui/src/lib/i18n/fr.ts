import type { Translations } from './translation';

const translations: Translations = {
	ticket: 'Billet',
	ticketOptions: 'Options de billet',
	includedInTicket: 'Inclus dans le billet',
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
	connections: 'Itinéraires',
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
	noItinerariesFound: 'Aucun itinéraire trouvé.',
	advancedSearchOptions: 'Options',
	selectModes: 'Sélectionner les modes de transport en commun',
	defaultSelectedModes: 'Tous les transports en commun',
	wheelchair: 'Correspondances accessibles',
	bikeRental: 'Utiliser véhicules partagés',
	bikeCarriage: 'Transport vélo',
	unreliableOptions: 'Selon la disponibilité des données, ces options peuvent ne pas être fiables.',
	timetableSources: 'Sources des horaires',
	tripCancelled: 'Voyage annulé',
	stopCancelled: 'Arrêt supprimé',
	inOutDisallowed: 'Impossible de monter/descendre',
	inDisallowed: 'Impossible de monter',
	outDisallowed: 'Impossible de descendre',
	unscheduledTrip: 'Voyage supplémentaire',
	WALK: 'À pied',
	BIKE: 'Vélo',
	RENTAL: 'Loué',
	CAR: 'Voiture',
	CAR_PARKING: 'Garage voiture',
	TRANSIT: 'Transports en commun',
	TRAM: 'Tram',
	SUBWAY: 'Métro',
	FERRY: 'Ferry',
	AIRPLANE: 'Avion',
	METRO: 'RER',
	BUS: 'Bus',
	COACH: 'Autocar',
	RAIL: 'Train',
	HIGHSPEED_RAIL: 'Train à grande vitesse',
	LONG_DISTANCE: 'Train intercité',
	NIGHT_RAIL: 'Train de nuit',
	REGIONAL_FAST_RAIL: 'Train express régional',
	REGIONAL_RAIL: 'Train régional',
	OTHER: 'Autres'
};

export default translations;
