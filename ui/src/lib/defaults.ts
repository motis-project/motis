import { type PlanData } from '@motis-project/motis-client';

export const defaultQuery = {
	time: undefined,
	fromPlace: undefined,
	toPlace: undefined,
	via: undefined,
	viaMinimumStay: undefined,
	arriveBy: false,
	timetableView: true,
	withFares: false,
	numLegAlternatives: 0,
	searchWindow: 900,
	pedestrianProfile: 'FOOT',
	transitModes: ['TRANSIT'],
	preTransitModes: ['WALK'],
	postTransitModes: ['WALK'],
	directModes: ['WALK'],
	preTransitRentalFormFactors: [],
	postTransitRentalFormFactors: [],
	directRentalFormFactors: [],
	preTransitRentalProviderGroups: [],
	postTransitRentalProviderGroups: [],
	directRentalProviderGroups: [],
	preTransitRentalPropulsionTypes: [],
	postTransitRentalPropulsionTypes: [],
	directRentalPropulsionTypes: [],
	ignorePreTransitRentalReturnConstraints: false,
	ignorePostTransitRentalReturnConstraints: false,
	ignoreDirectRentalReturnConstraints: false,
	requireBikeTransport: false,
	requireCarTransport: false,
	noCompulsoryReservation: false,
	elevationCosts: 'NONE',
	vehicleHeight: 4.0,
	vehicleWidth: 2.55,
	vehicleLength: 18.75,
	vehicleWeight: 40.0,
	vehicleHazmat: false,
	vehicleHazmatWater: false,
	vehicleAxleCount: 5,
	vehicleAxleLoad: 11.5,
	vehicleTrailer: true,
	vehicleTopSpeed: 80,
	vehicleLezAccess: true,
	useRoutedTransfers: false,
	joinInterlinedLegs: true,
	maxMatchingDistance: 25,
	maxTransfers: 14,
	maxTravelTime: 300 * 60,
	maxPreTransitTime: 900,
	maxPostTransitTime: 900,
	maxDirectTime: 1800,
	pedestrianSpeed: 1.2,
	cyclingSpeed: 4.2,
	fastestDirectFactor: 10,
	additionalTransferTime: undefined,
	transferTimeFactor: 1,
	numItineraries: 5,
	maxItineraries: undefined,
	passengers: 1,
	luggage: 0,
	slowDirect: false,
	isochronesOpacity: 600,
	algorithm: 'PONG'
};

export const omitDefaults = (query: PlanData['query']): PlanData['query'] => {
	const queryCopy: PlanData['query'] = { ...query };
	Object.keys(queryCopy).forEach((key) => {
		if (key in defaultQuery) {
			const value = queryCopy[key as keyof PlanData['query']];
			const defaultValue = defaultQuery[key as keyof typeof defaultQuery];
			if (JSON.stringify(value) === JSON.stringify(defaultValue)) {
				delete queryCopy[key as keyof PlanData['query']];
			}
		} else {
			console.warn(`Unknown query parameter: ${key}`);
		}
	});
	return queryCopy;
};
