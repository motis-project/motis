import type { PlanData } from '@motis-project/motis-client';

export const defaultQuery = {
	time: undefined,
	fromPlace: undefined,
	toPlace: undefined,
	via: undefined,
	viaMinimumStay: undefined,
	arriveBy: false,
	timetableView: true,
	withFares: false,
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
	elevationCosts: 'NONE',
	useRoutedTransfers: false,
	joinInterlinedLegs: true,
	maxMatchingDistance: 25,
	maxTransfers: 14,
	maxTravelTime: 30 * 60,
	maxPreTransitTime: 900,
	maxPostTransitTime: 900,
	maxDirectTime: 1800,
	fastestDirectFactor: 1.0,
	additionalTransferTime: 0,
	transferTimeFactor: 1,
	numItineraries: 5,
	circleResolution: undefined,
	maxItineraries: undefined,
	passengers: 1,
	luggage: 0,
	slowDirect: false,
	isochronesDisplayLevel: 'GEOMETRY_CIRCLES',
	isochronesColor: '#ffff00',
	isochronesOpacity: 250,
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
