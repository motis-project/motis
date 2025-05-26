import type { PlanData } from './api/openapi';

export const defaultQuery = {
	arriveBy: false,
	timetableView: true,
	withFares: false,
	pedestrianProfile: 'FOOT',
	transitModes: ['TRANSIT'],
	preTransitModes: ['WALK'],
	postTransitModes: ['WALK'],
	directModes: ['WALK'],
	preTransitRentalFormFactors: [],
	postTransitRentalFormFactors: [],
	directRentalFormFactors: [],
	preTransitRentalPropulsionTypes: [],
	postTransitRentalPropulsionTypes: [],
	directRentalPropulsionTypes: [],
	requireBikeTransport: false,
	requireCarTransport: false,
	elevationCosts: 'NONE',
	useRoutedTransfers: true,
	maxMatchingDistance: 250,
	maxPreTransitTime: 900,
	maxPostTransitTime: 900,
	maxDirectTime: 1800,
	fastestDirectFactor: 1.0,
	additionalTransferTime: 0,
	transferTimeFactor: 1,
	numItineraries: 5,
	passengers: 1,
	luggage: 0
};

export const omitDefaults = (query: PlanData['query']): PlanData['query'] => {
	const queryCopy: PlanData['query'] = { ...query };
	Object.keys(defaultQuery).forEach((key) => {
		const value = queryCopy[key as keyof PlanData['query']];
		const defaultValue = defaultQuery[key as keyof typeof defaultQuery];
		if (JSON.stringify(value) === JSON.stringify(defaultValue)) {
			delete queryCopy[key as keyof PlanData['query']];
		}
	});
	return queryCopy;
};
