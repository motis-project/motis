import type { Mode, RentalFormFactor } from '@motis-project/motis-client';

export const prePostDirectModes = [
	'WALK',
	'BIKE',
	'CAR',
	'FLEX',
	'CAR_DROPOFF',
	'CAR_PARKING',
	'RENTAL_BICYCLE',
	'RENTAL_CARGO_BICYCLE',
	'RENTAL_CAR',
	'RENTAL_MOPED',
	'RENTAL_SCOOTER_STANDING',
	'RENTAL_SCOOTER_SEATED',
	'RENTAL_OTHER'
] as const;
export type PrePostDirectMode = (typeof prePostDirectModes)[number];

export const getPrePostDirectModes = (
	modes: Mode[],
	formFactors: RentalFormFactor[]
): PrePostDirectMode[] => {
	return modes
		.filter((mode) => prePostDirectModes.includes(mode as PrePostDirectMode))
		.map((mode) => mode as PrePostDirectMode)
		.concat(formFactors.map((formFactor) => `RENTAL_${formFactor}` as PrePostDirectMode));
};

export const getFormFactors = (modes: PrePostDirectMode[]): RentalFormFactor[] => {
	return modes
		.filter((mode) => mode.startsWith('RENTAL_'))
		.map((mode) => mode.replace('RENTAL_', '')) as RentalFormFactor[];
};

export const prePostModesToModes = (modes: PrePostDirectMode[]): Mode[] => {
	const rentalMode: Mode[] = modes.some((mode) => mode.startsWith('RENTAL_')) ? ['RENTAL'] : [];
	const nonRentalModes = modes.filter((mode) => !mode.startsWith('RENTAL_'));
	return [...nonRentalModes, ...rentalMode].map((mode) => mode as Mode);
};

export const possibleTransitModes = [
	'AIRPLANE',
	'HIGHSPEED_RAIL',
	'LONG_DISTANCE',
	'NIGHT_RAIL',
	'COACH',
	'REGIONAL_FAST_RAIL',
	'REGIONAL_RAIL',
	'SUBURBAN',
	'SUBWAY',
	'TRAM',
	'BUS',
	'FERRY',
	'ODM',
	'FUNICULAR',
	'AERIAL_LIFT',
	'OTHER'
];
export type TransitMode = (typeof possibleTransitModes)[number];
