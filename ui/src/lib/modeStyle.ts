import type { Mode, Rental } from './api/openapi';

export type Colorable = { routeColor?: string; routeTextColor?: string; mode: Mode };

export type TripInfo = { tripId?: string; displayName?: string };

export type RentalInfo = { rental?: Rental };

export type LegLike = Colorable & TripInfo & RentalInfo;

export const getModeStyle = (l: LegLike): [string, string, string] => {
	switch (l.mode) {
		case 'WALK':
			return ['walk', 'hsl(var(--foreground) / 1)', 'hsl(var(--background) / 1)'];
		case 'BIKE':
			return ['bike', 'hsl(var(--foreground) / 1)', 'hsl(var(--background) / 1)'];

		case 'RENTAL':
			switch (l.rental?.formFactor) {
				case 'BICYCLE':
					return ['bike', '#075985', 'white'];
				case 'CARGO_BICYCLE':
					return ['cargo_bike', '#075985', 'white'];
				case 'CAR':
					return ['car', '#4c4947', 'white'];
				case 'MOPED':
					return ['moped', '#075985', 'white'];
				case 'SCOOTER_SEATED':
				case 'SCOOTER_STANDING':
					return ['scooter', '#075985', 'white'];
				case 'OTHER':
				default:
					return ['bike', '#075985', 'white'];
			}

		case 'CAR':
		case 'CAR_PARKING':
			return ['car', '#4c4947', 'white'];

		case 'FLEX':
		case 'ODM':
			return ['taxi', '#fdb813', 'white'];

		case 'TRANSIT':
		case 'BUS':
			return ['bus', '#ff9800', 'white'];
		case 'COACH':
			return ['bus', '#9ccc65', 'black'];

		case 'TRAM':
			return ['tram', '#ff9800', 'white'];

		case 'METRO':
			return ['sbahn', '#4caf50', 'white'];

		case 'SUBWAY':
			return ['ubahn', '#3f51b5', 'white'];

		case 'FERRY':
			return ['ship', '#00acc1', 'white'];

		case 'AIRPLANE':
			return ['plane', '#90a4ae', 'white'];

		case 'HIGHSPEED_RAIL':
			return ['train', '#9c27b0', 'white'];

		case 'LONG_DISTANCE':
			return ['train', '#e91e63', 'white'];

		case 'NIGHT_RAIL':
			return ['train', '#1a237e', 'white'];

		case 'REGIONAL_FAST_RAIL':
		case 'REGIONAL_RAIL':
		case 'RAIL':
			return ['train', '#f44336', 'white'];

		case 'FUNICULAR':
			return ['funicular', '#795548', 'white'];

		case 'CABLE_CAR':
		case 'AREAL_LIFT':
			return ['aerial_lift', '#795548', 'white'];
	}

	return ['train', '#000000', 'white'];
};

export const getColor = (l: Colorable): [string, string] => {
	const [_, defaultColor, defaultTextColor] = getModeStyle(l);
	return !l.routeColor || l.routeColor === '000000'
		? [defaultColor, defaultTextColor]
		: ['#' + l.routeColor, '#' + l.routeTextColor || '000000'];
};

export const routeBorderColor = (l: Colorable) => {
	return `border-color: ${getColor(l)[0]}`;
};

export const routeColor = (l: Colorable) => {
	const [color, textColor] = getColor(l);
	return `background-color: ${color}; color: ${textColor}; fill: ${textColor}`;
};
