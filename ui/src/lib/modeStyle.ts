import type { Mode } from './openapi';

export type Colorable = {
	routeColor?: string;
	routeTextColor?: string;
	mode: Mode;
};

export type TripInfo = {
	tripId?: string;
	routeShortName?: string;
};

export type LegLike = Colorable & TripInfo;

export const getModeStyle = (mode: Mode): [string, string, string] => {
	switch (mode) {
		case 'WALK':
		case 'FLEXIBLE':
			return ['walk', 'hsl(var(--foreground) / 1)', 'hsl(var(--background) / 1)'];

		case 'BIKE':
		case 'BIKE_TO_PARK':
		case 'BIKE_RENTAL':
			return ['bike', '#075985', 'white'];

    case 'SCOOTER_RENTAL':
      return ['scooter', '#075985', 'white'];

		case 'CAR':
		case 'CAR_TO_PARK':
		case 'CAR_HAILING':
		case 'CAR_SHARING':
		case 'CAR_PICKUP':
		case 'CAR_RENTAL':
			return ['car', '#333', 'white'];

		case 'TRANSIT':
		case 'BUS':
			return ['bus', '#ff9800', 'white'];
		case 'COACH':
			return ['bus', '#9ccc65', 'white'];

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
	}

	return ['train', '#000000', 'white'];
};

export const getColor = (l: Colorable): [string, string] => {
	const [_, defaultColor, defaultTextColor] = getModeStyle(l.mode);
	return !l.routeColor || l.routeColor === '000000'
		? [defaultColor, defaultTextColor]
		: ['#' + l.routeColor, '#' + l.routeTextColor];
};

export const routeBorderColor = (l: Colorable) => {
	return `border-color: ${getColor(l)[0]}`;
};

export const routeColor = (l: Colorable) => {
	const [color, textColor] = getColor(l);
	return `background-color: ${color}; color: ${textColor}; fill: ${textColor}`;
};
