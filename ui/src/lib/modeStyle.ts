import type { Leg, Mode } from './openapi';

export const getModeStyle = (mode: Mode): [string, string] => {
	switch (mode) {
		case 'WALK':
		case 'FLEXIBLE':
			return ['walk', 'hsl(var(--foreground) / 1)'];

		case 'BIKE':
		case 'BIKE_TO_PARK':
		case 'BIKE_RENTAL':
		case 'SCOOTER_RENTAL':
			return ['bike', '#333'];

		case 'CAR':
		case 'CAR_TO_PARK':
		case 'CAR_HAILING':
		case 'CAR_SHARING':
		case 'CAR_PICKUP':
		case 'CAR_RENTAL':
			return ['car', '#333'];

		case 'TRANSIT':
		case 'BUS':
			return ['bus', '#ff9800'];
		case 'COACH':
			return ['bus', '#9ccc65'];

		case 'TRAM':
			return ['tram', '#ff9800'];

		case 'METRO':
			return ['sbahn', '#4caf50'];

		case 'SUBWAY':
			return ['ubahn', '#3f51b5'];

		case 'FERRY':
			return ['ferry', '#00acc1'];

		case 'AIRPLANE':
			return ['plane', '#90a4ae'];

		case 'HIGHSPEED_RAIL':
			return ['train', '#9c27b0'];

		case 'LONG_DISTANCE':
			return ['train', '#e91e63'];

		case 'NIGHT_RAIL':
			return ['train', '#1a237e'];

		case 'REGIONAL_FAST_RAIL':
		case 'REGIONAL_RAIL':
		case 'RAIL':
			return ['train', '#f44336'];
	}

	return ['train', '#000000'];
};

export const getColor = (l: Leg): string => {
	const defaultColor = getModeStyle(l.mode)[1];
	return !l.routeColor || l.routeColor === '000000' ? defaultColor : '#' + l.routeColor;
};

export const routeBorderColor = (l: Leg) => {
	return `border-color: ${getColor(l)}`;
};

export const routeColor = (l: Leg) => {
	return `background-color: ${getColor(l)}; color: #FFF;`;
};
