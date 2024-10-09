import type { Mode } from './openapi';

export type LegLike = {
	routeShortName?: string;
	routeColor?: string;
	routeTextColor?: string;
	mode: Mode;
};

export const getModeStyle = (mode: Mode): [string, string, string] => {
	switch (mode) {
		case 'WALK':
		case 'FLEXIBLE':
			return ['walk', 'hsl(var(--foreground) / 1)', 'hsl(var(--background) / 1)'];

		case 'BIKE':
		case 'BIKE_TO_PARK':
		case 'BIKE_RENTAL':
		case 'SCOOTER_RENTAL':
			return ['bike', '#333', 'hsl(var(--foreground) / 1)'];

		case 'CAR':
		case 'CAR_TO_PARK':
		case 'CAR_HAILING':
		case 'CAR_SHARING':
		case 'CAR_PICKUP':
		case 'CAR_RENTAL':
			return ['car', '#333', 'hsl(var(--foreground) / 1)'];

		case 'TRANSIT':
		case 'BUS':
			return ['bus', '#ff9800', 'hsl(var(--foreground) / 1)'];
		case 'COACH':
			return ['bus', '#9ccc65', 'hsl(var(--foreground) / 1)'];

		case 'TRAM':
			return ['tram', '#ff9800', 'hsl(var(--foreground) / 1)'];

		case 'METRO':
			return ['sbahn', '#4caf50', 'hsl(var(--foreground) / 1)'];

		case 'SUBWAY':
			return ['ubahn', '#3f51b5', 'hsl(var(--foreground) / 1)'];

		case 'FERRY':
			return ['ferry', '#00acc1', 'hsl(var(--foreground) / 1)'];

		case 'AIRPLANE':
			return ['plane', '#90a4ae', 'hsl(var(--foreground) / 1)'];

		case 'HIGHSPEED_RAIL':
			return ['train', '#9c27b0', 'hsl(var(--foreground) / 1)'];

		case 'LONG_DISTANCE':
			return ['train', '#e91e63', 'hsl(var(--foreground) / 1)'];

		case 'NIGHT_RAIL':
			return ['train', '#1a237e', 'hsl(var(--foreground) / 1)'];

		case 'REGIONAL_FAST_RAIL':
		case 'REGIONAL_RAIL':
		case 'RAIL':
			return ['train', '#f44336', 'hsl(var(--foreground) / 1)'];
	}

	return ['train', '#000000', 'hsl(var(--foreground) / 1)'];
};

export const getColor = (l: LegLike): [string, string] => {
	const [_, defaultColor, defaultTextColor] = getModeStyle(l.mode);
	return !l.routeColor || l.routeColor === '000000'
		? [defaultColor, defaultTextColor]
		: ['#' + l.routeColor, '#' + l.routeTextColor];
};

export const routeBorderColor = (l: LegLike) => {
	return `border-color: ${getColor(l)[0]}`;
};

export const routeColor = (l: LegLike) => {
	const [color, textColor] = getColor(l);
	return `background-color: ${color}; color: ${textColor}; fill: ${textColor}`;
};
