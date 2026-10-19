import type { Mode, Rental } from '@motis-project/motis-client';

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
					return ['seated_scooter', '#075985', 'white'];
				case 'SCOOTER_STANDING':
					return ['scooter', '#075985', 'white'];
				case 'OTHER':
				default:
					return ['other', '#075985', 'white'];
			}

		case 'RIDE_SHARING':
			return ['car', '#217edb', 'white'];

		case 'CAR':
		case 'CAR_PARKING':
			return ['car', '#4c4947', 'white'];

		case 'HGV':
			return ['truck', '#2563eb', 'white'];

		case 'FLEX':
		case 'ODM':
			return ['taxi', '#fdb813', 'white'];

		case 'TRANSIT':
		case 'BUS':
			return ['bus', '#ff9800', 'white'];
		case 'COACH':
			return ['bus', '#9ccc65', 'black'];

		case 'TRAM':
			return ['tram', '#edce00', 'white'];

		case 'SUBURBAN':
			return ['train', '#4caf50', 'white'];

		case 'SUBWAY':
			return ['metro', '#3f51b5', 'white'];

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
			return ['tram', '#795548', 'white'];

		case 'AERIAL_LIFT':
			return ['aerial_lift', '#795548', 'white'];
	}

	return ['train', '#000000', 'white'];
};

// Black or white, whichever reads better on the given hex background
// (with or without '#'). White for anything that is not a hex color.
const contrastTextColor = (hex: string): string => {
	const m = /^#?([0-9a-f]{3}|[0-9a-f]{6})$/i.exec(hex.trim());
	if (!m) {
		return 'white';
	}
	const h = m[1].length === 3 ? [...m[1]].map((c) => c + c).join('') : m[1];
	const [r, g, b] = [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16) / 255);
	const lin = (c: number) => (c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
	const luminance = 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b);
	return luminance > 0.4 ? 'black' : 'white';
};

// Background and text color: the GTFS route colors, else the GBFS provider's
// brand color (it comes without a text color), else the mode's default.
export const getColor = (l: Colorable & RentalInfo): [string, string] => {
	const [_, defaultColor, defaultTextColor] = getModeStyle(l);
	if (l.routeColor) {
		return [
			`#${l.routeColor}`,
			l.routeTextColor ? `#${l.routeTextColor}` : contrastTextColor(l.routeColor)
		];
	}
	if (l.rental?.color) {
		return [l.rental.color, contrastTextColor(l.rental.color)];
	}
	return [defaultColor, defaultTextColor];
};

export const routeBorderColor = (l: Colorable & RentalInfo) => {
	return `border-color: ${getColor(l)[0]}`;
};

export const routeColor = (l: Colorable & RentalInfo) => {
	const [color, textColor] = getColor(l);
	return `background-color: ${color}; color: ${textColor}; fill: ${textColor}`;
};
