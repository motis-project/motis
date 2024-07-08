import { getModeStyle } from './modeStyle';
import type { Leg } from './openapi';

export const routeBorderColor = (l: Leg, defaultColor: string | undefined) => {
	const backgroundColor =
		!l.routeColor || l.routeColor === '000000' ? defaultColor || '000000' : l.routeColor;
	return `border-color: #${backgroundColor}`;
};

export const routeColor = (l: Leg) => {
	const [icon, defaultColor] = getModeStyle(l.mode);
	const backgroundColor =
		!l.routeColor || l.routeColor === '000000' ? defaultColor || '000000' : l.routeColor;
	const textColor = 'FFFFFF';
	return `background-color: #${backgroundColor}; color: #${textColor}`;
};
