import { getModeStyle } from './modeStyle';
import type { Leg } from './openapi';

export const getColor = (l: Leg): string => {
	const defaultColor = getModeStyle(l.mode)[1];
	return !l.routeColor || l.routeColor === '000000' ? defaultColor || '000000' : l.routeColor;
};

export const routeBorderColor = (l: Leg) => {
	return `border-color: #${getColor(l)}`;
};

export const routeColor = (l: Leg) => {
	return `background-color: #${getColor(l)}; color: #FFF;`;
};
