import type { Mode } from './openapi';

import { t } from './i18n/translation';

export const getModeName = (m: Mode) => {
	switch (m) {
		case 'WALK':
			return t.walk;
		case 'BIKE':
		case 'BIKE_RENTAL':
			return t.bike;
		case 'SCOOTER_RENTAL':
			return t.scooter;
		case 'CAR':
		case 'CAR_PARKING':
			return t.car;
		default:
			return `${m}`;
	}
};
